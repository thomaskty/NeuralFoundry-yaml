# app/services/kb_ingestion_service.py
import os
import aiofiles
import hashlib
import asyncio
from datetime import datetime, timezone
from sqlalchemy.future import select
from app.db.database import AsyncSessionLocal
from app.db.models import KnowledgeBase, KBDocument, GlobalDocument, GlobalChunk, KBChunkLink
from app.services.wrappers.async_embedding import get_batch_embeddings_async
from app.services.ingestion.document_processor import DocumentProcessor  # NEW
import logging

# Initialize document processor (singleton)
_document_processor = DocumentProcessor()
logger = logging.getLogger(__name__)


async def process_kb_file(kb_id, file_path, original_filename, replace_if_changed: bool = False):
    """
    Process uploaded file using Docling-based document processor.
    Supports multiple formats: PDF, Word, images, HTML, etc.

    Args:
        kb_id: Knowledge base UUID
        file_path: Temporary file path (will be deleted after processing)
        original_filename: Original filename from user upload
    """
    async with AsyncSessionLocal() as db:
        try:
            # 1. Verify KB exists
            kb_result = await db.execute(
                select(KnowledgeBase).where(KnowledgeBase.id == kb_id)
            )
            kb = kb_result.scalars().first()

            if not kb:
                raise ValueError(f"KnowledgeBase {kb_id} not found")

            # 2. Read file metadata
            file_size = os.path.getsize(file_path)

            # For text files, compute hash; for others, use filename
            try:
                async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
                    content = await f.read()
                text_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
            except:
                # Binary file (PDF, image, etc.)
                text_hash = hashlib.sha256(original_filename.encode()).hexdigest()

            mime_type = _detect_mime_type(original_filename)

            # 3. Check existing document in this KB by hash
            existing_same_hash = await db.execute(
                select(KBDocument).join(GlobalDocument).where(
                    KBDocument.kb_id == kb_id,
                    GlobalDocument.text_sha256 == text_hash
                )
            )
            existing_doc = existing_same_hash.scalars().first()
            if existing_doc:
                link_count = await db.execute(
                    select(KBChunkLink).where(KBChunkLink.document_id == existing_doc.id)
                )
                if link_count.scalars().first():
                    logger.info(f"Skipping {original_filename} (already chunked in this KB)")
                    return {
                        "kb_id": kb_id,
                        "document_id": str(existing_doc.id),
                        "chunks_added": 0,
                        "status": "skipped",
                        "message": f"Document '{original_filename}' already exists in KB."
                    }

            # 4. Check same filename in this KB (changed content)
            existing_same_name = await db.execute(
                select(KBDocument, GlobalDocument.text_sha256).join(GlobalDocument).where(
                    KBDocument.kb_id == kb_id,
                    GlobalDocument.filename == original_filename
                )
            )
            existing_name_rows = existing_same_name.all()
            if existing_name_rows:
                for doc, existing_hash in existing_name_rows:
                    if existing_hash != text_hash:
                        if replace_if_changed:
                            logger.info(f"Replacing {original_filename} (content changed)")
                            for d, _ in existing_name_rows:
                                await db.delete(d)
                            await db.flush()
                            break
                        else:
                            logger.info(f"Skipping {original_filename} (changed; set replace_if_changed to update)")
                            return {
                                "kb_id": kb_id,
                                "document_id": str(doc.id),
                                "chunks_added": 0,
                                "status": "changed_skip",
                                "message": f"Document '{original_filename}' changed but not updated."
                            }

            # 5. Global cache lookup
            global_doc_result = await db.execute(
                select(GlobalDocument).where(GlobalDocument.text_sha256 == text_hash)
            )
            global_doc = global_doc_result.scalars().first()

            if global_doc:
                # Reuse cached chunks
                logger.info(f"Reusing cached chunks for {original_filename}")

                kb_document = KBDocument(
                    kb_id=kb_id,
                    global_document_id=global_doc.id,
                    uploaded_by=kb.user_id,
                    created_at=datetime.now(timezone.utc),
                )
                db.add(kb_document)
                await db.flush()
                kb_document_id = str(kb_document.id)

                global_chunks = await db.execute(
                    select(GlobalChunk).where(GlobalChunk.document_id == global_doc.id).order_by(GlobalChunk.chunk_index)
                )
                global_chunks = global_chunks.scalars().all()

                kb_chunk_links = []
                for gc in global_chunks:
                    kb_chunk_links.append(
                        KBChunkLink(
                            kb_id=kb_id,
                            document_id=kb_document.id,
                            global_chunk_id=gc.id,
                            created_at=datetime.now(timezone.utc),
                        )
                    )
                db.add_all(kb_chunk_links)
                await db.commit()

                if os.path.exists(file_path):
                    os.remove(file_path)

                return {
                    "kb_id": kb_id,
                    "document_id": kb_document_id,
                    "chunks_added": len(kb_chunk_links),
                    "status": "reused",
                    "message": f"Document '{original_filename}' reused from global cache."
                }

            # 6. Create KBDocument entry
            # 6. Process file using Docling
            logger.info(f"Processing {original_filename} with Docling...")
            chunks = await _document_processor.process_file(file_path)
            logger.info(f"Extracted {len(chunks)} chunks from {original_filename}")

            if not chunks:
                raise ValueError(f"No content could be extracted from {original_filename}")

            chunk_texts = [chunk['text'] for chunk in chunks]

            logger.info(f"Generating embeddings for {len(chunk_texts)} chunks...")
            embeddings = await get_batch_embeddings_async(chunk_texts)

            # 7. Store global cache + KB links
            global_doc = GlobalDocument(
                filename=original_filename,
                mime_type=mime_type,
                text_sha256=text_hash,
                text_size=file_size,
                doc_metadata={"source": original_filename},
                created_at=datetime.now(timezone.utc),
            )
            db.add(global_doc)
            await db.flush()

            kb_document = KBDocument(
                kb_id=kb_id,
                global_document_id=global_doc.id,
                uploaded_by=kb.user_id,
                created_at=datetime.now(timezone.utc),
            )
            db.add(kb_document)
            await db.flush()
            kb_document_id = str(kb_document.id)

            global_chunk_objects = []
            kb_chunk_links = []
            for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
                emb_list = emb.tolist()
                global_chunk_objects.append(
                    GlobalChunk(
                        document_id=global_doc.id,
                        chunk_index=i,
                        text=chunk['text'],
                        token_count=len(chunk['text'].split()),
                        embedding=emb_list,
                        chunk_metadata=chunk['metadata'],
                        created_at=datetime.now(timezone.utc),
                    )
                )
            db.add_all(global_chunk_objects)
            await db.flush()

            for gc in global_chunk_objects:
                kb_chunk_links.append(
                    KBChunkLink(
                        kb_id=kb_id,
                        document_id=kb_document.id,
                        global_chunk_id=gc.id,
                        created_at=datetime.now(timezone.utc),
                    )
                )
            db.add_all(kb_chunk_links)
            await db.commit()

            # 8. Clean up temp file
            if os.path.exists(file_path):
                os.remove(file_path)

            logger.info(f"Successfully processed {original_filename}: {len(kb_chunk_links)} chunks")

            return {
                "kb_id": kb_id,
                "document_id": kb_document_id,
                "chunks_added": len(kb_chunk_links),
                "status": "ingested",
                "message": f"Document '{original_filename}' processed successfully."
            }

        except Exception as e:
            await db.rollback()

            # Clean up temp file on error
            if os.path.exists(file_path):
                os.remove(file_path)

            # Log the error
            logger.error(f"Error processing {original_filename}: {e}")
            import traceback
            traceback.print_exc()

            raise e


def _detect_mime_type(filename: str) -> str:
    """Detect MIME type from filename extension"""
    ext = os.path.splitext(filename)[1].lower()

    mime_types = {
        '.pdf': 'application/pdf',
        '.txt': 'text/plain',
        '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        '.doc': 'application/msword',
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.html': 'text/html',
        '.htm': 'text/html',
        '.md': 'text/markdown',
    }

    return mime_types.get(ext, 'application/octet-stream')
