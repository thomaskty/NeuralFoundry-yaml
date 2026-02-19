# app/services/kb_ingestion_service.py
import os
import aiofiles
import hashlib
import asyncio
from datetime import datetime, timezone
from sqlalchemy.future import select
from app.db.database import AsyncSessionLocal
from app.db.models import KnowledgeBase, KBDocument, KBChunk
from app.services.wrappers.async_embedding import get_batch_embeddings_async
from app.services.ingestion.document_processor import DocumentProcessor  # NEW

# Initialize document processor (singleton)
_document_processor = DocumentProcessor()


async def process_kb_file(kb_id, file_path, original_filename):
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

            # 3. Create KBDocument entry
            document = KBDocument(
                kb_id=kb_id,
                uploaded_by=kb.user_id,
                filename=original_filename,
                mime_type=_detect_mime_type(original_filename),
                text_sha256=text_hash,
                text_size=file_size,
                doc_metadata={"source": original_filename},
                created_at=datetime.now(timezone.utc),
            )
            db.add(document)
            await db.flush()

            # Capture document ID before commit
            document_id = document.id
            document_filename = document.filename

            # 4. Process file using Docling (intelligent chunking)
            print(f"Processing {original_filename} with Docling...")
            chunks = await _document_processor.process_file(file_path)
            print(f"Extracted {len(chunks)} chunks from {original_filename}")

            if not chunks:
                raise ValueError(f"No content could be extracted from {original_filename}")

            # 5. Extract text from chunks
            chunk_texts = [chunk['text'] for chunk in chunks]

            # 6. Generate batch embeddings using OpenAI
            print(f"Generating embeddings for {len(chunk_texts)} chunks...")
            embeddings = await get_batch_embeddings_async(chunk_texts)

            # 7. Store each chunk with metadata
            kb_chunk_objects = []
            for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
                kb_chunk_objects.append(
                    KBChunk(
                        kb_id=kb_id,
                        document_id=document_id,
                        chunk_index=i,
                        text=chunk['text'],
                        token_count=len(chunk['text'].split()),
                        embedding=emb.tolist(),
                        chunk_metadata=chunk['metadata'],  # Store Docling metadata
                        created_at=datetime.now(timezone.utc),
                    )
                )

            db.add_all(kb_chunk_objects)
            await db.commit()

            # 8. Clean up temp file
            if os.path.exists(file_path):
                os.remove(file_path)

            print(f"✅ Successfully processed {document_filename}: {len(kb_chunk_objects)} chunks")

            return {
                "kb_id": kb_id,
                "document_id": str(document_id),
                "chunks_added": len(kb_chunk_objects),
                "message": f"Document '{document_filename}' processed successfully."
            }

        except Exception as e:
            await db.rollback()

            # Clean up temp file on error
            if os.path.exists(file_path):
                os.remove(file_path)

            # Log the error
            print(f"❌ Error processing {original_filename}: {e}")
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