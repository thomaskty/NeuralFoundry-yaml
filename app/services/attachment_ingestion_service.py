# app/services/attachment_ingestion_service.py
import os
import aiofiles
from datetime import datetime, timezone
from sqlalchemy.future import select
from app.db.database import AsyncSessionLocal
from app.db.models import ChatAttachment, ChatAttachmentChunk
from app.services.wrappers.async_embedding import get_batch_embeddings_async
from app.services.ingestion.document_processor import DocumentProcessor

_document_processor = DocumentProcessor()


async def process_chat_attachment(chat_id: str, file_path: str, original_filename: str, user_id: str):
    """
    Process uploaded chat attachment using Docling.
    Similar to KB processing but for temporary chat files.
    """
    async with AsyncSessionLocal() as db:
        try:
            # 1. Get file metadata
            file_size = os.path.getsize(file_path)
            mime_type = _detect_mime_type(original_filename)

            # 2. Create ChatAttachment entry
            attachment = ChatAttachment(
                chat_id=chat_id,
                user_id=user_id,
                filename=original_filename,
                mime_type=mime_type,
                file_size=file_size,
                processing_status="processing",
                uploaded_at=datetime.now(timezone.utc),
            )
            db.add(attachment)
            await db.flush()

            attachment_id = attachment.id

            # 3. Process file with Docling
            print(f"Processing chat attachment: {original_filename}")
            chunks = await _document_processor.process_file(file_path)
            print(f"Extracted {len(chunks)} chunks from attachment")

            if not chunks:
                attachment.processing_status = "failed"
                await db.commit()
                raise ValueError(f"No content extracted from {original_filename}")

            # 4. Generate embeddings
            chunk_texts = [chunk['text'] for chunk in chunks]
            embeddings = await get_batch_embeddings_async(chunk_texts)

            # 5. Store chunks
            chunk_objects = []
            for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
                chunk_objects.append(
                    ChatAttachmentChunk(
                        attachment_id=attachment_id,
                        chat_id=chat_id,
                        chunk_index=i,
                        text=chunk['text'],
                        token_count=len(chunk['text'].split()),
                        embedding=emb.tolist(),
                        chunk_metadata=chunk['metadata'],  # ← CHANGED
                        created_at=datetime.now(timezone.utc),
                    )
                )

            db.add_all(chunk_objects)

            # 6. Update attachment status
            attachment.total_chunks = len(chunk_objects)
            attachment.processing_status = "completed"
            attachment.processed_at = datetime.now(timezone.utc)

            await db.commit()

            # 7. Cleanup
            if os.path.exists(file_path):
                os.remove(file_path)

            print(f"✅ Processed chat attachment: {original_filename} ({len(chunk_objects)} chunks)")

        except Exception as e:
            await db.rollback()
            if os.path.exists(file_path):
                os.remove(file_path)

            print(f"❌ Error processing chat attachment {original_filename}: {e}")
            import traceback
            traceback.print_exc()


def _detect_mime_type(filename: str) -> str:
    """Detect MIME type from filename"""
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
        '.md': 'text/markdown',
    }

    return mime_types.get(ext, 'application/octet-stream')