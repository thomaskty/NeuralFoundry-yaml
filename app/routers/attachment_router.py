# app/routers/attachment_router.py
from fastapi import APIRouter, UploadFile, File, BackgroundTasks, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
import uuid, os, tempfile
import aiofiles

from app.db.database import get_db
from app.db.models import ChatSession, ChatAttachment
from app.services.attachment_ingestion_service import process_chat_attachment

router = APIRouter()


# -------------------------------------------------------------------------
# 1. Upload attachment to chat
# -------------------------------------------------------------------------
@router.post("/chats/{chat_id}/attachments/upload", status_code=status.HTTP_202_ACCEPTED)
async def upload_chat_attachment(
        chat_id: str,
        background_tasks: BackgroundTasks,
        file: UploadFile = File(...),
        db: AsyncSession = Depends(get_db)
):
    """
    Upload a temporary file attachment to a chat.
    These files are processed and used for context in this chat only.
    """
    # Verify chat exists
    chat_result = await db.execute(select(ChatSession).where(ChatSession.id == chat_id))
    chat = chat_result.scalars().first()
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")

    # Check if filename already exists in this chat
    existing = await db.execute(
        select(ChatAttachment).where(
            ChatAttachment.chat_id == chat_id,
            ChatAttachment.filename == file.filename
        )
    )
    if existing.scalars().first():
        raise HTTPException(
            status_code=409,
            detail=f"File '{file.filename}' already attached to this chat"
        )

    # Save uploaded file to temp directory
    temp_dir = tempfile.gettempdir()
    file_path = os.path.join(temp_dir, f"{chat_id}_{uuid.uuid4().hex[:8]}_{file.filename}")

    async with aiofiles.open(file_path, "wb") as out_file:
        content = await file.read()
        await out_file.write(content)

    # Add background task for processing
    background_tasks.add_task(process_chat_attachment, chat_id, file_path, file.filename, chat.user_id)

    return {
        "message": f"File '{file.filename}' accepted for processing",
        "chat_id": chat_id,
        "filename": file.filename
    }


# -------------------------------------------------------------------------
# 2. List attachments in a chat
# -------------------------------------------------------------------------
@router.get("/chats/{chat_id}/attachments")
async def list_chat_attachments(
        chat_id: str,
        db: AsyncSession = Depends(get_db)
):
    """
    List all file attachments in a chat.
    """
    # Verify chat exists
    chat_result = await db.execute(select(ChatSession).where(ChatSession.id == chat_id))
    chat = chat_result.scalars().first()
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")

    # Get all attachments
    result = await db.execute(
        select(ChatAttachment)
        .where(ChatAttachment.chat_id == chat_id)
        .order_by(ChatAttachment.uploaded_at.desc())
    )
    attachments = result.scalars().all()

    return {
        "chat_id": chat_id,
        "total_attachments": len(attachments),
        "attachments": [
            {
                "id": str(att.id),
                "filename": att.filename,
                "mime_type": att.mime_type,
                "file_size": att.file_size,
                "total_chunks": att.total_chunks,
                "processing_status": att.processing_status,
                "uploaded_at": att.uploaded_at,
                "processed_at": att.processed_at
            }
            for att in attachments
        ]
    }


# -------------------------------------------------------------------------
# 3. Delete attachment from chat
# -------------------------------------------------------------------------
@router.delete("/chats/{chat_id}/attachments/{attachment_id}", status_code=status.HTTP_200_OK)
async def delete_chat_attachment(
        chat_id: str,
        attachment_id: str,
        db: AsyncSession = Depends(get_db)
):
    """
    Delete a file attachment from a chat.
    This removes all associated chunks (CASCADE).
    """
    # Verify attachment exists and belongs to this chat
    att_result = await db.execute(
        select(ChatAttachment).where(
            ChatAttachment.id == attachment_id,
            ChatAttachment.chat_id == chat_id
        )
    )
    attachment = att_result.scalars().first()

    if not attachment:
        raise HTTPException(status_code=404, detail="Attachment not found in this chat")

    filename = attachment.filename
    await db.delete(attachment)
    await db.commit()

    return {
        "message": f"Attachment '{filename}' deleted successfully",
        "attachment_id": attachment_id
    }