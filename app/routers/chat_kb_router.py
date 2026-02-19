from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from datetime import datetime, timezone
from typing import List

from app.db.database import get_db
from app.db.models import ChatSession, KnowledgeBase, ChatSessionKB

router = APIRouter()


# -------------------------------------------------------------------------
# 1. Attach KB to Chat
# -------------------------------------------------------------------------
@router.post("/chats/{chat_id}/knowledge-bases/{kb_id}", status_code=status.HTTP_201_CREATED)
async def attach_kb_to_chat(
        chat_id: str,
        kb_id: str,
        db: AsyncSession = Depends(get_db)
):
    """
    Attach a knowledge base to a chat session.
    A chat can have 0, 1, or many KBs attached.
    """
    # Verify chat exists
    chat_result = await db.execute(select(ChatSession).where(ChatSession.id == chat_id))
    chat = chat_result.scalars().first()
    if not chat:
        raise HTTPException(status_code=404, detail="Chat session not found")

    # Verify KB exists
    kb_result = await db.execute(select(KnowledgeBase).where(KnowledgeBase.id == kb_id))
    kb = kb_result.scalars().first()
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")

    # Capture KB title before commit
    kb_title = kb.title

    # Check if already attached
    existing = await db.execute(
        select(ChatSessionKB).where(
            ChatSessionKB.chat_id == chat_id,
            ChatSessionKB.kb_id == kb_id
        )
    )
    if existing.scalars().first():
        raise HTTPException(
            status_code=409,
            detail=f"Knowledge base '{kb_title}' is already attached to this chat"
        )

    # Create attachment
    attachment = ChatSessionKB(
        chat_id=chat_id,
        kb_id=kb_id,
        attached_at=datetime.now(timezone.utc)
    )
    db.add(attachment)

    # Capture attached_at before commit
    attached_at = attachment.attached_at

    await db.commit()

    return {
        "message": f"Knowledge base '{kb_title}' attached to chat",
        "chat_id": str(chat_id),
        "kb_id": str(kb_id),
        "kb_title": kb_title,
        "attached_at": attached_at
    }


# -------------------------------------------------------------------------
# 2. Detach KB from Chat
# -------------------------------------------------------------------------
@router.delete("/chats/{chat_id}/knowledge-bases/{kb_id}", status_code=status.HTTP_200_OK)
async def detach_kb_from_chat(
        chat_id: str,
        kb_id: str,
        db: AsyncSession = Depends(get_db)
):
    """
    Detach a knowledge base from a chat session.
    """
    # Find the attachment
    result = await db.execute(
        select(ChatSessionKB).where(
            ChatSessionKB.chat_id == chat_id,
            ChatSessionKB.kb_id == kb_id
        )
    )
    attachment = result.scalars().first()

    if not attachment:
        raise HTTPException(
            status_code=404,
            detail="Knowledge base is not attached to this chat"
        )

    await db.delete(attachment)
    await db.commit()

    return {
        "message": "Knowledge base detached from chat",
        "chat_id": str(chat_id),
        "kb_id": str(kb_id)
    }


# -------------------------------------------------------------------------
# 3. List all KBs attached to a Chat
# -------------------------------------------------------------------------
@router.get("/chats/{chat_id}/knowledge-bases")
async def list_chat_kbs(
        chat_id: str,
        db: AsyncSession = Depends(get_db)
):
    """
    List all knowledge bases attached to a specific chat session.
    Returns empty list if no KBs are attached.
    """
    # Verify chat exists
    chat_result = await db.execute(select(ChatSession).where(ChatSession.id == chat_id))
    chat = chat_result.scalars().first()
    if not chat:
        raise HTTPException(status_code=404, detail="Chat session not found")

    # Get all attachments with KB details
    result = await db.execute(
        select(ChatSessionKB, KnowledgeBase)
        .join(KnowledgeBase, ChatSessionKB.kb_id == KnowledgeBase.id)
        .where(ChatSessionKB.chat_id == chat_id)
        .order_by(ChatSessionKB.attached_at.desc())
    )

    attachments = result.all()

    return {
        "chat_id": str(chat_id),
        "total_kbs": len(attachments),
        "knowledge_bases": [
            {
                "kb_id": str(kb.id),
                "title": kb.title,
                "description": kb.description,
                "attached_at": attachment.attached_at
            }
            for attachment, kb in attachments
        ]
    }


# -------------------------------------------------------------------------
# 4. List all Chats using a specific KB
# -------------------------------------------------------------------------
@router.get("/knowledge-bases/{kb_id}/chats")
async def list_kb_chats(
        kb_id: str,
        db: AsyncSession = Depends(get_db)
):
    """
    List all chat sessions that are using a specific knowledge base.
    Useful to see which chats depend on a KB before deletion.
    """
    # Verify KB exists
    kb_result = await db.execute(select(KnowledgeBase).where(KnowledgeBase.id == kb_id))
    kb = kb_result.scalars().first()
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")

    # Get all chats using this KB
    result = await db.execute(
        select(ChatSessionKB, ChatSession)
        .join(ChatSession, ChatSessionKB.chat_id == ChatSession.id)
        .where(ChatSessionKB.kb_id == kb_id)
        .order_by(ChatSessionKB.attached_at.desc())
    )

    attachments = result.all()

    return {
        "kb_id": str(kb_id),
        "kb_title": kb.title,
        "total_chats": len(attachments),
        "chats": [
            {
                "chat_id": str(chat.id),
                "chat_title": chat.title,
                "attached_at": attachment.attached_at
            }
            for attachment, chat in attachments
        ]
    }


# -------------------------------------------------------------------------
# 5. Bulk attach multiple KBs to a Chat
# -------------------------------------------------------------------------
@router.post("/chats/{chat_id}/knowledge-bases/bulk", status_code=status.HTTP_201_CREATED)
async def bulk_attach_kbs_to_chat(
        chat_id: str,
        kb_ids: List[str],
        db: AsyncSession = Depends(get_db)
):
    """
    Attach multiple knowledge bases to a chat session at once.
    Skips KBs that are already attached.
    """
    # Verify chat exists
    chat_result = await db.execute(select(ChatSession).where(ChatSession.id == chat_id))
    chat = chat_result.scalars().first()
    if not chat:
        raise HTTPException(status_code=404, detail="Chat session not found")

    attached = []
    skipped = []
    not_found = []

    for kb_id in kb_ids:
        # Check if KB exists
        kb_result = await db.execute(select(KnowledgeBase).where(KnowledgeBase.id == kb_id))
        kb = kb_result.scalars().first()

        if not kb:
            not_found.append(kb_id)
            continue

        # Check if already attached
        existing = await db.execute(
            select(ChatSessionKB).where(
                ChatSessionKB.chat_id == chat_id,
                ChatSessionKB.kb_id == kb_id
            )
        )

        if existing.scalars().first():
            skipped.append({"kb_id": kb_id, "title": kb.title})
            continue

        # Attach KB
        attachment = ChatSessionKB(
            chat_id=chat_id,
            kb_id=kb_id,
            attached_at=datetime.now(timezone.utc)
        )
        db.add(attachment)
        attached.append({"kb_id": kb_id, "title": kb.title})

    await db.commit()

    return {
        "chat_id": str(chat_id),
        "attached": attached,
        "skipped": skipped,
        "not_found": not_found,
        "summary": f"Attached {len(attached)}, skipped {len(skipped)}, not found {len(not_found)}"
    }