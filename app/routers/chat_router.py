import uuid
from datetime import datetime, timezone
from sqlalchemy.orm import selectinload
from fastapi import APIRouter, HTTPException, status, Depends
from sqlalchemy.future import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import get_db
from app.db.models import ChatMessage, ChatSession, User
from app.models.schemas import ChatCreate, MessageCreate, ChatSummaryRead
from app.services.pipelines.chat_pipelines import generate_response_with_kb

router = APIRouter()

# -------------------------------------------------------------------------
# 1. Chat message handler with KB support and metadata : ENHANCED
# -------------------------------------------------------------------------
@router.post("/chats/{chat_id}/messages", status_code=status.HTTP_201_CREATED)
async def conversation_chat(chat_id: str, body: MessageCreate):
    """
    Handle user messages and generate an assistant reply.
    Now supports Knowledge Base context automatically and returns metadata!

    Returns:
        - reply: Clean response text
        - metadata: Source information (KB, chat history usage)
    """
    if not body.content.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    # Use KB-aware pipeline (returns dict with reply + metadata)
    response_data = await generate_response_with_kb(chat_id, body.content)

    return {
        "chat_id": chat_id,
        "reply": response_data["reply"],
        "metadata": response_data.get("metadata", {})
    }


# -------------------------------------------------------------------------
# 2. Create chat session (supports user_id) : validated
# -------------------------------------------------------------------------
@router.post("/users/{user_id}/chats", status_code=status.HTTP_201_CREATED)
async def create_chat_for_user(user_id: str, payload: ChatCreate, db: AsyncSession = Depends(get_db)):
    """
    Create a new chat session linked to a specific user.
    """
    # Verify user exists
    user_result = await db.execute(select(User).where(User.id == user_id))
    user = user_result.scalars().first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    chat = ChatSession(
        title=payload.title or "Untitled",
        system_prompt=payload.system_prompt or None,
        created_at=datetime.now(timezone.utc),
        user_id=user_id,
    )
    db.add(chat)
    await db.commit()
    await db.refresh(chat)

    return {
        "chat_id": str(chat.id),
        "user_id": str(user_id),
        "title": chat.title,
        "system_prompt": chat.system_prompt,
        "created_at": chat.created_at
    }


# -------------------------------------------------------------------------
# 3. Fetch chat + messages : validated (with ORDER BY fix)
# -------------------------------------------------------------------------
@router.get("/chats/{chat_id}")
async def get_chat(chat_id: str, db: AsyncSession = Depends(get_db)):
    """
    Fetch all messages for a given chat_id, properly ordered by timestamp.
    """
    # Fetch the chat session + related user + messages
    result = await db.execute(
        select(ChatSession)
        .options(selectinload(ChatSession.user), selectinload(ChatSession.messages))
        .where(ChatSession.id == chat_id)
    )
    chat = result.scalars().first()

    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")

    # Sort messages by created_at to ensure correct order
    sorted_messages = sorted(chat.messages, key=lambda m: m.created_at)

    return {
        "chat_id": str(chat.id),
        "title": chat.title,
        "created_at": chat.created_at,
        "user": {
            "user_id": str(chat.user.id) if chat.user else None,
            "username": chat.user.username if chat.user else None
        },
        "system_prompt": chat.system_prompt,
        "messages": [
            {
                "id": m.id,
                "role": m.role,
                "content": m.content,
                "created_at": m.created_at
            }
            for m in sorted_messages
        ],
    }


# -------------------------------------------------------------------------
# 4. Delete chat
# -------------------------------------------------------------------------
@router.delete("/chats/{chat_id}", status_code=status.HTTP_200_OK)
async def delete_chat(chat_id: str, db: AsyncSession = Depends(get_db)):
    """
    Delete a chat session and its messages.
    """
    chat_result = await db.execute(select(ChatSession).where(ChatSession.id == chat_id))
    chat = chat_result.scalars().first()
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")

    await db.delete(chat)
    await db.commit()

    return {"message": f"Chat {chat_id} deleted"}


# -------------------------------------------------------------------------
# 5. List all chats : all chats (list of all the chats created by multiple users)
# -------------------------------------------------------------------------
@router.get("/chats", response_model=list[ChatSummaryRead])
async def get_all_chats(db: AsyncSession = Depends(get_db)):
    """
    Fetch all chat sessions with their messages.
    """
    result = await db.execute(select(ChatSession))
    chats = result.scalars().all()

    all_chats = []
    for chat in chats:
        messages_result = await db.execute(
            select(ChatMessage)
            .where(ChatMessage.session_id == chat.id)
            .order_by(ChatMessage.created_at)  # Order by timestamp
        )
        messages = messages_result.scalars().all()

        all_chats.append({
            "chat_id": str(chat.id),
            "user_id": str(chat.user_id) if chat.user_id else None,
            "title": chat.title,
            "messages": [
                {"id": m.id, "role": m.role, "content": m.content, "created_at": m.created_at}
                for m in messages
            ],
        })

    return all_chats


# -------------------------------------------------------------------------
# 6. List all chats per user : validated ( chat content is not there but chat_id is available )
# -------------------------------------------------------------------------
@router.get("/users/{user_id}/chats")
async def get_user_chats(user_id: str, db: AsyncSession = Depends(get_db)):
    """
    Fetch all chats belonging to a specific user.
    """
    user_result = await db.execute(select(User).where(User.id == user_id))
    user = user_result.scalars().first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    chat_result = await db.execute(select(ChatSession).where(ChatSession.user_id == user_id))
    chats = chat_result.scalars().all()

    return {
        "user_id": str(user_id),
        "username": user.username,
        "chats": [
            {"chat_id": str(c.id), "title": c.title, "created_at": c.created_at}
            for c in chats
        ],
    }


# Optional: allow creation without user linkage (for backward compatibility)
@router.post("/chats", status_code=status.HTTP_201_CREATED)
async def create_chat(payload: ChatCreate, db: AsyncSession = Depends(get_db)):
    """
    Create a chat session not linked to any specific user.
    """
    chat = ChatSession(
        title=payload.title or "Untitled",
        created_at=datetime.now(timezone.utc),
        user_id=None,
    )
    db.add(chat)
    await db.commit()
    await db.refresh(chat)

    return {"chat_id": str(chat.id), "title": chat.title, "created_at": chat.created_at}