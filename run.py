import argparse
import asyncio
import os
import shutil
import tempfile
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml
from sqlalchemy import text, select

from app.db.database import AsyncSessionLocal, engine
from app.db.models import Base, User, ChatSession, KnowledgeBase, ChatSessionKB
from app.services.kb_ingestion_service import process_kb_file
from app.services.attachment_ingestion_service import process_chat_attachment
from app.services.pipelines.chat_pipelines import generate_response_with_kb


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError("YAML root must be a mapping/object")
    return data


async def _ensure_db():
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS pgcrypto;"))
        await conn.run_sync(Base.metadata.create_all)

        # Basic indexes (idempotent)
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_chat_messages_session_id ON chat_messages (session_id);",
            "CREATE INDEX IF NOT EXISTS idx_kb_chunks_kb_id ON kb_chunks (kb_id);",
            "CREATE INDEX IF NOT EXISTS idx_kb_chunks_document_id ON kb_chunks (document_id);",
            "CREATE INDEX IF NOT EXISTS idx_kb_documents_kb_filename ON kb_documents (kb_id, filename);",
            "CREATE INDEX IF NOT EXISTS idx_chat_session_kbs_chat_id ON chat_session_kbs (chat_id);",
            "CREATE INDEX IF NOT EXISTS idx_chat_session_kbs_kb_id ON chat_session_kbs (kb_id);",
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_chat_session_kbs_unique ON chat_session_kbs (chat_id, kb_id);",
        ]

        vector_indexes = [
            """
            CREATE INDEX IF NOT EXISTS idx_chat_messages_embedding_ivfflat 
            ON chat_messages USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_kb_chunks_embedding_ivfflat 
            ON kb_chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
            """,
        ]

        for sql in indexes:
            try:
                await conn.execute(text(sql))
            except Exception:
                pass

        for sql in vector_indexes:
            try:
                await conn.execute(text(sql))
            except Exception:
                pass


async def _get_or_create_user(username: str) -> User:
    async with AsyncSessionLocal() as db:
        result = await db.execute(select(User).where(User.username == username))
        user = result.scalars().first()
        if user:
            return user

        user = User(username=username)
        db.add(user)
        await db.commit()
        await db.refresh(user)
        return user


async def _get_or_create_chat(user_id: Optional[str], chat_cfg: Dict[str, Any]) -> ChatSession:
    chat_id = chat_cfg.get("id")
    title = chat_cfg.get("title")
    system_prompt = chat_cfg.get("system_prompt")

    async with AsyncSessionLocal() as db:
        chat = None
        if chat_id:
            result = await db.execute(select(ChatSession).where(ChatSession.id == chat_id))
            chat = result.scalars().first()
            if not chat:
                raise ValueError(f"Chat not found for id: {chat_id}")
        elif title and user_id:
            result = await db.execute(
                select(ChatSession).where(ChatSession.user_id == user_id, ChatSession.title == title)
            )
            chat = result.scalars().first()

        if not chat:
            chat = ChatSession(user_id=user_id, title=title, system_prompt=system_prompt)
            db.add(chat)
            await db.commit()
            await db.refresh(chat)
        else:
            if system_prompt is not None and chat.system_prompt != system_prompt:
                chat.system_prompt = system_prompt
                await db.commit()
                await db.refresh(chat)

        return chat


async def _get_or_create_kb(user_id: str, kb_cfg: Dict[str, Any]) -> KnowledgeBase:
    title = kb_cfg.get("title")
    if not title:
        raise ValueError("KB entry missing required field: title")
    description = kb_cfg.get("description")

    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(KnowledgeBase).where(KnowledgeBase.user_id == user_id, KnowledgeBase.title == title)
        )
        kb = result.scalars().first()
        if kb:
            return kb

        kb = KnowledgeBase(user_id=user_id, title=title, description=description)
        db.add(kb)
        await db.commit()
        await db.refresh(kb)
        return kb


async def _attach_kb(chat_id: str, kb_id: str):
    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(ChatSessionKB).where(ChatSessionKB.chat_id == chat_id, ChatSessionKB.kb_id == kb_id)
        )
        existing = result.scalars().first()
        if existing:
            return

        link = ChatSessionKB(chat_id=chat_id, kb_id=kb_id)
        db.add(link)
        await db.commit()


def _normalize_file_entry(entry: Union[str, Dict[str, Any]]) -> Tuple[str, str]:
    if isinstance(entry, str):
        path = entry
        name = os.path.basename(path)
        return path, name

    if isinstance(entry, dict):
        path = entry.get("path")
        if not path:
            raise ValueError("File entry missing required field: path")
        name = entry.get("name") or os.path.basename(path)
        return path, name

    raise ValueError("File entry must be a string path or a mapping with 'path'")


def _copy_to_temp(path: str, name: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    temp_dir = tempfile.mkdtemp(prefix="nf_run_")
    temp_path = os.path.join(temp_dir, name)
    shutil.copy2(path, temp_path)
    return temp_path


async def _ingest_kb_files(kb_id: str, files: List[Union[str, Dict[str, Any]]]):
    for entry in files:
        path, name = _normalize_file_entry(entry)
        temp_path = _copy_to_temp(path, name)
        await process_kb_file(kb_id, temp_path, name)


async def _ingest_attachments(chat_id: str, user_id: str, files: List[Union[str, Dict[str, Any]]]):
    for entry in files:
        path, name = _normalize_file_entry(entry)
        temp_path = _copy_to_temp(path, name)
        await process_chat_attachment(chat_id, temp_path, name, user_id)


async def run_from_yaml(path: str):
    data = _load_yaml(path)

    user_cfg = data.get("user") or {}
    chat_cfg = data.get("chat") or {}
    kb_cfgs = data.get("knowledge_bases") or []
    attachments = data.get("attachments") or []
    messages = data.get("messages") or []

    if not isinstance(kb_cfgs, list):
        raise ValueError("knowledge_bases must be a list")
    if not isinstance(attachments, list):
        raise ValueError("attachments must be a list")
    if not isinstance(messages, list):
        raise ValueError("messages must be a list")

    username = user_cfg.get("username")
    if not username:
        raise ValueError("user.username is required")

    await _ensure_db()

    user = await _get_or_create_user(username)
    chat = await _get_or_create_chat(str(user.id), chat_cfg)

    kb_ids = []
    for kb_cfg in kb_cfgs:
        if not isinstance(kb_cfg, dict):
            raise ValueError("Each knowledge_bases entry must be an object")
        kb = await _get_or_create_kb(str(user.id), kb_cfg)
        kb_ids.append(str(kb.id))
        files = kb_cfg.get("files") or []
        if files:
            await _ingest_kb_files(str(kb.id), files)

    for kb_id in kb_ids:
        await _attach_kb(str(chat.id), kb_id)

    if attachments:
        await _ingest_attachments(str(chat.id), str(user.id), attachments)

    print(f"User: {user.username} ({user.id})")
    print(f"Chat: {chat.title or 'Untitled'} ({chat.id})")

    for i, msg in enumerate(messages, start=1):
        if not isinstance(msg, str):
            raise ValueError("messages entries must be strings")
        print(f"\n--- Message {i} ---")
        print(f"User: {msg}")
        result = await generate_response_with_kb(str(chat.id), msg)
        print(f"Assistant: {result['reply']}")


def main():
    parser = argparse.ArgumentParser(description="Run NeuralFoundry from a YAML config")
    parser.add_argument("yaml_path", help="Path to YAML config")
    args = parser.parse_args()

    asyncio.run(run_from_yaml(args.yaml_path))


if __name__ == "__main__":
    main()
