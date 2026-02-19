import argparse
import asyncio
import os
import shutil
import tempfile
import logging
import sys
import hashlib
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path

import yaml
from sqlalchemy import text, select

from app.db.database import AsyncSessionLocal, engine
from app.db.models import Base, User, ChatSession, KnowledgeBase, ChatSessionKB, ChatAttachment
from app.services.kb_ingestion_service import process_kb_file
from app.services.attachment_ingestion_service import process_chat_attachment
from app.services.pipelines.chat_pipelines import generate_response_with_kb

# Configure base logging (file handler attached per-run)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)
result_logger = logging.getLogger("neuralfoundry.result")
if not result_logger.handlers:
    result_logger.setLevel(logging.INFO)
    result_logger.propagate = False
    result_logger.addHandler(logging.StreamHandler(sys.stdout))


def _load_yaml(path: str) -> Dict[str, Any]:
    """Load and validate YAML configuration file."""
    path_obj = Path(path)
    
    if not path_obj.exists():
        raise FileNotFoundError(f"YAML config file not found: {path}")
    
    if not path_obj.is_file():
        raise ValueError(f"Path is not a file: {path}")
    
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML syntax in {path}: {e}")
    
    if data is None:
        raise ValueError("YAML file is empty")
    
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping/object, got {type(data).__name__}")
    
    logger.info(f"Loaded configuration from {path}")
    return data


async def _ensure_db():
    """Ensure database is ready: create extensions, tables, and indexes."""
    logger.info("Initializing database...")
    
    try:
        async with engine.begin() as conn:
            # Create extensions
            logger.info("  Creating extensions (vector, pgcrypto)...")
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS pgcrypto;"))
            
            # Create tables
            logger.info("  Creating tables from models...")
            await conn.run_sync(Base.metadata.create_all)

            # Basic indexes (idempotent)
            indexes = [
                "CREATE INDEX IF NOT EXISTS idx_chat_messages_session_id ON chat_messages (session_id);",
                "CREATE INDEX IF NOT EXISTS idx_kb_documents_kb_id ON kb_documents (kb_id);",
                "CREATE INDEX IF NOT EXISTS idx_kb_documents_global_document_id ON kb_documents (global_document_id);",
                "CREATE INDEX IF NOT EXISTS idx_kb_chunk_links_kb_id ON kb_chunk_links (kb_id);",
                "CREATE INDEX IF NOT EXISTS idx_kb_chunk_links_document_id ON kb_chunk_links (document_id);",
                "CREATE INDEX IF NOT EXISTS idx_kb_chunk_links_global_chunk_id ON kb_chunk_links (global_chunk_id);",
                "CREATE INDEX IF NOT EXISTS idx_global_chunks_document_id ON global_chunks (document_id);",
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
                CREATE INDEX IF NOT EXISTS idx_global_chunks_embedding_ivfflat 
                ON global_chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
                """,
            ]

            logger.info("  Creating regular indexes...")
            for sql in indexes:
                try:
                    await conn.execute(text(sql))
                except Exception as e:
                    logger.debug(f"Index creation note: {e}")
                    pass

            logger.info("  Creating vector indexes...")
            for sql in vector_indexes:
                try:
                    await conn.execute(text(sql))
                except Exception as e:
                    logger.debug(f"Vector index creation note: {e}")
                    pass
        
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise


async def _get_or_create_user(username: str) -> User:
    """Get or create a user by username."""
    async with AsyncSessionLocal() as db:
        result = await db.execute(select(User).where(User.username == username))
        user = result.scalars().first()
        if user:
            logger.info(f"  User '{username}' already exists (ID: {user.id})")
            return user

        user = User(username=username)
        db.add(user)
        await db.commit()
        await db.refresh(user)
        logger.info(f"  Created user '{username}' (ID: {user.id})")
        return user


async def _get_or_create_chat(user_id: Optional[str], chat_cfg: Dict[str, Any]) -> ChatSession:
    """Get or create a chat session."""
    chat_id = chat_cfg.get("id")
    title = chat_cfg.get("title", "Untitled Chat")
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
            logger.info(f"  Created chat '{title}' (ID: {chat.id})")
        else:
            logger.info(f"  Chat '{title}' already exists (ID: {chat.id})")
            if system_prompt is not None and chat.system_prompt != system_prompt:
                chat.system_prompt = system_prompt
                await db.commit()
                await db.refresh(chat)
                logger.info(f"    Updated system prompt")

        return chat


async def _get_or_create_kb(user_id: str, kb_cfg: Dict[str, Any]) -> KnowledgeBase:
    """Get or create a knowledge base."""
    title = kb_cfg.get("title")
    if not title:
        raise ValueError("KB entry missing required field: 'title'")
    description = kb_cfg.get("description", "")

    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(KnowledgeBase).where(KnowledgeBase.user_id == user_id, KnowledgeBase.title == title)
        )
        kb = result.scalars().first()
        if kb:
            logger.info(f"    KB '{title}' already exists (ID: {kb.id})")
            return kb

        kb = KnowledgeBase(user_id=user_id, title=title, description=description)
        db.add(kb)
        await db.commit()
        await db.refresh(kb)
        logger.info(f"    Created KB '{title}' (ID: {kb.id})")
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


def _compute_text_hash(file_path: str, original_filename: str) -> str:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        return hashlib.sha256(content.encode("utf-8")).hexdigest()
    except Exception:
        return hashlib.sha256(original_filename.encode()).hexdigest()


async def _attachment_already_chunked(chat_id: str, file_path: str, original_filename: str) -> bool:
    file_size = os.path.getsize(file_path)
    file_hash = _compute_text_hash(file_path, original_filename)
    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(ChatAttachment).where(
                ChatAttachment.chat_id == chat_id,
                ChatAttachment.filename == original_filename,
                ChatAttachment.file_size == file_size,
            )
        )
        for attachment in result.scalars().all():
            if attachment.processing_status != "completed" or (attachment.total_chunks or 0) == 0:
                continue
            meta = attachment.file_metadata or {}
            stored_hash = meta.get("sha256")
            if stored_hash is None or stored_hash == file_hash:
                return True
        return False


async def _ingest_kb_files(kb_id: str, files: List[Union[str, Dict[str, Any]]], replace_if_changed: bool = False):
    for entry in files:
        path, name = _normalize_file_entry(entry)
        temp_path = _copy_to_temp(path, name)
        result = await process_kb_file(kb_id, temp_path, name, replace_if_changed=replace_if_changed)
        status = result.get("status")
        message = result.get("message", "")
        if status:
            logger.info(f"    [{status}] {message}")


async def _ingest_attachments(chat_id: str, user_id: str, files: List[Union[str, Dict[str, Any]]]):
    for entry in files:
        path, name = _normalize_file_entry(entry)
        if await _attachment_already_chunked(chat_id, path, name):
            logger.info(f"  Skipping attachment '{name}' (already chunked)")
            continue
        temp_path = _copy_to_temp(path, name)
        await process_chat_attachment(chat_id, temp_path, name, user_id)


async def run_from_yaml(path: str):
    """Run NeuralFoundry from a YAML configuration file."""
    # Set up per-run log file in logs/ based on YAML filename
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{Path(path).stem}.log"
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(file_handler)

    logger.info("=" * 70)
    logger.info("NeuralFoundry YAML Runner")
    logger.info("=" * 70)
    
    try:
        # Load and validate config
        logger.info("\nLoading configuration...")
        data = _load_yaml(path)

        user_cfg = data.get("user") or {}
        chat_cfg = data.get("chat") or {}
        kb_cfgs = data.get("knowledge_bases") or []
        attachments = data.get("attachments") or []
        messages = data.get("messages") or []
        output_cfg = data.get("output_file", False)

        # Validate configuration structure
        if not isinstance(kb_cfgs, list):
            raise ValueError("'knowledge_bases' must be a list")
        if not isinstance(attachments, list):
            raise ValueError("'attachments' must be a list")
        if not isinstance(messages, list):
            raise ValueError("'messages' must be a list")
        if not isinstance(output_cfg, (bool, str)):
            raise ValueError("'output_file' must be a boolean or string path")

        username = user_cfg.get("username")
        if not username:
            raise ValueError("'user.username' is required in YAML config")

        # Initialize database
        await _ensure_db()

        # Create/get user
        logger.info("\nUser setup:")
        user = await _get_or_create_user(username)

        # Create/get chat session
        logger.info("\nChat setup:")
        chat = await _get_or_create_chat(str(user.id), chat_cfg)

        # Process knowledge bases
        kb_ids = []
        if kb_cfgs:
            logger.info("\nKnowledge bases:")
            for i, kb_cfg in enumerate(kb_cfgs, 1):
                if not isinstance(kb_cfg, dict):
                    raise ValueError(f"Knowledge base #{i} must be an object/mapping")
                
                logger.info(f"  [{i}/{len(kb_cfgs)}] Processing '{kb_cfg.get('title', 'Unknown')}'...")
                kb = await _get_or_create_kb(str(user.id), kb_cfg)
                kb_ids.append(str(kb.id))
                
                files = kb_cfg.get("files") or []
                replace_if_changed = bool(kb_cfg.get("replace_if_changed", False))
                if files:
                    logger.info(f"    Ingesting {len(files)} file(s)...")
                    try:
                        await _ingest_kb_files(str(kb.id), files, replace_if_changed=replace_if_changed)
                        logger.info("    Files ingested successfully")
                    except Exception as e:
                        logger.error(f"    Error ingesting files: {e}")
                        raise

        # Attach KBs to chat
        if kb_ids:
            logger.info(f"\nAttaching {len(kb_ids)} KB(s) to chat...")
            for kb_id in kb_ids:
                await _attach_kb(str(chat.id), kb_id)
            logger.info("  All KBs attached")

        # Process attachments
        if attachments:
            logger.info(f"\nProcessing {len(attachments)} attachment(s)...")
            try:
                await _ingest_attachments(str(chat.id), str(user.id), attachments)
                logger.info("  Attachments processed")
            except Exception as e:
                logger.error(f"  Error processing attachments: {e}")
                raise

        # Display setup summary
        logger.info("\n" + "=" * 70)
        logger.info("Session Summary")
        logger.info("=" * 70)
        logger.info(f"User:        {user.username} ({user.id})")
        logger.info(f"Chat:        {chat.title or 'Untitled'} ({chat.id})")
        logger.info(f"KBs:         {len(kb_ids)}")
        logger.info(f"Attachments: {len(attachments)}")
        logger.info(f"Queries:     {len(messages)}")

        # Resolve output path (optional)
        output_path = None
        if output_cfg:
            if isinstance(output_cfg, str):
                output_path = output_cfg
            else:
                outputs_dir = Path("outputs")
                outputs_dir.mkdir(parents=True, exist_ok=True)
                output_path = str(outputs_dir / f"{Path(path).stem}.out.txt")
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(f"# NeuralFoundry Run Output\n")
                f.write(f"Source YAML: {path}\n\n")

        # Process messages
        if messages:
            logger.info("\n" + "=" * 70)
            logger.info("Processing Queries")
            logger.info("=" * 70)
            
            for i, msg in enumerate(messages, start=1):
                if not isinstance(msg, str):
                    raise ValueError(f"Message #{i} must be a string")
                
                logger.info(f"\n[{i}/{len(messages)}] Query: {msg[:80]}{'...' if len(msg) > 80 else ''}")
                try:
                    result = await generate_response_with_kb(str(chat.id), msg)
                    reply = result.get("reply", "")
                    result_logger.info("\n" + "-" * 70)
                    result_logger.info(f"Query {i}: {msg}")
                    result_logger.info("-" * 70)
                    result_logger.info(reply)
                    result_logger.info("-" * 70 + "\n")
                    if output_path:
                        with open(output_path, "a", encoding="utf-8") as f:
                            f.write(f"## Query {i}\n")
                            f.write(f"{msg}\n\n")
                            f.write(f"## Answer {i}\n")
                            f.write(f"{reply}\n\n")
                except Exception as e:
                    logger.error(f"Error processing query: {e}")
                    raise
        else:
            logger.info("\nNo queries to process")

        logger.info("\n" + "=" * 70)
        logger.info("Execution completed successfully")
        logger.info("=" * 70)

    except Exception as e:
        logger.error(f"\nExecution failed: {e}", exc_info=True)
        raise SystemExit(1)
    finally:
        logger.removeHandler(file_handler)
        file_handler.close()


def main():
    """Entry point for the YAML runner."""
    parser = argparse.ArgumentParser(
        description="Run NeuralFoundry RAG pipeline from YAML configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py run.yaml
  python run.py ./configs/chat_config.yaml
  python run.py documents/run_with_attachments.yaml
        """
    )
    parser.add_argument("yaml_path", help="Path to YAML configuration file")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("Verbose logging enabled")
    
    try:
        asyncio.run(run_from_yaml(args.yaml_path))
    except KeyboardInterrupt:
        logger.warning("\nExecution interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
