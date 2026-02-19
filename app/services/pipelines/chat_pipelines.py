# app/services/pipelines/chat_pipelines.py
import asyncio
from typing import Optional, List, Dict
from datetime import datetime, timezone
from openai import AsyncOpenAI
from app.core.config import settings
from app.services.vector_stores.pgvector_vs import PgVectorStore
from app.services.wrappers.async_embedding import get_embedding_async

# Initialize components (module-level singletons)
_pgv = PgVectorStore()
_openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)


def _format_relative_time(created_at) -> str:
    """Convert timestamp to relative time string"""
    try:
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at.replace('Z', '+00:00'))

        now = datetime.now(timezone.utc)
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)

        diff = now - created_at

        if diff.seconds < 60:
            return "just now"
        elif diff.seconds < 3600:
            mins = diff.seconds // 60
            return f"{mins} minute{'s' if mins > 1 else ''} ago"
        elif diff.seconds < 86400:
            hours = diff.seconds // 3600
            return f"{hours} hour{'s' if hours > 1 else ''} ago"
        elif diff.days == 1:
            return "1 day ago"
        elif diff.days < 7:
            return f"{diff.days} days ago"
        elif diff.days < 30:
            weeks = diff.days // 7
            return f"{weeks} week{'s' if weeks > 1 else ''} ago"
        else:
            return created_at.strftime("%b %d, %Y")
    except:
        return "recently"


async def _search_kb_if_attached(user_emb, kb_ids, max_kb_per_kb, kb_chunk_threshold):
    """Helper function to handle KB search with proper async handling"""
    if not kb_ids:
        return []
    return await _pgv.search_kb_chunks(
        vec=user_emb,
        kb_ids=kb_ids,
        limit_per_kb=max_kb_per_kb,
        threshold=kb_chunk_threshold
    )


def _build_hybrid_context(
        recent_messages: List[Dict],
        older_messages: List[Dict],
        kb_results: List[Dict],
        attachment_results: List[Dict],  # NEW
        custom_system_prompt: Optional[str]
) -> str:
    """
    Build structured context with natural conversation format and metadata.
    Now includes chat attachments!
    """
    context_parts = []

    # Base system prompt
    base_prompt = custom_system_prompt if custom_system_prompt else "You are a helpful AI assistant."
    context_parts.append(base_prompt)
    context_parts.append("")

    # Recent Conversation Section
    if recent_messages:
        context_parts.append("=" * 60)
        context_parts.append("RECENT CONVERSATION (Last {} messages)".format(len(recent_messages)))
        context_parts.append("=" * 60)
        context_parts.append("")

        for msg in recent_messages:
            time_str = _format_relative_time(msg.get('created_at'))
            role = msg['role'].capitalize()
            context_parts.append(f"{role} ({time_str}):")
            context_parts.append(msg['content'])
            context_parts.append("")

    # Older Relevant Conversation Section
    if older_messages:
        context_parts.append("=" * 60)
        context_parts.append("RELEVANT PAST CONVERSATION (From earlier messages)")
        context_parts.append("=" * 60)
        context_parts.append("")

        for msg in older_messages:
            time_str = _format_relative_time(msg.get('created_at'))
            similarity = msg.get('similarity', 0)
            role = msg['role'].capitalize()
            context_parts.append(f"{role} ({time_str} - Similarity: {similarity:.2f}):")
            context_parts.append(msg['content'])
            context_parts.append("")

    # Chat Attachments Context Section (NEW)
    if attachment_results:
        context_parts.append("=" * 60)
        context_parts.append("ATTACHED FILES CONTEXT")
        context_parts.append("=" * 60)
        context_parts.append("")

        for chunk in attachment_results:
            filename = chunk.get("metadata", {}).get("filename", "Unknown file")
            similarity = chunk.get("similarity", 0)

            context_parts.append(f"📎 From attached file: \"{filename}\"")
            context_parts.append(f"   Similarity: {similarity:.2f}")
            context_parts.append("")
            context_parts.append(chunk['text'])
            context_parts.append("")

    # Knowledge Base Context Section
    if kb_results:
        context_parts.append("=" * 60)
        context_parts.append("KNOWLEDGE BASE CONTEXT")
        context_parts.append("=" * 60)
        context_parts.append("")

        for chunk in kb_results:
            kb_title = chunk.get("kb_title", "Unknown KB")
            filename = chunk.get("filename", "Unknown")
            similarity = chunk.get("similarity", 0)

            context_parts.append(f"📚 From: \"{filename}\" (KB: {kb_title})")
            context_parts.append(f"   Similarity: {similarity:.2f}")
            context_parts.append("")
            context_parts.append(chunk['text'])
            context_parts.append("")

    # Instructions
    context_parts.append("=" * 60)
    context_parts.append("")
    context_parts.append(
        "Instructions:\n"
        "Based on the above context, answer the user's current question "
        "naturally and conversationally. Use the information provided but "
        "respond as if you naturally know this. Do not mention that you're "
        "using conversation history, knowledge bases, or attached files."
    )

    return "\n".join(context_parts)


async def generate_response_with_kb(
        chat_id: str,
        user_text: str,
        chat_history_threshold: Optional[float] = None,
        kb_chunk_threshold: Optional[float] = None,
        recent_window: Optional[int] = None,
        older_retrieval: Optional[int] = None,
        max_kb_per_kb: Optional[int] = None
) -> Dict:
    """
    Generate response using hybrid context approach with OpenAI.
    Now includes chat attachments in the context!
    """
    # Use settings defaults
    chat_history_threshold = chat_history_threshold or settings.CHAT_HISTORY_THRESHOLD
    kb_chunk_threshold = kb_chunk_threshold or settings.KB_CHUNK_THRESHOLD
    recent_window = recent_window or settings.RECENT_MESSAGE_WINDOW
    older_retrieval = older_retrieval or settings.OLDER_MESSAGE_RETRIEVAL
    max_kb_per_kb = max_kb_per_kb or settings.MAX_KB_CHUNKS_PER_KB

    # 🐛 DEBUG: Print query
    print(f"\n{'=' * 60}")
    print(f"🐛 NEW QUERY: {user_text}")
    print(f"🐛 Chat ID: {chat_id}")
    print(f"🐛 KB Threshold: {kb_chunk_threshold}")
    print(f"{'=' * 60}\n")

    # 1. Generate embedding for user query
    user_emb = await get_embedding_async(user_text)

    # 2. Store user message
    await _pgv.add_message(
        session_id=chat_id,
        role="user",
        content=user_text,
        embedding=user_emb
    )

    # 3. Get chat's custom system prompt
    custom_system_prompt = await _pgv.get_chat_system_prompt(chat_id)

    # 4. Get attached KB IDs
    kb_ids = await _pgv.get_attached_kb_ids(chat_id)

    # 🐛 DEBUG: Print KB attachment info
    print(f"🐛 Attached KB IDs: {kb_ids}")
    print(f"🐛 Number of KBs attached: {len(kb_ids)}")

    # 5. Parallel fetch: Recent messages + Older messages + KB chunks + Attachments (NEW)
    recent_messages, older_messages, kb_results, attachment_results = await asyncio.gather(
        _pgv.get_recent_messages(chat_id, limit=recent_window),
        _pgv.search_similar_excluding_recent(
            vec=user_emb,
            session_id=chat_id,
            exclude_recent_count=recent_window,
            limit=older_retrieval,
            threshold=chat_history_threshold
        ),
        _search_kb_if_attached(user_emb, kb_ids, max_kb_per_kb, kb_chunk_threshold),
        _pgv.search_chat_attachments(user_emb, chat_id, limit=3, threshold=kb_chunk_threshold),  # NEW
        return_exceptions=True
    )

    # Handle errors
    if isinstance(recent_messages, Exception):
        print(f"❌ Error fetching recent messages: {recent_messages}")
        recent_messages = []

    if isinstance(older_messages, Exception):
        print(f"❌ Error fetching older messages: {older_messages}")
        older_messages = []

    if isinstance(kb_results, Exception):
        print(f"❌ Error fetching KB chunks: {kb_results}")
        kb_results = []

    if isinstance(attachment_results, Exception):
        print(f"❌ Error fetching attachments: {attachment_results}")
        attachment_results = []

    # 🐛 DEBUG: Print retrieval results
    print(f"\n{'=' * 60}")
    print(f"🐛 RETRIEVAL RESULTS:")
    print(f"   - Recent messages: {len(recent_messages)}")
    print(f"   - Older messages: {len(older_messages)}")
    print(f"   - KB chunks: {len(kb_results)}")
    print(f"   - Attachment chunks: {len(attachment_results)}")  # NEW

    if kb_results:
        print(f"\n🐛 KB RESULTS:")
        for i, chunk in enumerate(kb_results[:3]):
            print(f"   [{i + 1}] Similarity: {chunk.get('similarity', 0):.4f}")
            print(f"       KB: {chunk.get('kb_title', 'Unknown')}")
            print(f"       File: {chunk.get('filename', 'Unknown')}")
            print(f"       Text preview: {chunk['text'][:100]}...")
    else:
        print(f"   ⚠️  NO KB RESULTS FOUND!")
        if kb_ids:
            print(f"   ⚠️  KBs are attached but no chunks matched threshold {kb_chunk_threshold}")
        else:
            print(f"   ⚠️  No KBs are attached to this chat!")

    # NEW: Debug attachment results
    if attachment_results:
        print(f"\n🐛 ATTACHMENT RESULTS:")
        for i, chunk in enumerate(attachment_results[:3]):
            print(f"   [{i + 1}] Similarity: {chunk.get('similarity', 0):.4f}")
            print(f"       File: {chunk.get('metadata', {}).get('filename', 'Unknown')}")
            print(f"       Text preview: {chunk['text'][:100]}...")
    else:
        print(f"   ℹ️  No attachments found in this chat")

    print(f"{'=' * 60}\n")

    # 6. Build structured context (now includes attachments)
    system_content = _build_hybrid_context(
        recent_messages=recent_messages,
        older_messages=older_messages,
        kb_results=kb_results,
        attachment_results=attachment_results,  # NEW
        custom_system_prompt=custom_system_prompt
    )

    # 7. Generate response using OpenAI
    assistant_reply = ""
    try:
        response = await _openai_client.chat.completions.create(
            model=settings.DEFAULT_LLM_MODEL,
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_text}
            ],
            temperature=settings.LLM_TEMPERATURE,
            max_tokens=settings.LLM_MAX_TOKENS
        )
        assistant_reply = response.choices[0].message.content.strip()
    except Exception as e:
        print(f"❌ Error generating response: {e}")
        assistant_reply = "I apologize, but I encountered an error generating a response."

    # 8. Prepare metadata (now includes attachments)
    kb_sources = []
    for chunk in kb_results:
        kb_title = chunk.get("kb_title", "Unknown")
        filename = chunk.get("filename", "Unknown")
        source_info = f"{kb_title} - {filename}"
        if source_info not in kb_sources:
            kb_sources.append(source_info)

    attachment_sources = []  # NEW
    for chunk in attachment_results:
        filename = chunk.get("metadata", {}).get("filename", "Unknown")
        if filename not in attachment_sources:
            attachment_sources.append(filename)

    sources_used = []
    if recent_messages or older_messages:
        sources_used.append("conversation history")
    if kb_results:
        sources_used.append("knowledge base")
    if attachment_results:
        sources_used.append("attached files")  # NEW

    metadata = {
        "sources_used": sources_used,
        "kb_sources": kb_sources,
        "attachment_sources": attachment_sources,  # NEW
        "recent_messages_count": len(recent_messages),
        "older_messages_count": len(older_messages),
        "kb_results_count": len(kb_results),
        "attachment_results_count": len(attachment_results),  # NEW
        "using_kb": len(kb_results) > 0,
        "using_attachments": len(attachment_results) > 0,  # NEW
        "using_history": len(recent_messages) > 0 or len(older_messages) > 0
    }

    # 9. Store clean assistant response
    reply_emb = await get_embedding_async(assistant_reply)
    await _pgv.add_message(
        session_id=chat_id,
        role="assistant",
        content=assistant_reply,
        embedding=reply_emb
    )

    # 10. Return response with metadata
    return {
        "reply": assistant_reply,
        "metadata": metadata
    }
