import numpy as np
from sqlalchemy import text
from typing import List, Optional
from app.db.database import engine


def _vec_literal(embedding: np.ndarray) -> str:
    """Convert numpy array to PostgreSQL vector literal"""
    return "[" + ",".join(map(str, embedding.tolist())) + "]"


class PgVectorStore:
    """Vector store for chat messages and KB chunks using pgvector"""

    async def add_message(
            self,
            session_id: str,
            role: str,
            content: str,
            embedding: np.ndarray | None
    ):
        """Add a chat message with optional embedding"""
        async with engine.begin() as conn:
            if embedding is not None:
                vec_literal = _vec_literal(embedding)
                sql = f"""
                    INSERT INTO chat_messages (session_id, role, content, embedding)
                    VALUES ('{session_id}', '{role}', '{content.replace("'", "''")}', '{vec_literal}'::vector)
                """
            else:
                sql = f"""
                    INSERT INTO chat_messages (session_id, role, content)
                    VALUES ('{session_id}', '{role}', '{content.replace("'", "''")}')
                """
            await conn.exec_driver_sql(sql)

    async def get_recent_messages(
            self,
            session_id: str,
            limit: int = 10
    ) -> List[dict]:
        """
        Get the most recent N messages from a chat session.
        Returns messages in chronological order (oldest first).
        Used for maintaining conversation context.

        Args:
            session_id: Chat session ID
            limit: Number of recent messages to retrieve

        Returns:
            List of dicts with keys: id, session_id, role, content, created_at
        """
        sql = f"""
            SELECT 
                id,
                session_id,
                role,
                content,
                created_at
            FROM chat_messages
            WHERE session_id = '{session_id}'
            ORDER BY created_at DESC
            LIMIT {limit};
        """

        async with engine.begin() as conn:
            result = await conn.exec_driver_sql(sql)
            rows = result.mappings().all()
            # Reverse to get chronological order (oldest first)
            return list(reversed([dict(row) for row in rows]))

    async def search_similar_excluding_recent(
            self,
            vec: np.ndarray,
            session_id: str,
            exclude_recent_count: int = 10,
            limit: int = 3,
            threshold: float = 0.75
    ) -> List[dict]:
        """
        Search for similar messages excluding the most recent N messages.
        Used for finding relevant context from older conversation.

        Args:
            vec: Query embedding vector
            session_id: Chat session ID
            exclude_recent_count: Number of recent messages to exclude
            limit: Maximum number of results
            threshold: Minimum similarity threshold (0-1)

        Returns:
            List of dicts with keys: id, session_id, role, content, similarity, created_at
        """
        vec_literal = _vec_literal(vec)

        # Get IDs of recent messages to exclude
        recent_ids_sql = f"""
            SELECT id FROM chat_messages
            WHERE session_id = '{session_id}'
            ORDER BY created_at DESC
            LIMIT {exclude_recent_count}
        """

        sql = f"""
            WITH recent_message_ids AS ({recent_ids_sql})
            SELECT 
                id, 
                session_id, 
                role, 
                content,
                created_at,
                1 - (embedding <=> '{vec_literal}'::vector) AS similarity
            FROM chat_messages
            WHERE session_id = '{session_id}'
              AND id NOT IN (SELECT id FROM recent_message_ids)
              AND 1 - (embedding <=> '{vec_literal}'::vector) >= {threshold}
            ORDER BY similarity DESC
            LIMIT {limit};
        """

        async with engine.begin() as conn:
            result = await conn.exec_driver_sql(sql)
            rows = result.mappings().all()
            return [dict(row) for row in rows]

    async def search_similar(
            self,
            vec: np.ndarray,
            limit: int = 5,
            threshold: float | None = None,
            session_filter: str | None = None
    ) -> List[dict]:
        """
        Search for similar messages in chat history.
        Legacy method - kept for backward compatibility.

        Args:
            vec: Query embedding vector
            limit: Maximum number of results
            threshold: Minimum similarity threshold (0-1)
            session_filter: Filter by specific session_id

        Returns:
            List of dicts with keys: id, session_id, role, content, similarity
        """
        vec_literal = _vec_literal(vec)

        # Build WHERE clause
        where_clauses = []
        if session_filter:
            where_clauses.append(f"session_id = '{session_filter}'")

        if threshold is not None:
            where_clauses.append(f"1 - (embedding <=> '{vec_literal}'::vector) >= {threshold}")

        where_clause = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""

        sql = f"""
            SELECT 
                id, 
                session_id, 
                role, 
                content,
                1 - (embedding <=> '{vec_literal}'::vector) AS similarity
            FROM chat_messages
            {where_clause}
            ORDER BY similarity DESC
            LIMIT {limit};
        """

        async with engine.begin() as conn:
            result = await conn.exec_driver_sql(sql)
            rows = result.mappings().all()
            return [dict(row) for row in rows]

    async def search_kb_chunks(
            self,
            vec: np.ndarray,
            kb_ids: List[str],
            limit_per_kb: int = 3,
            threshold: float | None = None
    ) -> List[dict]:
        """
        Search for similar chunks across multiple knowledge bases.

        Args:
            vec: Query embedding vector
            kb_ids: List of knowledge base IDs to search
            limit_per_kb: Maximum chunks to return per KB
            threshold: Minimum similarity threshold (0-1)

        Returns:
            List of dicts with keys: id, kb_id, document_id, chunk_index,
            text, similarity, filename, kb_title
        """
        if not kb_ids:
            return []

        vec_literal = _vec_literal(vec)

        # Build WHERE clause for KB filtering
        kb_filter = "', '".join(kb_ids)
        where_clause = f"WHERE c.kb_id IN ('{kb_filter}')"

        if threshold is not None:
            where_clause += f" AND 1 - (c.embedding <=> '{vec_literal}'::vector) >= {threshold}"

        # Use ROW_NUMBER() to limit results per KB
        sql = f"""
            WITH ranked_chunks AS (
                SELECT 
                    c.id,
                    c.kb_id,
                    c.document_id,
                    c.chunk_index,
                    c.text,
                    c.token_count,
                    d.filename,
                    k.title as kb_title,
                    1 - (c.embedding <=> '{vec_literal}'::vector) AS similarity,
                    ROW_NUMBER() OVER (
                        PARTITION BY c.kb_id 
                        ORDER BY 1 - (c.embedding <=> '{vec_literal}'::vector) DESC
                    ) as rank
                FROM kb_chunks c
                JOIN kb_documents d ON c.document_id = d.id
                JOIN knowledge_bases k ON c.kb_id = k.id
                {where_clause}
            )
            SELECT 
                id,
                kb_id,
                document_id,
                chunk_index,
                text,
                token_count,
                filename,
                kb_title,
                similarity
            FROM ranked_chunks
            WHERE rank <= {limit_per_kb}
            ORDER BY similarity DESC;
        """

        async with engine.begin() as conn:
            result = await conn.exec_driver_sql(sql)
            rows = result.mappings().all()
            return [dict(row) for row in rows]

    async def get_attached_kb_ids(self, chat_id: str) -> List[str]:
        """
        Get all knowledge base IDs attached to a chat session.

        Args:
            chat_id: Chat session ID

        Returns:
            List of KB IDs (as strings)
        """
        sql = f"""
            SELECT kb_id::text
            FROM chat_session_kbs
            WHERE chat_id = '{chat_id}'
            ORDER BY attached_at DESC;
        """

        async with engine.begin() as conn:
            result = await conn.exec_driver_sql(sql)
            rows = result.fetchall()
            return [row[0] for row in rows]

    async def get_chat_system_prompt(self, chat_id: str) -> Optional[str]:
        """
        Get the custom system prompt for a chat session.

        Args:
            chat_id: Chat session ID

        Returns:
            System prompt string or None
        """
        sql = f"""
            SELECT system_prompt
            FROM chat_sessions
            WHERE id = '{chat_id}';
        """

        async with engine.begin() as conn:
            result = await conn.exec_driver_sql(sql)
            row = result.fetchone()
            return row[0] if row and row[0] else None

    async def search_chat_attachments(
            self,
            vec: np.ndarray,
            chat_id: str,
            limit: int = 3,
            threshold: float | None = None
    ) -> List[dict]:
        """Search for similar chunks in chat attachments."""
        vec_literal = _vec_literal(vec)

        where_clause = f"WHERE c.chat_id = '{chat_id}'"

        if threshold is not None:
            where_clause += f" AND 1 - (embedding <=> '{vec_literal}'::vector) >= {threshold}"

        sql = f"""
            SELECT 
                c.id,
                c.attachment_id,
                c.chat_id,
                c.chunk_index,
                c.text,
                c.token_count,
                c.chunk_metadata,
                a.filename,
                1 - (c.embedding <=> '{vec_literal}'::vector) AS similarity
            FROM chat_attachment_chunks c
            JOIN chat_attachments a ON c.attachment_id = a.id
            {where_clause}
            ORDER BY similarity DESC
            LIMIT {limit};
        """

        async with engine.begin() as conn:
            result = await conn.exec_driver_sql(sql)
            rows = result.mappings().all()

            print(f"🐛 Attachment Search - Found {len(rows)} chunks")
            if rows:
                print(f"🐛 Attachment Search - Top similarity: {rows[0]['similarity']:.4f}")

            # Build metadata dict for compatibility with chat_pipelines
            results = []
            for row in rows:
                row_dict = dict(row)
                # Rename chunk_metadata to metadata for compatibility
                row_dict['metadata'] = row_dict.pop('chunk_metadata', {})
                # Add filename to metadata if not present
                if 'filename' not in row_dict['metadata']:
                    row_dict['metadata']['filename'] = row_dict.get('filename', 'Unknown')
                results.append(row_dict)

            return results