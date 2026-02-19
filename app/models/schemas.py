from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime
from typing import List, Optional, Dict, Any
from uuid import UUID


# -------------------------------------------------------------------------
# CHAT SCHEMAS
# -------------------------------------------------------------------------
class ChatCreate(BaseModel):
    title: Optional[str] = Field(default=None, description="Chat title")
    system_prompt: Optional[str] = Field(
        default=None, description="Optional system prompt for initializing the assistant behavior"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "title": "Research helper",
                    "system_prompt": "You are an expert research assistant who provides concise answers."
                }
            ]
        }
    )


class MessageCreate(BaseModel):
    role: str = Field(..., description="Message role, e.g., user or assistant")
    content: str = Field(..., min_length=1, description="User message text")

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {"role": "user", "content": "Explain FastAPI in three sentences"}
            ]
        }
    )


class MessageRead(BaseModel):
    id: int
    role: str
    content: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ChatSummaryRead(BaseModel):
    chat_id: str
    messages: list[MessageRead]


class UserCreate(BaseModel):
    username: str


class UserRead(BaseModel):
    id: UUID
    username: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


# -------------------------------------------------------------------------
# KNOWLEDGE BASE SCHEMAS
# -------------------------------------------------------------------------
class KnowledgeBaseCreate(BaseModel):
    title: str = Field(..., description="Title of the knowledge base")
    description: Optional[str] = Field(default=None, description="Optional description")

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {"title": "KB_GENAI_PROJECTS", "description": "Projects related to generative ai"}
            ]
        }
    )


class KnowledgeBaseRead(BaseModel):
    id: UUID
    user_id: UUID
    title: str
    description: Optional[str]
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


# -------------------------------------------------------------------------
# DOCUMENTS + CHUNKS
# -------------------------------------------------------------------------
class KBDocumentCreate(BaseModel):
    filename: str
    mime_type: Optional[str] = None
    text_sha256: str
    text_size: Optional[int] = None


class KBDocumentRead(BaseModel):
    id: UUID
    kb_id: UUID
    filename: Optional[str]
    mime_type: Optional[str]
    text_sha256: str
    text_size: Optional[int]
    doc_metadata: Optional[Dict[str, Any]]
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class KBChunkRead(BaseModel):
    id: UUID
    document_id: UUID
    chunk_index: int
    text: str
    token_count: Optional[int]
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


# -------------------------------------------------------------------------
# CHAT ↔ KB RELATION
# -------------------------------------------------------------------------
class ChatSessionKBLink(BaseModel):
    chat_id: UUID
    kb_id: UUID
    attached_at: datetime

    model_config = ConfigDict(from_attributes=True)