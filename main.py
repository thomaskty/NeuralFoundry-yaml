import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from sqlalchemy import text

from app.routers import user_router, chat_router, kb_router, chat_kb_router,attachment_router
from app.db.database import engine


logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)
logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup and shutdown events.
    Runs once when the app starts and when it shuts down.
    """
    # Startup: Ensure extensions, create tables, ensure indexes exist
    print("🚀 Initializing database...")
    await ensure_extensions()
    await create_tables()
    await ensure_indexes()
    print("✅ Database ready")

    yield

    # Shutdown
    print("👋 Shutting down...")


async def create_tables():
    """Create all database tables if they don't exist"""
    from app.db.models import Base

    async with engine.begin() as conn:
        # Create all tables defined in models
        await conn.run_sync(Base.metadata.create_all)

    print("✅ Tables created/verified")


async def ensure_extensions():
    """Create required extensions if they don't exist"""
    async with engine.begin() as conn:
        try:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        except Exception as e:
            print(f"⚠️  Extension creation warning: {e}")


async def ensure_indexes():
    """
    Create indexes if they don't exist.
    Uses IF NOT EXISTS so it's safe to run multiple times.
    """
    indexes = [
        "CREATE INDEX IF NOT EXISTS idx_chat_messages_session_id ON chat_messages (session_id);",
        "CREATE INDEX IF NOT EXISTS idx_kb_chunks_kb_id ON kb_chunks (kb_id);",
        "CREATE INDEX IF NOT EXISTS idx_kb_chunks_document_id ON kb_chunks (document_id);",
        "CREATE INDEX IF NOT EXISTS idx_kb_documents_kb_filename ON kb_documents (kb_id, filename);",
        "CREATE INDEX IF NOT EXISTS idx_chat_session_kbs_chat_id ON chat_session_kbs (chat_id);",
        "CREATE INDEX IF NOT EXISTS idx_chat_session_kbs_kb_id ON chat_session_kbs (kb_id);",
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_chat_session_kbs_unique ON chat_session_kbs (chat_id, kb_id);",
    ]

    # Vector indexes (these might fail if not enough data, that's OK)
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

    async with engine.begin() as conn:
        # Create basic indexes
        for sql in indexes:
            try:
                await conn.execute(text(sql))
            except Exception as e:
                print(f"⚠️  Index creation warning: {e}")

        # Try vector indexes (may fail if insufficient data)
        for sql in vector_indexes:
            try:
                await conn.execute(text(sql))
            except Exception as e:
                if "not enough" in str(e).lower() or "ivfflat" in str(e).lower():
                    # This is expected when there's not enough data
                    pass
                else:
                    print(f"⚠️  Vector index warning: {e}")


# Create FastAPI app with lifespan
app = FastAPI(
    title="Neural Foundry API",
    description="RAG-powered chat API with knowledge bases",
    version="1.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(user_router.router, prefix="/api/v1", tags=["Users"])
app.include_router(chat_router.router, prefix="/api/v1", tags=["Chats"])
app.include_router(kb_router.router, prefix="/api/v1", tags=["Knowledge Bases"])
app.include_router(chat_kb_router.router, prefix="/api/v1", tags=["Chat-KB Links"])
app.include_router(attachment_router.router, prefix="/api/v1", tags=["Chat Attachments"])


@app.get("/")
async def root():
    return {
        "message": "Neural Foundry API",
        "status": "running",
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test database connection
        async with engine.begin() as conn:
            await conn.execute(text("SELECT 1"))
        return {"status": "healthy", "database": "connected"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
