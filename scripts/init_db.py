# scripts/init_db.py
import asyncio
from app.db.database import engine
from app.db.models import Base

async def init():
    async with engine.begin() as conn:
        # create pgvector extension before creating tables
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        # optionally ensure pgcrypto if you used gen_random_uuid
        await conn.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto;")
        await conn.run_sync(Base.metadata.create_all)
    print("DB initialized.")

if __name__ == "__main__":
    asyncio.run(init())
