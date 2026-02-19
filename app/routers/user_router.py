# app/routers/user_router.py
from fastapi import APIRouter, HTTPException, status, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from app.db.database import get_db
from app.db.models import User
from app.models.schemas import UserCreate

router = APIRouter()


# -------------------------------------------------------------------------
# 1. Login - Auto-create if new username (UPDATED)
# -------------------------------------------------------------------------
@router.post("/users/login", status_code=status.HTTP_200_OK)
async def login_user(
        username: str = Query(..., description="Username to login with"),
        db: AsyncSession = Depends(get_db)
):
    """
    Login with username. Auto-creates user if username doesn't exist.
    This enables self-registration on first login.
    """
    # Check if user exists
    result = await db.execute(select(User).where(User.username == username))
    user = result.scalars().first()

    if user:
        # User exists - normal login
        return {
            "id": str(user.id),
            "username": user.username,
            "created_at": user.created_at,
            "is_new_user": False
        }

    # User doesn't exist - create new user automatically
    new_user = User(username=username)
    db.add(new_user)
    await db.commit()
    await db.refresh(new_user)

    return {
        "id": str(new_user.id),
        "username": new_user.username,
        "created_at": new_user.created_at,
        "is_new_user": True,  # Flag to show "Welcome!" message in frontend
        "message": "Account created successfully!"
    }


# -------------------------------------------------------------------------
# 2. Create user (admin only - for manual creation if needed)
# -------------------------------------------------------------------------
@router.post("/users", status_code=status.HTTP_201_CREATED)
async def create_user(
        payload: UserCreate,
        db: AsyncSession = Depends(get_db)
):
    """
    Manually create a new user (admin endpoint).
    Optional - since login now auto-creates users.
    """
    # Check if username already exists
    existing = await db.execute(select(User).where(User.username == payload.username))
    if existing.scalars().first():
        raise HTTPException(
            status_code=400,
            detail=f"Username '{payload.username}' already exists"
        )

    user = User(username=payload.username)
    db.add(user)
    await db.commit()
    await db.refresh(user)

    return {
        "id": str(user.id),
        "username": user.username,
        "created_at": user.created_at
    }


# -------------------------------------------------------------------------
# 3. Get all users (for admin/testing)
# -------------------------------------------------------------------------
@router.get("/users")
async def list_users(db: AsyncSession = Depends(get_db)):
    """List all users in the system."""
    result = await db.execute(select(User))
    users = result.scalars().all()

    return [
        {
            "id": str(user.id),
            "username": user.username,
            "created_at": user.created_at
        }
        for user in users
    ]


# -------------------------------------------------------------------------
# 4. Get user by ID
# -------------------------------------------------------------------------
@router.get("/users/{user_id}")
async def get_user(user_id: str, db: AsyncSession = Depends(get_db)):
    """Get user details by ID."""
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalars().first()

    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    return {
        "id": str(user.id),
        "username": user.username,
        "created_at": user.created_at
    }