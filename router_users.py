import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from auth import get_current_user
import dal_users


logger = logging.getLogger('app')
router = APIRouter(prefix="/users", tags=["users"])


class UserCreate(BaseModel):
    user_name: str = Field(..., min_length=1, max_length=50)
    email: str
    password: str = Field(..., min_length=4, max_length=100)


class UserUpdate(BaseModel):
    user_name: Optional[str] = None
    email: Optional[str] = None
    password: Optional[str] = None


@router.post("", status_code=201)
def create_new_user(user: UserCreate):
    logger.info(f"✎ POST /users - Creating account: {user.user_name}")
    result = dal_users.insert_user(user.user_name, user.email, user.password)

    if result is None:
        logger.warning(f"❌ POST /users - Duplicate username: {user.user_name}")
        raise HTTPException(status_code=400, detail="Username already exists")

    logger.info(f"✔️ POST /users - Account created: {user.user_name}")
    return {"message": "User created successfully", "user": result}


@router.get("", response_model=list)
@router.get("/", response_model=list)
def get_users():
    logger.info(f"👥 GET /users - Listing all accounts")
    result = dal_users.get_all_users()
    logger.info(f"✔️ GET /users - Found {len(result)} accounts")
    return result


@router.get("/{user_id}")
def get_user(user_id: int):
    logger.info(f"🔍 GET /users/{user_id} - Retrieving account")
    user = dal_users.get_user_by_id(user_id)
    if user is None:
        logger.warning(f"❌ GET /users/{user_id} - Account not found")
        raise HTTPException(status_code=404, detail="User not found")
    logger.info(f"✔️ GET /users/{user_id} - Account found: {user.get('user_name')}")
    return user


@router.put("/{user_id}")
def update_existing_user(user_id: int, user_data: UserUpdate):
    logger.info(f"📝 PUT /users/{user_id} - Modifying account")
    current = dal_users.get_user_by_id(user_id)
    if current is None:
        logger.warning(f"❌ PUT /users/{user_id} - Account not found")
        raise HTTPException(status_code=404, detail="User not found")

    new_username = user_data.user_name or current["user_name"]
    new_email = user_data.email or current["email"]
    new_password = user_data.password

    result = dal_users.update_user(user_id, new_username, new_email, new_password)

    if result is None:
        logger.warning(f"❌ PUT /users/{user_id} - Modification failed")
        raise HTTPException(status_code=404, detail="User not found")
    if result == "duplicate":
        raise HTTPException(status_code=400, detail="Username or email already exists")

    logger.info(f"✔️ PUT /users/{user_id} - Account updated")
    return result


@router.delete("/{user_id}")
def delete_existing_user(user_id: int, current_user=Depends(get_current_user)):
    if current_user["id"] != user_id:
        logger.warning(f"❌ DELETE /users/{user_id} - Unauthorized deletion attempt by {current_user['user_name']}")
        raise HTTPException(status_code=403, detail="Can only delete own account")

    logger.info(f"🔄 DELETE /users/{user_id} - Removing account")
    deleted = dal_users.delete_user(user_id)
    if deleted is None:
        logger.warning(f"❌ DELETE /users/{user_id} - Account not found")
        raise HTTPException(status_code=404, detail="User not found")

    logger.info(f"✔️ DELETE /users/{user_id} - Account deleted")
    return {"message": "User deleted successfully", "user": deleted}


@router.delete("/table/recreate")
def recreate_users_table():
    dal_users.recreate_table_users()
    return {"message": "Users table dropped and recreated successfully"}
