import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

import dal_users
from auth import create_access_token


logger = logging.getLogger('app')
router = APIRouter(prefix="/auth", tags=["auth"])


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=255)
    password: str = Field(..., min_length=1)


def flush_logs():
    for handler in logging.getLogger().handlers:
        handler.flush()


@router.post("/login")
def login(credentials: LoginRequest):
    logger.info(f"🔐 POST /auth/login - Authenticating: {credentials.username}")
    try:
        if not dal_users.login_user(credentials.username, credentials.password):
            logger.warning(f"🚫 POST /auth/login - Authentication denied: {credentials.username}")
            flush_logs()
            raise HTTPException(status_code=401, detail="Invalid username or password")

        user = dal_users.get_user_by_username(credentials.username)
        
        if not user:
            logger.error(f"⚠️ POST /auth/login - User not found post-auth: {credentials.username}")
            flush_logs()
            raise HTTPException(status_code=500, detail="Authentication error")
        
        token = create_access_token(credentials.username)
        logger.info(f"✔️ POST /auth/login - User authenticated: {credentials.username}")
        flush_logs()
        return {"access_token": token, "token_type": "bearer", "user_name": user["user_name"], "id": user["id"]}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"⚠️ POST /auth/login - Exception: {str(e)}", exc_info=True)
        flush_logs()
        raise HTTPException(status_code=500, detail=f"Login error: {str(e)}")
