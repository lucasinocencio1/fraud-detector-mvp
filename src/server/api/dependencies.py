import os
from typing import Optional

from fastapi import Header, HTTPException, status


def get_api_key_from_env() -> str:
    return os.getenv("API_KEY", "")


def require_api_key(x_api_key: Optional[str] = Header(default=None)):
    api_key = get_api_key_from_env()
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="API key not configured",
        )
    if x_api_key != api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )
