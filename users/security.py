from fastapi import Header, HTTPException, status

from config import settings


async def verify_open_webui_secret_key(x_api_key: str = Header(None)):
    if any([not x_api_key, not x_api_key == settings.OPEN_WEBUI_SECRET_KEY]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Non autorizzato: API Key non valida o mancante",
        )
    return x_api_key
