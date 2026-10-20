import logging

import httpx
from fastapi import FastAPI, Response, status
from sqlalchemy import text

from chat.router.v1 import router as chat_router
from config import settings
from core.database import get_db_context
from core.lifespan import lifespan

logger = logging.getLogger(__name__)


# FastAPI main entrypoint. Starts routes and lifespan
app = FastAPI(
    title="Anatema API",
    lifespan=lifespan,
    docs_url=settings.DOC_URL,
    redoc_url=settings.REDOC_URL,
)

app.include_router(chat_router, prefix="/api/v1")


@app.get("/health")
async def health_check():
    try:
        local_mcp_urls = [
            "http://imap_mcp:8000/health",
            "http://weather_mcp:8000/health",
            "http://caldav_mcp:3000/health",
            "http://knowledge_mcp:8000/health",
        ]
        async with httpx.AsyncClient() as client:
            results = [await client.get(url, timeout=10) for url in local_mcp_urls]
        with get_db_context() as db:
            db.execute(text("SELECT 1"))
            results.append(True)
    except Exception:
        logger.exception("Errore durante l'healthcheck di Anatema")
        results = [False]
    if all(results):
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    return Response(status_code=status.HTTP_417_EXPECTATION_FAILED)
