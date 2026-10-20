import logging

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
        with get_db_context() as db:
            db.execute(text("SELECT 1"))
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    except Exception:
        logger.exception("Errore durante l'healthcheck di Anatema")
        return Response(status_code=status.HTTP_417_EXPECTATION_FAILED)