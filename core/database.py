from contextlib import contextmanager

from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from config import settings

target_db = f"{settings.POSTGRES_DB}_db"

SQLALCHEMY_DATABASE_URL = f"postgresql://{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{target_db}"

engine = create_engine(SQLALCHEMY_DATABASE_URL)


def ensure_database_exists():
    try:
        with engine.connect():
            pass
    except OperationalError as e:
        error_msg = str(e).lower()
        if "does not exist" in error_msg or getattr(e.orig, "pgcode", None) == "3D000":
            admin_url = f"postgresql://{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/postgres"
            admin_engine = create_engine(admin_url, isolation_level="AUTOCOMMIT")

            with admin_engine.connect() as conn:
                conn.execute(text(f'CREATE DATABASE "{target_db}"'))
        else:
            raise e


ensure_database_exists()

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


@contextmanager
def get_db_context():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
