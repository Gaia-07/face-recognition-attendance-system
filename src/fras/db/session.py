"""Database engine configuration and session dependency providers."""

import os
from collections.abc import Generator
from pathlib import Path

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from fras.db.models import Base

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent.parent.parent / "data" / "fras.db"

if os.getenv("VERCEL") == "1":
    tmp_db = Path("/tmp/fras.db")
    if not tmp_db.exists() and DEFAULT_DB_PATH.exists():
        import shutil

        try:
            shutil.copyfile(DEFAULT_DB_PATH, tmp_db)
        except Exception:
            pass
    DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{tmp_db}")
else:
    DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DEFAULT_DB_PATH}")

# Ensure directory exists for SQLite
if DATABASE_URL.startswith("sqlite:///"):
    sqlite_file = Path(DATABASE_URL.replace("sqlite:///", ""))
    sqlite_file.parent.mkdir(parents=True, exist_ok=True)

engine: Engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
    echo=False,
)


SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db(custom_engine: Engine | None = None) -> None:
    """Creates all database tables defined in metadata."""
    target_engine = custom_engine or engine
    Base.metadata.create_all(bind=target_engine)


def get_db() -> Generator[Session, None, None]:
    """Dependency generator yielding an active database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
