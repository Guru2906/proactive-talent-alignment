"""
backend/database.py
===================
Database setup and session management using SQLAlchemy ORM and SQLite.

The database stores application state, user profiles, assessment sessions,
answers, recommendation snapshots, and feedback.

ML datasets, models, and embeddings remain strictly file/artifact-based.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Generator
import yaml

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import declarative_base, sessionmaker, Session

# Determine database path
_THIS_DIR = Path(__file__).resolve().parent
BASE_DIR = _THIS_DIR.parent

# Read from config if available, else default to data/application/career_recommendation.db
_config_path = BASE_DIR / "config" / "pipeline_config.yaml"
_db_rel_path = "data/application/career_recommendation.db"

if _config_path.exists():
    try:
        with open(_config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
            _db_rel_path = cfg.get("paths", {}).get("database", _db_rel_path)
    except Exception:
        pass

_db_file = BASE_DIR / _db_rel_path
_db_file.parent.mkdir(parents=True, exist_ok=True)

# Allow environment variable override
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    f"sqlite:///{_db_file.as_posix()}"
)

# SQLite foreign key enforcement
@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    """Enable foreign key constraints for SQLite connections."""
    try:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()
    except Exception:
        pass

# SQLite specific settings: check_same_thread=False allows FastAPI multi-threading
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {},
    echo=False,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db_path() -> Path:
    """Return the resolved Path to the SQLite database file."""
    return _db_file


def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency that provides a transactional SQLAlchemy session.
    Automatically closes session after request finishes.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_session() -> Session:
    """Directly create a new database session (caller is responsible for closing)."""
    return SessionLocal()


def init_db(custom_engine=None) -> None:
    """
    Create all tables in the database if they don't already exist.
    """
    from backend import db_models  # Import to register models with Base.metadata
    target_engine = custom_engine or engine
    Base.metadata.create_all(bind=target_engine)
