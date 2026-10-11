"""
backend/init_db.py
==================
Database initialization CLI utility.

Creates the SQLite database directory if needed, creates all tables,
verifies schema integrity, and prints status.

Usage:
  python -m backend.init_db
"""

from __future__ import annotations

import sys
from pathlib import Path
from sqlalchemy import inspect

# Ensure workspace root in path
_BASE_DIR = Path(__file__).resolve().parent.parent
if str(_BASE_DIR) not in sys.path:
    sys.path.insert(0, str(_BASE_DIR))

from backend.database import engine, get_db_path, init_db
import backend.db_models  # Ensure models are registered with Base.metadata


def initialize_database() -> None:
    """Initialize application SQLite database and print schema details."""
    db_path = get_db_path()
    print(f"============================================================")
    print(f"  INITIALIZING SQLITE APPLICATION DATABASE")
    print(f"============================================================")
    print(f"Target DB Path: {db_path.resolve()}")

    # Ensure parent directory exists
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # Initialize tables
    init_db(engine)

    # Inspect created tables
    inspector = inspect(engine)
    tables = inspector.get_table_names()

    print(f"Status: Database initialized successfully.")
    print(f"Created/Verified Tables ({len(tables)}):")
    for t in sorted(tables):
        cols = [c["name"] for c in inspector.get_columns(t)]
        fks = [f"{fk['constrained_columns']} -> {fk['referred_table']}.{fk['referred_columns']}" for fk in inspector.get_foreign_keys(t)]
        fk_str = f" | FKs: {', '.join(fks)}" if fks else ""
        print(f"  • {t:26} ({len(cols)} columns: {', '.join(cols[:4])}...{fk_str})")

    print(f"============================================================")


if __name__ == "__main__":
    initialize_database()
