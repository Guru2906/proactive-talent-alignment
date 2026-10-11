"""
backend/data_loader.py
========================
Data loader for the new backend.

Loads:
  - Role catalog (data/role_catalog/emerging_roles.json)
  - Role index by role_id and role_name for fast lookups

Uses LRU cache for performance — catalog is loaded once at startup.
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

BASE = Path(__file__).resolve().parent.parent
CATALOG_PATH = BASE / "data" / "role_catalog" / "emerging_roles.json"


@lru_cache(maxsize=1)
def load_role_catalog() -> List[Dict[str, Any]]:
    """Load the role catalog JSON. Cached for performance."""
    if not CATALOG_PATH.exists():
        logger.error(
            f"Role catalog not found at {CATALOG_PATH}. "
            "Please run: python scripts/build_role_catalog.py"
        )
        return []
    with open(CATALOG_PATH, encoding="utf-8") as f:
        catalog = json.load(f)
    logger.info(f"Loaded role catalog: {len(catalog)} entries")
    return catalog


@lru_cache(maxsize=1)
def build_role_index() -> Dict[str, Dict[str, Any]]:
    """Build a lookup dict: role_id → role entry."""
    catalog = load_role_catalog()
    idx: Dict[str, Dict[str, Any]] = {}
    for role in catalog:
        rid = role.get("role_id")
        if rid:
            idx[rid] = role
        # Also index by normalized name for name-based lookups
        name = str(role.get("role_name", "")).strip()
        if name:
            idx[f"name:{name.lower()}"] = role
    return idx


def get_role_by_id(role_id: str) -> Optional[Dict[str, Any]]:
    """Look up a role by its role_id."""
    idx = build_role_index()
    return idx.get(role_id)


def get_role_by_name(name: str) -> Optional[Dict[str, Any]]:
    """Look up a role by its name (case-insensitive)."""
    idx = build_role_index()
    return idx.get(f"name:{name.lower().strip()}")


def get_emerging_roles() -> List[Dict[str, Any]]:
    """Return only emerging cluster roles."""
    return [r for r in load_role_catalog() if r.get("source_type") == "emerging_cluster"]


def get_known_roles() -> List[Dict[str, Any]]:
    """Return only known Pipeline-A roles."""
    return [r for r in load_role_catalog() if r.get("source_type") == "known_role"]
