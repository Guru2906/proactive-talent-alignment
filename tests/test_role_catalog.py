"""
tests/test_role_catalog.py
===========================
Tests for the generated canonical role catalog.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest
from backend.data_loader import load_role_catalog, get_role_by_name, get_emerging_roles, get_known_roles

CATALOG_PATH = Path(__file__).resolve().parents[1] / "data" / "role_catalog" / "emerging_roles.json"


def test_catalog_file_exists():
    assert CATALOG_PATH.exists(), f"Catalog file not found at {CATALOG_PATH}"


def test_catalog_loads_and_non_empty():
    catalog = load_role_catalog()
    assert isinstance(catalog, list)
    assert len(catalog) >= 50, f"Expected at least 50 roles, got {len(catalog)}"


def test_catalog_contains_both_types():
    known = get_known_roles()
    emerging = get_emerging_roles()
    assert len(known) >= 30, f"Expected >= 30 known roles, got {len(known)}"
    assert len(emerging) >= 15, f"Expected >= 15 emerging cluster roles, got {len(emerging)}"


def test_catalog_schema_fields():
    catalog = load_role_catalog()
    required_keys = [
        "role_name",
        "source_type",
        "required_skills",
        "frontier_skills",
    ]
    for role in catalog:
        for k in required_keys:
            assert k in role, f"Missing key '{k}' in role '{role.get('role_name')}'"


def test_no_empty_role_names():
    catalog = load_role_catalog()
    for role in catalog:
        assert role["role_name"] and len(role["role_name"].strip()) > 0
