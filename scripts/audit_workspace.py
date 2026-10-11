"""
scripts/audit_workspace.py
===========================
Generates:
1. PROJECT_FILE_AUDIT.csv
2. DATASET_AUDIT.csv
3. DATASET_CLASSIFICATION analysis
"""

from __future__ import annotations

import hashlib
import os
import re
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def get_file_hash(path: Path) -> str:
    try:
        h = hashlib.md5()
        with open(path, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return ""

def scan_all_files():
    records = []
    # Search all code for usage references
    all_code_text = {}
    for ext in [".py", ".ipynb", ".yaml", ".yml", ".json", ".md", ".html"]:
        for p in ROOT.rglob(f"*{ext}"):
            if ".venv" in p.parts or "__pycache__" in p.parts:
                continue
            try:
                all_code_text[str(p.relative_to(ROOT))] = p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                pass

    for p in ROOT.rglob("*"):
        if not p.is_file():
            continue
        if ".venv" in p.parts or "__pycache__" in p.parts or ".pytest_cache" in p.parts or ".git" in p.parts:
            continue

        rel_path = str(p.relative_to(ROOT)).replace("\\", "/")
        name = p.name
        ext = p.suffix.lower()
        size = p.stat().st_size
        fhash = get_file_hash(p) if size < 100 * 1024 * 1024 else "LARGE"

        # Check who references this file
        referenced_by = []
        for src_file, content in all_code_text.items():
            if src_file == rel_path:
                continue
            if name in content or rel_path in content:
                referenced_by.append(src_file)

        records.append({
            "path": rel_path,
            "filename": name,
            "file_type": ext,
            "size": size,
            "hash": fhash,
            "referenced_by_count": len(referenced_by),
            "referenced_by": "; ".join(referenced_by[:5]),
        })

    return pd.DataFrame(records)

if __name__ == "__main__":
    df = scan_all_files()
    print(f"Total files audited: {len(df)}")
    print(f"Unique hashes: {df['hash'].nunique()}")
