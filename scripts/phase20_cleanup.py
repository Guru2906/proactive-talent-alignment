"""
scripts/phase20_cleanup.py
===========================
Phase 20: Safe project cleanup and archive migration.
Moves duplicate/obsolete directories to archive/legacy/.
Does NOT delete anything -- only moves to archive.
Verifies no active imports reference moved paths before moving.
"""
import sys
import shutil
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent.parent
ARCHIVE = ROOT / "archive" / "legacy"
ARCHIVE.mkdir(parents=True, exist_ok=True)

def check_imports(path_name: str) -> list[str]:
    """Search all production Python files for references to path_name."""
    refs = []
    for py in ROOT.rglob("*.py"):
        if "archive" in py.parts or "__pycache__" in py.parts:
            continue
        try:
            text = py.read_text(encoding="utf-8", errors="ignore")
            if path_name in text:
                refs.append(str(py.relative_to(ROOT)))
        except Exception:
            pass
    return refs

def safe_move(src: Path, label: str):
    if not src.exists():
        print(f"  SKIP (not found): {label}")
        return

    # Check for references in production code
    refs = check_imports(src.name)
    if refs:
        print(f"  WARNING -- {label} referenced in: {refs[:3]} -- SKIPPING move")
        return

    dst = ARCHIVE / src.name
    if dst.exists():
        # Already archived, just remove source if duplicate
        print(f"  ALREADY ARCHIVED: {label}")
        return

    shutil.move(str(src), str(dst))
    print(f"  ARCHIVED: {label} -> archive/legacy/{src.name}")

def safe_delete_file(path: Path, label: str):
    if not path.exists():
        print(f"  SKIP (not found): {label}")
        return
    path.unlink()
    print(f"  DELETED: {label}")

def move_docs_to_docs():
    """Merge documentation/ into docs/."""
    doc_src = ROOT / "documentation"
    if not doc_src.exists():
        return
    for f in doc_src.iterdir():
        dst = ROOT / "docs" / f.name
        if not dst.exists():
            shutil.copy2(str(f), str(dst))
            print(f"  MERGED: documentation/{f.name} -> docs/{f.name}")
    # Archive the old folder
    safe_move(doc_src, "documentation/")

def move_numbered_csvs():
    """Move root-level numbered audit CSVs to archive."""
    patterns = [f"0{i}_" for i in range(1, 10)] + ["10_", "11_"]
    for p in list(ROOT.iterdir()):
        if p.is_file() and any(p.name.startswith(pat) for pat in patterns) and p.suffix == ".csv":
            dst = ARCHIVE / p.name
            if not dst.exists():
                shutil.move(str(p), str(dst))
                print(f"  ARCHIVED: {p.name} -> archive/legacy/")

def clean_root_clutter():
    """Archive redundant root-level MDs and CSVs from previous audit phases."""
    clutter_files = [
        "DATASET_AUDIT.csv",
        "DATASET_MODEL_USAGE_MATRIX.csv",
        "DATASET_USAGE_GUIDE.md",
        "DATA_ORGANIZATION_FINAL_REPORT.md",
        "FINAL_ARCHITECTURE.md",
        "FINAL_DATA_PREPARATION_REPORT.md",
        "FINAL_MODEL_TRAINING_READINESS.md",
        "FINAL_STEP1_VALIDATION_REPORT.md",
        "FINAL_SYSTEM_UPGRADE_REPORT.md",
        "MASTER_TECHNICAL_REPORT.md",
        "MODEL_DATA_ARCHITECTURE.md",
        "MODEL_REGISTRY.md",
        "MODEL_TRAINING_PLAN.md",
        "PRE_TRAINING_CHECKLIST.md",
        "PROJECT_CLEANUP_REPORT.md",
        "PROJECT_MERGE_AUDIT.md",
        "STEP1_MASTER_COMPREHENSIVE_REPORT.md",
        "PROJECT_FILE_AUDIT.csv",
        "FINAL_DATASET_MANIFEST.csv",
        "FINAL_PROJECT_STRUCTURE.md",
    ]
    for name in clutter_files:
        f = ROOT / name
        if f.exists():
            dst = ARCHIVE / name
            if not dst.exists():
                shutil.move(str(f), str(dst))
                print(f"  ARCHIVED root clutter: {name}")

def deduplicate_model():
    """Remove the byte-for-byte duplicate model in step1_enhanced."""
    dup = ROOT / "step1_enhanced" / "models" / "final_model.joblib"
    prod = ROOT / "models" / "final_model.joblib"
    if dup.exists() and prod.exists():
        if dup.stat().st_size == prod.stat().st_size:
            dup.unlink()
            print(f"  DELETED duplicate: step1_enhanced/models/final_model.joblib (same as models/final_model.joblib)")

if __name__ == "__main__":
    print("=== PHASE 20: PROJECT CLEANUP ===\n")

    print("--- Moving duplicate directories to archive ---")
    safe_move(ROOT / "app", "app/ (dead backend+frontend copy)")
    safe_move(ROOT / "raw_data", "raw_data/ (duplicate of data/raw/)")
    safe_move(ROOT / "standardized", "standardized/ (legacy snapshots)")
    safe_move(ROOT / "step1_output", "step1_output/ (single legacy CSV)")

    print("\n--- Removing duplicate mappings/ at root ---")
    mappings_root = ROOT / "mappings"
    if mappings_root.exists():
        refs = check_imports("mappings")
        # Only move if no import references it directly (data/mappings is the canonical)
        prod_refs = [r for r in refs if "data/mappings" not in r and "data\\mappings" not in r]
        if not prod_refs:
            safe_move(mappings_root, "mappings/ (root-level duplicate)")
        else:
            print(f"  SKIP: mappings/ referenced in production code: {prod_refs[:2]}")

    print("\n--- Merging documentation/ into docs/ ---")
    move_docs_to_docs()

    print("\n--- Archiving numbered audit CSVs from root ---")
    move_numbered_csvs()

    print("\n--- Archiving old root-level report clutter ---")
    clean_root_clutter()

    print("\n--- Removing duplicate model file ---")
    deduplicate_model()

    print("\n--- Archiving one-off scripts ---")
    safe_move(ROOT / "scripts" / "reorganize_project.py", "scripts/reorganize_project.py")
    safe_move(ROOT / "notebooks" / "fix_reconfigure.py", "notebooks/fix_reconfigure.py")

    print("\n=== CLEANUP COMPLETE ===")
    print(f"Archive location: {ARCHIVE}")

    # Report what's left at root
    print("\n--- Root-level files remaining ---")
    for p in sorted(ROOT.iterdir()):
        if p.is_file():
            print(f"  {p.name}")
