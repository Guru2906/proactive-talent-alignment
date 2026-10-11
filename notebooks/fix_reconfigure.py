"""
fix_reconfigure.py
-------------------
Guards sys.stdout.reconfigure() calls in all src/ modules so they work
inside both a terminal (TextIOWrapper) and a Jupyter kernel (OutStream).
Run from project root:  python notebooks/fix_reconfigure.py
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from pathlib import Path

FILES = [
    "src/ranking/final_ranker.py",
    "src/preprocessing/job_cleaning.py",
    "src/forecasting/cluster_forecaster.py",
    "src/explainability/shap_analysis.py",
    "src/evaluation/leakage_tests.py",
    "src/evaluation/emerging_backtest.py",
    "src/evaluation/ablation.py",
    "src/emerging/emergence_score.py",
    "src/embeddings/sentence_embeddings.py",
    "src/clustering/hdbscan_discovery.py",
    "src/clustering/cluster_tracking.py",
    "src/clustering/cluster_labeling.py",
]

OLD = 'sys.stdout.reconfigure(encoding="utf-8", errors="replace")'
NEW = 'if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8", errors="replace")'

fixed = 0
for rel in FILES:
    p = Path(rel)
    if not p.exists():
        print(f"  MISSING: {rel}")
        continue
    text = p.read_text(encoding="utf-8")
    if OLD in text:
        p.write_text(text.replace(OLD, NEW), encoding="utf-8")
        print(f"  Fixed  : {rel}")
        fixed += 1
    else:
        print(f"  OK     : {rel}  (already guarded or different)")

print(f"\nDone.  {fixed}/{len(FILES)} files patched.")
