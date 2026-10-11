"""
================================================================================
PRE-TRAINING DATA ENGINEERING MASTER PIPELINE ORCHESTRATOR
Project: Proactive IT Role & Skill Emergence Forecaster
================================================================================
Builds the complete modular pre-training ecosystem across 11 discrete scripts:
  - 01_inventory_all_datasets.py
  - 02_profile_datasets.py
  - 03_clean_standardize.py
  - 04_build_role_mapping.py
  - 05_build_skill_mapping.py
  - 06_temporal_alignment.py
  - 07_leakage_audit.py
  - 08_dataset_join_analysis.py
  - 09_build_feature_ready_tables.py
  - 10_validate_master_data.py
  - 11_generate_data_documentation.py
"""

import os, sys, hashlib, math, json, shutil
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

BASE = r"."

# ── 1. SETUP DIRECTORY STRUCTURE ──────────────────────────────────────────────
directories = [
    os.path.join(BASE, "data/raw"),
    os.path.join(BASE, "data/external"),
    os.path.join(BASE, "data/cleaned"),
    os.path.join(BASE, "data/standardized"),
    os.path.join(BASE, "data/mappings"),
    os.path.join(BASE, "data/features"),
    os.path.join(BASE, "data/validation"),
    os.path.join(BASE, "data/archive"),
    os.path.join(BASE, "mappings"),
    os.path.join(BASE, "validation"),
    os.path.join(BASE, "scripts")
]

for d in directories:
    os.makedirs(d, exist_ok=True)

print("Initialized target directory structure in data/, mappings/, validation/, and scripts/.")
