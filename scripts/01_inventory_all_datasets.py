"""
01_inventory_all_datasets.py
Scans the entire repository and catalogs every CSV, Excel, JSON, Parquet dataset.
Outputs: 01_DATASET_INVENTORY.csv
"""
import os, hashlib, json
import pandas as pd

BASE = r"."

def compute_md5(fp):
    h = hashlib.md5()
    with open(fp, 'rb') as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

records = []
for root, dirs, files in os.walk(BASE):
    if any(x in root for x in ['.git', '__pycache__', '.system_generated', 'archive']):
        continue
    for f in files:
        if f.endswith(('.csv', '.xlsx', '.xls', '.json', '.parquet')):
            fp = os.path.join(root, f)
            rel_p = os.path.relpath(fp, BASE)
            sz_mb = round(os.path.getsize(fp) / (1024*1024), 3)
            md5 = compute_md5(fp)
            ext = os.path.splitext(f)[1].lower()
            
            # Fast scan
            n_rows, n_cols, col_list, dtypes_str = "N/A", "N/A", "", ""
            min_date, max_date = "N/A", "N/A"
            u_roles, u_skills, u_companies = "N/A", "N/A", "N/A"
            geo = "Global / US / India"
            dup_pct, null_pct = 0.0, 0.0
            
            # Categorization
            nature = "Static Reference"
            purpose = "Lookup / Metadata"
            suitable_ml = "NO"
            recommended_usage = "REFERENCE"
            leakage_risk = "SAFE"
            decision = "USE AS REFERENCE"
            
            fn_low = f.lower()
            if 'adzuna' in fn_low:
                nature = "Monthly Time Series"
                purpose = "High-frequency IT hiring demand volume (30 continuous months)"
                suitable_ml = "YES"
                recommended_usage = "PRIMARY_TEMPORAL_BACKBONE"
                decision = "USE DIRECTLY"
                min_date, max_date = "2023-05", "2025-10"
            elif 'linkedin' in fn_low or 'postings' in fn_low:
                nature = "Cross-Sectional Snapshot"
                purpose = "Cross-sectional structural hiring, company breadth, and seniority"
                suitable_ml = "YES"
                recommended_usage = "STRUCTURAL_MARKET_PRIOR"
                decision = "USE AFTER TRANSFORMATION"
                min_date, max_date = "2024-04", "2024-04"
            elif 'naukri' in fn_low:
                nature = "Historical Snapshot"
                purpose = "Pre-AI baseline hiring (2015-2017) to quantify era novelty"
                suitable_ml = "YES"
                recommended_usage = "HISTORICAL_ERA_NOVELTY"
                decision = "USE AFTER TRANSFORMATION"
                min_date, max_date = "2015-01", "2017-12"
            elif 'skill_year' in fn_low or 'emerging_skill' in fn_low:
                nature = "Annual Longitude / Survey"
                purpose = "Developer skill adoption % and longitudinal skill velocity (CAGR)"
                suitable_ml = "YES"
                recommended_usage = "SKILL_VELOCITY_COVARIATE"
                decision = "USE DIRECTLY"
            elif 'national_m' in fn_low:
                nature = "Annual Historical Time Series"
                purpose = "BLS national occupational employment and wages (2018-2025)"
                suitable_ml = "YES"
                recommended_usage = "MACRO_HISTORICAL_VALIDATION"
                decision = "USE FOR VALIDATION"
                min_date, max_date = "2018", "2025"
            elif 'future' in fn_low or 'openings' in fn_low or 'projection' in fn_low:
                nature = "10-Year Macro Projection"
                purpose = "BLS 10-year projected employment growth and replacement openings"
                suitable_ml = "NO (Lookahead Risk)"
                recommended_usage = "LONG_TERM_VALIDATION_ONLY"
                leakage_risk = "FUTURE INFORMATION (Exclude from Training Features)"
                decision = "USE FOR VALIDATION"
                min_date, max_date = "2024", "2034"
            elif 'current_roles' in fn_low or 'onet' in fn_low:
                nature = "Occupational Taxonomy"
                purpose = "O*NET standard occupational skills and technology density"
                suitable_ml = "YES"
                recommended_usage = "TAXONOMIC_PRIOR"
                decision = "USE AS REFERENCE"
            
            try:
                if ext == '.csv':
                    df_sample = pd.read_csv(fp, nrows=5)
                    n_cols = len(df_sample.columns)
                    col_list = '; '.join(df_sample.columns.tolist()[:10])
                    with open(fp, 'rb') as lf:
                        n_rows = sum(1 for _ in lf) - 1
                elif ext in ['.xlsx', '.xls']:
                    df_sample = pd.read_excel(fp, nrows=5)
                    n_cols = len(df_sample.columns)
                    col_list = '; '.join(str(c) for c in df_sample.columns.tolist()[:10])
                    n_rows = "Excel Multi-Row"
            except:
                pass
                
            records.append({
                'filename': f,
                'exact_path': rel_p,
                'file_size_mb': sz_mb,
                'md5_hash': md5,
                'row_count': n_rows,
                'column_count': n_cols,
                'columns_sample': col_list,
                'temporal_nature': nature,
                'dataset_purpose': purpose,
                'min_date': min_date,
                'max_date': max_date,
                'geographic_coverage': geo,
                'suitable_for_ml': suitable_ml,
                'recommended_usage': recommended_usage,
                'leakage_risk': leakage_risk,
                'final_decision': decision
            })

inv_df = pd.DataFrame(records)
inv_df.to_csv(os.path.join(BASE, "01_DATASET_INVENTORY.csv"), index=False)
inv_df.to_csv(os.path.join(BASE, "data_governance/MASTER_DATASET_INVENTORY.csv"), index=False)
print(f"01_DATASET_INVENTORY.csv created with {len(inv_df)} entries.")