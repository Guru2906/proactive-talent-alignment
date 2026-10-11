"""
05_build_skill_mapping.py
Builds the canonical skill taxonomy across 91 standard skills with category, AI/frontier tags.
Outputs: 04_SKILL_MAPPING.csv & mappings/skill_mapping.csv
"""
import os
import pandas as pd

BASE = r"."

so_path = os.path.join(BASE, "raw_data/external_knowledge/emerging_skill_clusters_8yr.csv")
so_df = pd.read_csv(so_path)

rows = []
for _, r in so_df.iterrows():
    sk = str(r['Skill']).strip()
    cl = int(r['Cluster'])
    cat = str(r['Category'])
    adp = float(r['2024'])
    
    is_ai = 1 if cat == 'AITool' or cl in [3, 4] or any(k in sk.lower() for k in ['ai', 'copilot', 'gpt', 'gemini', 'claude']) else 0
    is_front = 1 if is_ai or cl in [3, 4] or adp >= 30.0 else 0
    
    rows.append({
        'source_dataset': 'Stack Overflow 2024 Developer Survey + LinkedIn Relational Skills',
        'original_skill': sk,
        'canonical_skill': sk,
        'skill_category': cat,
        'skill_cluster_id': cl,
        'adoption_percentage_2024': adp,
        'is_ai_related': is_ai,
        'is_frontier_skill': is_front,
        'mapping_confidence': 1.00,
        'mapping_method': 'Official Stack Overflow 2024 Standardized Vocabulary'
    })

df = pd.DataFrame(rows)
df.to_csv(os.path.join(BASE, "04_SKILL_MAPPING.csv"), index=False)
df.to_csv(os.path.join(BASE, "mappings/skill_mapping.csv"), index=False)
print(f"04_SKILL_MAPPING.csv generated ({len(df)} canonical skills).")