# Master Data Dictionary & Variable Schema
## Proactive IT Role & Skill Emergence Forecaster

---

## 1. Role Demand Master Dataset (`role_demand_monthly.csv`)
- **Grain**: 1 row = 1 Canonical IT Role × 1 Month
- **Key Columns**:
  - `role` (string): Standardized Canonical IT Role name (e.g., `Generative AI Engineer`).
  - `month` (string, `YYYY-MM`): Forecast origin observation month (May 2023 – Oct 2025).
  - `month_idx` (int, 0–29): Chronological sequential integer index.
  - `lag1`, `lag2`, `lag3`, `lag6` (float, basis points): Historical role market share at $t-1, t-2, t-3, t-6$.
  - `roll3m`, `roll6m` (float, bps): Rolling moving average market share.
  - `mom1`, `mom3`, `mom6` (float): Percentage momentum change.
  - `slope6m` (float): Linear regression trend slope over past 6 months.
  - `accel` (float): Second derivative acceleration of market share.
  - `li_posting_count` (int): Total hiring postings in LinkedIn April 2024 snapshot.
  - `li_unique_companies` (int): Number of distinct hiring organizations.
  - `li_unique_industries` (int): Number of distinct hiring industry sectors.
  - `li_seniority_entropy` (float): Shannon entropy of junior/mid/senior distribution.
  - `li_frontier_skill_ratio` (float, 0.0–1.0): Proportion of frontier AI/ML skills in role profile.
  - `skill_ai_tool_count` (int): Count of modern AI tools in role stack.
  - `sk_vel` (float): Primary skill velocity index.
  - `era_novelty_score` (float, 0.0–1.0): Measure of role novelty relative to 2015–2017 pre-AI era.
  - `target_t6` (float, bps): Actual observed market share at $t+6$ (Training Target).

---

## 2. Role-Skill Evidence Matrix (`role_skill_evidence.csv`)
- **Grain**: 1 row = 1 Role × 1 Skill × 1 Source
- **Key Columns**:
  - `canonical_role` (string): Canonical Role identifier.
  - `canonical_skill` (string): Standardized skill name.
  - `posting_co_occurrences` (int): Total raw job postings containing both role and skill.
  - `market_penetration_rate` (float, 0.0–1.0): Fraction of role postings mentioning this skill.
  - `conditional_probability_P_skill_given_role` (float): Empirical probability $P(	ext{skill}|	ext{role})$.
  - `is_frontier_tech` (int, 0 or 1): Flag indicating GenAI, LLM, MLOps, or Agentic AI classification.
  - `so_2024_survey_validated` (int, 0 or 1): Flag indicating Stack Overflow 2024 developer adoption.

---

## 3. Occupational Employment History (`occupation_employment_history.csv`)
- **Grain**: 1 row = 1 SOC Code × 1 Year (2018–2025)
- **Key Columns**:
  - `year` (int): BLS OEWS survey reference year.
  - `soc_code` (string): Standard Occupational Classification code (e.g., `15-1252`).
  - `soc_title` (string): Official SOC occupation title (e.g., `Software Developers`).
  - `total_employment` (float): Total national US employment count.
  - `annual_mean_wage_usd` (float): Mean annual compensation in USD.

---

## 4. Future Occupation Projections (`future_occupation_projection.csv`)
- **Grain**: 1 row = 1 SOC Code × 1 Projection Period (2024–2034)
- **Key Columns**:
  - `2025 National Employment Matrix code` (string): SOC Code.
  - `2025 National Employment Matrix title` (string): Official occupation title.
  - `Employment, 2025` (float, thousands): Baseline national employment.
  - `Employment, 2035` (float, thousands): 10-year projected employment.
  - `Employment change, percent, 2025-35` (float): Projected 10-year percentage growth rate.
  - `Occupational openings, 2025-35 annual average` (float, thousands): Annual replacement openings.
