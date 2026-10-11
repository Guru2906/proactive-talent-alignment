# RECOMMENDATION ENGINE SPECIFICATION & SCORING FORMULAS

**Project:** Proactive IT Role & Skill Emergence Forecaster  
**Document:** Recommendation Engine Math & Scoring Specification  
**Version:** 2.0  
**Configuration File:** `config/recommendation_config.yaml`  

---

## 1. Master Recommendation Scoring Formula

For any role $R$ and user profile $U$, the **Final Recommendation Score** is computed as:

$$\text{Final Score}(U, R) = \sum_{i=1}^6 w_i \cdot S_i(U, R)$$

Where all weights satisfy $\sum_{i=1}^6 w_i = 1.0$, and all component scores $S_i \in [0, 1]$.

### Configurable Component Weights ($w_i$)

| Index | Component $S_i$ | Weight $w_i$ | Description |
|:---:|:---|:---:|:---|
| $S_1$ | **BAIO Archetype Fit** | $0.30$ | Cosine similarity between normalized user BAIO vector and role family BAIO profile |
| $S_2$ | **Interest Alignment** | $0.25$ | Overlap between user declared/inferred interests and role technical category |
| $S_3$ | **Work-Style Alignment** | $0.15$ | Match between user behavioral style (building, researching, operating) and role environment |
| $S_4$ | **Skill Alignment** | $0.15$ | Jaccard index between user's current skills and role required skills (adjusted for experience) |
| $S_5$ | **Future Opportunity** | $0.10$ | Normalized 6M/12M projected market demand growth from LightGBM / Holt forecasts |
| $S_6$ | **Emergence Signal** | $0.05$ | Discovery cluster emergence score (Pipeline B) or canonical emergence tier (Pipeline A) |

> [!IMPORTANT]
> **Emergence Signal Capping**: $w_6 = 0.05$ (capped at $\le 0.15$). This guarantees that high macroeconomic market emergence alone cannot artificially elevate a role to #1 if the personal fit ($S_1, S_2, S_3$) is low for that user.

---

## 2. Component Scoring Equations

### 2.1 BAIO Archetype Fit ($S_1$)

Let $v_u = [B_u, A_u, I_u, O_u]$ be the normalized user vector from the 20-question questionnaire ($\sum v_u = 1.0$), and $v_r = [B_r, A_r, I_r, O_r]$ be the target role family vector:

$$S_1(U, R) = \frac{1}{2} \left( \frac{v_u \cdot v_r}{\|v_u\|_2 \|v_r\|_2} + 1 \right)$$

### 2.2 Interest Alignment ($S_2$)

Let $I_u$ be the set of user interests and $M(I_u)$ be the mapped categories:

$$S_2(U, R) = \frac{\sum_{i \in I_u} \mathbb{I}(\text{RoleFamily}(R) \in M(i))}{|I_u|}$$

### 2.3 Skill Alignment ($S_3$)

Let $K_u$ be the user's declared technical skills and $K_r$ be the empirical required skills for role $R$:

$$J(K_u, K_r) = \frac{|K_u \cap K_r|}{|K_u \cup K_r|}$$

$$S_3(U, R) = \min(1.0, J(K_u, K_r) \times \text{ExpScale}(\text{ExperienceLevel}))$$

### 2.4 Future Opportunity ($S_4$)

Let $g$ be the projected percentage growth rate ($\text{growth\_pct\_6m}$ or cluster growth):

$$S_4(R) = \begin{cases} 
\max\left(0.0, 0.30 + \frac{g}{200}\right) & \text{if } g \le 0 \\
\min\left(1.0, 0.30 + 0.70 \times \frac{g}{100}\right) & \text{if } g > 0 
\end{cases}$$

---

## 3. The Three User-Facing Output Metrics

To ensure total transparency, the application displays three distinct scores for each recommendation rather than one opaque number:

### 1. 🎯 Personal Match %
Measures intrinsic fit with the user's passion, style, and natural problem-solving habits:
$$\text{Personal Match} = 0.50 \cdot S_1 + 0.30 \cdot S_2 + 0.20 \cdot S_3$$

### 2. 📈 Future Potential %
Measures external macroeconomic demand velocity and emergence:
$$\text{Future Potential} = 0.70 \cdot S_5 + 0.30 \cdot S_6$$

### 3. ⚡ Current Readiness %
Measures immediate competency overlap (how easily the user could enter this role today):
$$\text{Current Readiness} = S_4$$

---

## 4. Explainability Generation Logic

Each recommendation generates dynamic, rule-based explanations based on threshold evaluations:

- **Key Matching Strengths ($\ge 65\%$ threshold)**:
  - Strong Archetype alignment ($S_1 \ge 0.70$)
  - Specific domain passion alignment ($S_2 \ge 0.70$)
  - Exact skill overlaps ($\text{UserSkills} \cap \text{RoleSkills}$)
  - Strong market growth trajectory ($S_5 \ge 0.65$)
- **Challenges / Upskilling Needs**:
  - Missing core skills ($\text{RoleSkills} \setminus \text{UserSkills}$)
  - Frontier/advanced skills required for senior performance
  - Low initial readiness warning if $S_4 < 0.35$
