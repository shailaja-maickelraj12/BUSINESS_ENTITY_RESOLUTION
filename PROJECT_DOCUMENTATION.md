# Amazon ML Challenge 2026: Business Entity Resolution
## Comprehensive Technical Documentation & Architecture Report

---

### Executive Metadata
- **Project Title:** High-Performance Business Entity Resolution Pipeline
- **Evaluation Metric:** Macro-average $F_{0.5}$ (Precision-Weighted, Singletons Included)
- **Supported Geographies:** United States, India, and France
- **System Constraints:** Offline Execution (No External APIs/Geocoding), Model $\le$ 8B Parameters (LightGBM GBDT)
- **Primary Deliverables:**
  - `output/matching_results.tsv` (Leaderboard Scoring File)
  - `output/candidate_pairs.tsv` (Candidate Evaluation File)
  - `Documentation_template.md` (Methodology Report)
  - `code/business_entity_resolution/` (Runnable Reproducible Pipeline)

---

## 1. Challenge & Problem Definition

### 1.1 The Entity Resolution (ER) Problem
In modern commercial ecosystems, business identity data originates from multiple independent, uncoordinated sources. Each source contributes noisy, incomplete, or formatted fragments of entity records without a shared unique primary key.

- **Source 1 (Reference Source):** Deduplicated reference dataset of business entities. Every entity in Source 1 must receive a prediction.
- **Source 2 & Source 3 (Candidate Sources):** Uncleaned, heterogeneous datasets containing business records.
- **Match Multiplicity:** A Source 1 entity may match zero records (singletons), exactly one record, or multiple records across Source 2 and Source 3.

### 1.2 The Open-World Geography Challenge
While training data is drawn from **United States** and **India**, the evaluation test set introduces a third country: **France**.
- The pipeline treats country as an open string label.
- Normalization and blocking channels generalize across jurisdictions without hardcoded geographic barriers.

### 1.3 Strict Evaluation Metric: Macro-Average $F_{0.5}$
The competition is evaluated using the macro-averaged $F_{0.5}$ score across all Source 1 entities:
$$\text{Precision} = \frac{|\text{Predicted} \cap \text{True}|}{|\text{Predicted}|}, \quad \text{Recall} = \frac{|\text{Predicted} \cap \text{True}|}{|\text{True}|}$$

$$F_{0.5} = \frac{(1 + 0.5^2) \times \text{Precision} \times \text{Recall}}{0.5^2 \times \text{Precision} + \text{Recall}} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$

#### Crucial Metric Nuances:
1. **Precision-Heavy Weighting ($2\times$):** In real-world entity resolution, incorrectly merging two distinct companies (false positive) is far more damaging than missing an association (false negative).
2. **Singleton Handling:** A Source 1 entity with zero true matches scores **1.0** if an empty list is predicted, and **0.0** if any match is predicted. Since 60–70% of entities are singletons, predicting matches on weak candidates devastates the macro-score.

---

## 2. End-to-End Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                 Phase 0: Data Ingestion & Audit             │
│  - TSV Parsing (sep='\t', string dtype, UTF-8)              │
│  - Entity-Grouped Stratified Train/Val Split                │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                 Phase 1: Normalization Engine               │
│  - Legal Suffix Mapping (US: Inc, LLC; IN: Pvt Ltd; FR: SAS)│
│  - Structured Address Parsing (Postal codes, House numbers) │
│  - Unicode Accent Removal & Punctuation Stripping           │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│            Phase 2: Multi-Channel Candidate Blocking        │
│  - Channel 1: Exact Core Name & Token Signatures            │
│  - Channel 2: Rare Token Inverted Index (DF <= 500)         │
│  - Channel 3: Structured Address Keys (PIN, House No.)      │
│  - Channel 4: Character 3-gram TF-IDF Cosine Retrieval     │
│  - Channel 5: Combined Name+Address TF-IDF Retrieval        │
│  => Recall: 96.72%, Reduction Ratio: 99.966%, ~34 Cands/S1  │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│         Phase 3: Pairwise & Group Feature Engineering       │
│  - 34 Pairwise Similarity & Overlap Features                │
│  - RapidFuzz Ratios (Levenshtein, Partial, Token Sort/Set)  │
│  - Address Numeric & Postal Verification Indicators         │
│  - Entity Group Ranking & Score Margins                     │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│         Phase 4: Gradient Boosted Matching & Tuning         │
│  - LightGBM Binary Classifier (Handling 1:10 Imbalance)     │
│  - Macro-F0.5 Grid Search (Optimal tau = 0.69)              │
│  - Precision Guard for Singletons                           │
│  => Macro-F0.5: 97.27%, Precision: 99.18%, Rec: 94.47%      │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│         Phase 5: Streaming Test Inference Engine            │
│  - Country-Partitioned Execution (France -> US -> India)    │
│  - Chunked S1 Processing (50k entities) to Bound Peak RAM   │
│  - Strict Invariant: matched_ids SUBSET OF candidate_ids    │
│  - Outputs: matching_results.tsv & candidate_pairs.tsv      │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│         Phase 6: Submission Verification & Packaging        │
│  - Validated with student_resource/utils/validate_submission│
│  - Documentation_template.md + Reproducibility README       │
│  - Final Standalone Submission Package (.zip)               │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Detailed Component Deep-Dive

### 3.1 Normalization Engine (`src/normalize.py`)
Entity names and addresses exhibit severe surface noise. Our normalizer applies deterministic, jurisdiction-aware transformations:

1. **Country Normalization:** Standardizes string aliases to canonical forms (`US`, `India`, `France`).
2. **Legal Suffix Canonicalization:**
   - **United States:** `inc`, `incorporated` $\to$ `inc`; `llc`, `l.l.c.` $\to$ `llc`; `corp`, `corporation` $\to$ `corp`.
   - **India:** `private limited`, `pvt ltd`, `pvt limited` $\to$ `pvt ltd`.
   - **France:** `sarl`, `s.a.r.l.` $\to$ `sarl`; `sas`, `s.a.s.` $\to$ `sas`; `sasu` $\to$ `sasu`; `sa` $\to$ `sa`; `eurl` $\to$ `eurl`.
   - Generates two name representations: `name_normalized` (full clean string) and `name_core` (string with legal suffix stripped).
3. **Structured Address Extraction:**
   - **Postal Codes:** Regular-expression extraction of US 5-digit ZIP codes (`^\d{5}$`), Indian 6-digit PIN codes (`^\d{6}$`), and French 5-digit codes.
   - **House / Unit Numbers:** Identifies leading building numbers (`123`, `G-21`, `Door No 4A`).
   - **Numeric Tokens:** Isolates all digit sequences for exact numerical intersection checks.

---

### 3.2 Multi-Channel Blocking (`src/blocking.py`)
Pairwise comparison of 1.73M S1 entities against 10M S2/S3 records requires evaluating $> 1.7 \times 10^{13}$ pairs, which is computationally intractable. Our candidate generator reduces this space by 99.966% using 5 complementary channels:

1. **Channel 1 — Exact Core Name & Token Signature:** Exact lookup on `name_core` and token signatures (alphabetically sorted unique tokens).
2. **Channel 2 — Rare Token Inverted Index:** Names are tokenized; tokens occurring in $> 500$ records are pruned as uninformative stopwords. For rare tokens, up to 10 candidates are retrieved.
3. **Channel 3 — Structured Address Keys:** Inverted index on postal codes and house numbers (frequency capped at 500).
4. **Channel 4 — Character 3-Gram TF-IDF Cosine Retrieval:** Sub-word character 3-gram retrieval with cosine similarity threshold $\ge 0.50$ (top-$k=5$). Captures misspellings, typographical errors, and minor phonetics.
5. **Channel 5 — Combined Text TF-IDF Retrieval:** Full-text representation with cosine similarity threshold $\ge 0.40$ (top-$k=3$). Captures multi-field associations where name and address are partially blended.

#### Measured Blocking Benchmarks:
| Iteration | Pair Recall Ceiling | Entity Recall Ceiling | Mean Cands/S1 | Reduction Ratio | Runtime |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Iteration A (Baseline Rules)** | 88.22% | 70.30% | 30.89 | 99.9696% | 2.0s |
| **Iteration B (+Char & Combined TF-IDF)** | **96.72%** | **91.24%** | **34.71** | **99.9659%** | **12.2s** |
| **Iteration C (Tuned Lean)** | 93.73% | 83.76% | 24.36 | 99.9760% | 13.2s |

- **US Recall Ceiling:** **99.23%**
- **India Recall Ceiling:** **92.78%**
- **Source 2 Recall:** **97.98%** | **Source 3 Recall:** **95.49%**

---

### 3.3 Feature Engineering (`src/features.py`)
For each candidate pair $(S_1, \text{Cand})$, our pipeline computes a 34-dimensional feature vector:

#### Group 1: Name Similarities (14 Features)
- `name_lev`: Full Levenshtein ratio.
- `name_partial`: Partial string matching ratio.
- `name_token_sort`: Token sort ratio (order-invariant matching).
- `name_token_set`: Token set ratio (subset/duplicate-invariant matching).
- `name_core_sort`: Token sort ratio on core name (suffix stripped).
- `name_jaccard`: Word token Jaccard similarity.
- `name_3gram_jaccard`, `name_4gram_jaccard`: Character n-gram Jaccard overlap.
- `prefix_3_match`, `prefix_5_match`: Binary match on leading 3 and 5 characters.
- `first_token_match`: Binary indicator whether primary entity token matches.
- `suffix_match`: Legal suffix agreement (1.0 = match, 0.5 = both empty, 0.0 = conflict).
- `name_len_diff`, `name_len_ratio`: String length disparity measures.

#### Group 2: Address Similarities (11 Features)
- `addr_lev`, `addr_token_sort`, `addr_token_set`: Address string distance ratios.
- `addr_jaccard`: Word token Jaccard overlap on address.
- `postal_match`: Tri-state match indicator (1.0 = match, 0.0 = mismatch, -1.0 = either missing).
- `house_match`: Tri-state match indicator on building/house number.
- `numeric_jaccard`: Jaccard similarity across all numeric tokens in addresses.
- `s1_addr_empty`, `cand_addr_empty`, `both_addr_empty`: Missingness indicators.
- `addr_len_ratio`: Address length compatibility.

#### Group 3: Context & Source Indicators (6 Features)
- `full_token_set`: Combined name + address token set ratio.
- `is_s2`, `is_s3`: Source file binary flags.
- `country_is_us`, `country_is_in`, `country_is_fr`: Geographic indicators.

#### Group 4: Entity-Level Group Ranking (3 Features)
- `candidate_rank`: Ordinal rank of candidate for this $S_1$ entity sorted by name similarity.
- `name_sort_diff_to_best`: Margin between this candidate's similarity and the top-ranked candidate for the same $S_1$ entity.
- `total_candidates_for_s1`: Candidate set cardinality (density/ambiguity measure).

---

### 3.4 Model Training & Threshold Calibration (`src/train.py`)
- **Classifier:** LightGBM Gradient Boosted Decision Tree (`LGBMClassifier`).
- **Configuration:** 350 trees, learning rate 0.06, max depth 6, 31 leaves, subsample 0.85, colsample_bytree 0.85.
- **Grouped Validation:** Split strictly grouped by $S_1$ entity ID to prevent data leakage.
- **Threshold Optimization:** Grid search $\tau \in [0.35, 0.90]$ with step $0.02$ evaluating the exact competition metric.

#### Empirical Validation Progression:
| Threshold ($\tau$) | Macro-$F_{0.5}$ | Macro-Precision | Macro-Recall | Singleton Accuracy | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 0.35 | 96.94% | 98.54% | 95.09% | 95.61% | Loose threshold, high recall |
| 0.45 | 97.13% | 98.81% | 94.93% | 96.49% | Balanced threshold |
| 0.55 | 97.11% | 98.88% | 94.78% | 96.49% | Standard 0.5 baseline |
| 0.65 | 97.26% | 99.10% | 94.65% | 97.37% | Precision-favored |
| **0.69** | **97.27%** | **99.18%** | **94.47%** | **97.37%** | **Optimal Macro-$F_{0.5}$** |
| 0.75 | 97.12% | 99.21% | 94.10% | 97.37% | Slight over-filtering |
| 0.85 | 96.98% | 99.35% | 93.52% | 98.25% | Aggressive filtering |

#### Top 10 Features by Split Importance:
1. `full_token_set` (685 splits)
2. `name_core_sort` (605 splits)
3. `name_lev` (531 splits)
4. `total_candidates_for_s1` (529 splits)
5. `name_partial` (501 splits)
6. `name_jaccard` (490 splits)
7. `addr_len_ratio` (476 splits)
8. `addr_token_set` (453 splits)
9. `candidate_rank` (451 splits)
10. `name_len_ratio` (437 splits)

---

### 3.5 Streaming Test Inference Engine (`src/predict.py`)
Processing 1.73M test reference entities across 10M candidate records requires careful memory management:
1. **Country Partitioning:** France, US, and India are processed sequentially. Since matches never cross national boundaries, candidate indexes are built and destroyed per country.
2. **Chunked Streaming:** Test $S_1$ entities are processed in 50,000-entity chunks. Outputs are appended directly to disk, keeping memory consumption bounded ($< 3.5\text{ GB}$ peak RAM).
3. **Subset Guarantee Enforcement:**
   $$\text{matched\_entity\_ids} \subseteq \text{candidate\_entity\_ids}$$
   The pipeline filters predictions against the exact generated candidate set before writing to disk.

---

## 4. Verification & Validation Protocol

The pipeline outputs were verified against the official submission validator:
```bash
python student_resource/utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir student_resource/dataset/test
```

### Verified Criteria:
- [x] Exact tab-separated structure (`\t`), no unintended quoting.
- [x] Exact required headers: `source1_entity_id \t matched_entity_ids` and `source1_entity_id \t candidate_entity_ids`.
- [x] Singletons formatted with empty string.
- [x] No duplicate entity IDs in candidate or match lists.
- [x] Zero self-matches (all matches are exclusively S2 or S3 IDs).
- [x] Strict subset property maintained across 100% of rows.
- [x] France records processed without geographic exception.

---

## 5. Final Submission Package Structure

```
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv         # Final entity matches (leaderboard scored)
│   └── candidate_pairs.tsv          # Candidate pairs from blocking stage
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       │   ├── normalize.py         # Normalization & legal suffix parser
│       │   ├── blocking.py          # Multi-channel candidate blocker
│       │   ├── blocking_diagnostics.py
│       │   ├── features.py          # 34-dimensional pairwise feature engine
│       │   ├── train.py             # LightGBM training & F0.5 threshold tuner
│       │   ├── predict.py           # Streaming test inference pipeline
│       │   ├── config.py            # Global paths & settings
│       │   ├── data_loader.py       # Deterministic TSV loaders
│       │   └── split.py             # Grouped train/val split generator
│       ├── tests/                   # 18 passing pytest test cases
│       ├── README.md                # Full execution instructions
│       └── requirements.txt         # Pinned production dependencies
└── Documentation_template.md        # Filled official methodology write-up
```

---

## 6. How to Reproduce

```bash
# 1. Activate Environment
.venv\Scripts\activate

# 2. Run All Unit Tests
pytest code/business_entity_resolution/tests/

# 3. Train Matcher and Tune Threshold
python code/business_entity_resolution/src/train.py

# 4. Run Test Inference
python code/business_entity_resolution/src/predict.py

# 5. Run Submission Validator
python student_resource/utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir student_resource/dataset/test
```
