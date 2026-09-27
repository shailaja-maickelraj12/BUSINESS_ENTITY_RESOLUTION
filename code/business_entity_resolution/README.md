# Business Entity Resolution Pipeline — Amazon ML Challenge 2026

## 1. Overview
This package provides a scalable, offline, machine-learning pipeline for large-scale Business Entity Resolution (ER) across heterogeneous, noisy data sources:
- **Reference Source 1**: Deduplicated ground-truth reference entity catalog.
- **Candidate Sources 2 & 3**: Noisy, multi-source records (zero, one, or multiple matches per S1 entity).
- **Supported Geographies**: United States, India, and France (open set of string country labels).

The solution optimizes for the competition evaluation metric: **Macro-average $F_{0.5}$** (precision-heavy, penalizing false merges $2\times$ over false negatives, with singletons included).

---

## 2. Pipeline Architecture

```
Raw Sources (S1, S2, S3)
         │
         ▼
[Phase 1: Normalization]
   - Legal suffix canonicalization (US: Inc, LLC, Corp; India: Pvt Ltd; France: SARL, SAS, SA, EURL)
   - Structured postal code, house number, and numeric token extraction
   - Unicode accent stripping and whitespace normalization
         │
         ▼
[Phase 2: Multi-Channel Candidate Blocking]
   - Channel 1: Exact core name & token signatures
   - Channel 2: Inverted index on rare name tokens (frequency thresholding)
   - Channel 3: Structured address matching (postal codes & house numbers)
   - Channel 4: Character 3-gram TF-IDF cosine retrieval
   - Channel 5: Combined name+address TF-IDF cosine retrieval
   (Recall Ceiling: 96.72%, Mean Candidates: ~34 per S1, Reduction Ratio: 99.966%)
         │
         ▼
[Phase 3: Pairwise Feature Engineering]
   - 34 discriminative features: Levenshtein, partial ratio, token sort/set ratios
   - Jaccard token overlaps, character n-gram similarities, prefix matches
   - Postal and house number exact match indicators (-1.0 missing, 0 mismatch, 1.0 match)
   - Group-level entity ranking (candidate rank, similarity difference to top candidate)
         │
         ▼
[Phase 4: Gradient Boosted Matching Model]
   - LightGBM binary classifier trained on group-split pairs
   - Calibrated decision threshold ($\tau = 0.69$) maximizing Macro-$F_{0.5}$
   - Precision guard for singletons (correctly predicting empty match yields 1.0)
         │
         ▼
[Phase 5: Streaming Test Inference]
   - Country-partitioned batching to strictly bound memory
   - Streams `output/matching_results.tsv` and `output/candidate_pairs.tsv`
```

---

## 3. Environment & Installation

Requirements: Python 3.10+ (tested on Python 3.13)

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 4. End-to-End Execution Guide

### Step 1: Run Unit Tests
Verify that normalization, blocking, and feature engineering modules pass all tests:
```bash
pytest tests/
```

### Step 2: Run Blocking Evaluation (Phase 2 Diagnostics)
Evaluates blocking recall ceilings and generates diagnostic reports:
```bash
python src/run_blocking.py
```
Outputs generated in `reports/`:
- `reports/phase2_blocking_report.md`
- `reports/phase2_candidate_statistics.tsv`
- `reports/phase2_missed_matches.tsv`

### Step 3: Train Matcher & Tune Macro-$F_{0.5}$ (Phase 4)
Trains the LightGBM classifier and tunes the decision threshold on validation entities:
```bash
python src/train.py --train-sample 8000 --val-sample 2000 --distractors 30000
```
Artifacts generated:
- Model: `artifacts/model/matcher_lgb.pkl`
- Config: `artifacts/model/model_config.json`
- Report: `reports/phase4_training_report.md`

### Step 4: Run Test Inference (Phase 5)
Processes test data and produces the required submission TSVs:
```bash
python src/predict.py
```
Deliverables produced in `output/`:
- `output/matching_results.tsv`
- `output/candidate_pairs.tsv`

### Step 5: Validate Submission Files
Run the official local submission validator:
```bash
python ../../student_resource/utils/validate_submission.py \
  --matching ../../output/matching_results.tsv \
  --candidate ../../output/candidate_pairs.tsv \
  --test-dir ../../student_resource/dataset/test
```
Must output `PASS`.
