# Phase 0: Comprehensive Data Audit & Exploratory Data Analysis

## 1. Overview & Dataset Shapes

The Amazon ML Challenge 2026 dataset involves multi-source business entity resolution across three sources:
- **Source 1 (`S1-`):** Deduplicated reference source.
- **Source 2 (`S2-`):** Independent business records.
- **Source 3 (`S3-`):** Independent business records.

### Complete File Inventory

| Dataset File | Rows | Columns | Empty `business_name` | Empty `business_address` | Empty `country` | Country Distribution |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `train_source1.tsv` | 2,206,821 | 4 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) | US: 1,323,633 (59.98%), India: 883,188 (40.02%) |
| `train_source2.tsv` | 5,034,616 | 4 | 0 (0.00%) | 168,967 (3.36%) | 0 (0.00%) | US: 3,016,817 (59.92%), India: 2,017,799 (40.08%) |
| `train_source3.tsv` | 5,285,603 | 4 | 0 (0.00%) | 175,916 (3.33%) | 0 (0.00%) | US: 3,170,056 (59.98%), India: 2,115,547 (40.02%) |
| `train_ground_truth.tsv` | 2,206,821 | 2 | — | — | — | Ground truth pairs for all 2,206,821 S1 entities |
| `test_source1.tsv` | 1,732,544 | 4 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) | India: 809,986 (46.75%), US: 663,106 (38.27%), France: 259,452 (14.98%) |
| `test_source2.tsv` | 4,887,273 | 4 | 0 (0.00%) | 129,408 (2.65%) | 0 (0.00%) | India: 2,312,565 (47.32%), US: 1,871,330 (38.29%), France: 703,378 (14.39%) |
| `test_source3.tsv` | 5,082,316 | 4 | 0 (0.00%) | 136,098 (2.68%) | 0 (0.00%) | India: 2,405,000 (47.32%), US: 1,945,701 (38.28%), France: 731,615 (14.39%) |

---

## 2. Missing Value & Null Patterns

1. **`business_name`**: 0 missing values across all train and test files (100% presence).
2. **`country`**: 0 missing values across all train and test files (100% presence).
3. **`business_address`**:
   - `train_source1.tsv` & `test_source1.tsv`: 100% complete (0 missing).
   - `train_source2.tsv`: 168,967 (3.36%) records have empty address strings (`""`).
   - `train_source3.tsv`: 175,916 (3.33%) records have empty address strings (`""`).
   - `test_source2.tsv`: 129,408 (2.65%) records have empty address strings (`""`).
   - `test_source3.tsv`: 136,098 (2.68%) records have empty address strings (`""`).
   - **Ground truth analysis**: Out of 69,144 sampled positive pairs, **3,013 (4.36%) true positive matches** have an empty address in the matched S2 or S3 record.

**Decision:** Blocking must not strictly require address matching. At least one candidate generation channel must rely on core business name similarity. Feature engineering must include `is_address_missing` flags.

---

## 3. Ground Truth Matching Characteristics

From `train_ground_truth.tsv` (2,206,821 Source 1 entities):

- **Total Ground Truth positive pairs:** 7,638,365 pairs (mean: 3.46 matches per S1 entity; max matches: 11).
- **Match counts distribution:**
  - **0 matches (Singletons):** 123,247 (5.58%)
  - **1 match:** 119,157 (5.40%)
  - **2 matches:** 375,212 (17.00%)
  - **3 matches:** 530,841 (24.05%)
  - **4+ matches:** 1,058,364 (47.96%)
- **Source split of matched records:**
  - Source 2 (`S2-`): 3,693,619 (48.36%)
  - Source 3 (`S3-`): 3,944,746 (51.64%)
- **Multi-source overlap:**
  - S1 entities matching **BOTH** S2 and S3: 1,776,047 (80.48% of all S1, 85.24% of non-singletons).
- **Intra-source multiplicity check:**
  - S1 matching **>1 record from S2**: 1,129,968 (51.20% of S1 entities)
  - S1 matching **>1 record from S3**: 1,224,128 (55.47% of S1 entities)

**Critical Decision:** We must **NOT** enforce a limit of at most one match per source. Enforcing a 1-match-per-source heuristic would discard over half of all valid matches.

---

## 4. Cross-Country Consistency

- 10,000 true positive pairs were checked across S1 and their ground-truth matches:
  - **0 pairs cross country boundaries (0.00%)**.
  - S1 US entities only match S2/S3 US entities.
  - S1 India entities only match S2/S3 India entities.

**Decision:** Candidate generation and matching can be strictly partitioned by country (`country == country`). This is completely lossless and drastically reduces candidate search space.

---

## 5. Entity ID Leakage Check

- Pearson correlation between numeric S1 IDs and numeric match IDs: **-0.0124**
- Mean absolute difference between S1 ID and matched ID: **335,170,951** (std: 236,473,392)
- Mean absolute difference for random pairs: **330,696,360** (std: 235,145,215)
- Exact numerical match count in 5,000 true pairs: **0 (0.00%)**

**Decision:** IDs are purely synthetic hash keys. The pipeline must not use numeric ID values as features.

---

## 6. Observed Noise Patterns

### A. US Records
- **Legal suffixes:** `Inc.`, `Incorporated`, `LLC`, `L.L.C.`, `Corp`, `Corporation`, `PC`, `Co.`.
- **Street abbreviations:** `Rd`/`Road`, `St`/`Street`, `Ave`/`Avenue`, `Dr`/`Drive`, `Cir`/`Circle`, `Blvd`/`Boulevard`.
- **Address reordering:** City, state, street, unit numbers in permuted sequences.
- **Accented characters:** Occasional un-stripped accents (e.g., `Ínc.`, `Wéalth`).

### B. India Records
- **Prefixes:** `M/s`, `Sri`, `Shri`, `The`.
- **Legal forms:** `Private Limited`, `Pvt Ltd`, `LLP`, `Limited`, `Enterprises`, `Services`.
- **Address complexity:** Landmark references ("Behind Bikaner Sweet", "Opp. Railway St."), floor/plot numbering ("D - 7", "Shop No. 25, Plot No.15/A"), transliterated scripts (e.g. Gurmukhi characters).
- **Missing components:** Frequent absence of standard PIN codes or street names.

### C. France Test Records
- **Legal forms:** `SARL`, `SASU`, `SAS`, `EURL`, `SCI`, `SNC`, `SA`, `Association de`.
- **Street types:** `Rue`, `R.`, `Avenue`, `AV`, `Boulevard`, `BD`, `Allée`, `Impasse`, `Chemin`, `Place`, `Quai`.
- **Accents & diacritics:** `é`, `è`, `ê`, `à`, `ç`, `ô`, `î`, `ë`.
- **Numbering formats:** `5 bis`, `NO. 5`, `(41)`.
