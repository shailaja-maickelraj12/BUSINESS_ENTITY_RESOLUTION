"""Phase 0 EDA Script: Business Entity Resolution Dataset Analysis.

Loads train/test files, computes shapes, null counts, country distributions,
ground-truth match distributions, checks for source multiplicity,
extracts random and hard positive pairs side-by-side, samples France records,
and tests entity_id numbering leakage.
"""

import os
import sys
import gc
import json
import random
import numpy as np
import pandas as pd
from collections import Counter
from rapidfuzz import fuzz

random.seed(42)
np.random.seed(42)

DATA_DIR = "student_resource/dataset"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")

def log(msg):
    print(f"[EDA] {msg}", flush=True)

def analyze_files():
    log("=== PART 1: FILE SHAPES, NULLS, AND COUNTRY DISTRIBUTIONS ===")
    files = {
        "train_source1": os.path.join(TRAIN_DIR, "train_source1.tsv"),
        "train_source2": os.path.join(TRAIN_DIR, "train_source2.tsv"),
        "train_source3": os.path.join(TRAIN_DIR, "train_source3.tsv"),
        "train_ground_truth": os.path.join(TRAIN_DIR, "train_ground_truth.tsv"),
        "test_source1": os.path.join(TEST_DIR, "test_source1.tsv"),
        "test_source2": os.path.join(TEST_DIR, "test_source2.tsv"),
        "test_source3": os.path.join(TEST_DIR, "test_source3.tsv"),
    }

    file_stats = {}
    for name, path in files.items():
        log(f"Reading {name}...")
        df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
        nrows, ncols = df.shape
        empty_counts = {col: int((df[col] == "").sum()) for col in df.columns}
        
        stat = {
            "path": path,
            "rows": nrows,
            "cols": ncols,
            "columns": list(df.columns),
            "empty_counts": empty_counts,
        }
        if "country" in df.columns:
            country_dist = df["country"].value_counts().to_dict()
            stat["country_dist"] = country_dist
            log(f"  {name}: {nrows:,} rows, {ncols} cols. Countries: {country_dist}")
        else:
            log(f"  {name}: {nrows:,} rows, {ncols} cols.")
        file_stats[name] = stat
        del df
        gc.collect()

    return file_stats

def analyze_ground_truth():
    log("\n=== PART 2: GROUND TRUTH MATCH DISTRIBUTIONS ===")
    gt_path = os.path.join(TRAIN_DIR, "train_ground_truth.tsv")
    gt = pd.read_csv(gt_path, sep="\t", dtype=str, keep_default_na=False)
    
    total_s1 = len(gt)
    # Parse matched_entity_ids
    match_counts = []
    s2_counts = []
    s3_counts = []
    both_sources_count = 0
    multi_s2_count = 0
    multi_s3_count = 0
    
    for _, row in gt.iterrows():
        raw_matches = row["matched_entity_ids"].strip()
        if not raw_matches:
            matches = []
        else:
            matches = [m.strip() for m in raw_matches.split(",") if m.strip()]
        
        n_m = len(matches)
        match_counts.append(n_m)
        
        s2_m = [m for m in matches if m.startswith("S2-")]
        s3_m = [m for m in matches if m.startswith("S3-")]
        s2_counts.append(len(s2_m))
        s3_counts.append(len(s3_m))
        
        if len(s2_m) > 0 and len(s3_m) > 0:
            both_sources_count += 1
        if len(s2_m) > 1:
            multi_s2_count += 1
        if len(s3_m) > 1:
            multi_s3_count += 1

    match_dist = Counter(match_counts)
    log(f"Total S1 entities in Ground Truth: {total_s1:,}")
    log(f"Singletons (0 matches): {match_dist[0]:,} ({match_dist[0] / total_s1 * 100:.2f}%)")
    log(f"1 match: {match_dist[1]:,} ({match_dist[1] / total_s1 * 100:.2f}%)")
    log(f"2 matches: {match_dist[2]:,} ({match_dist[2] / total_s1 * 100:.2f}%)")
    log(f"3 matches: {match_dist[3]:,} ({match_dist[3] / total_s1 * 100:.2f}%)")
    log(f"4+ matches: {sum(v for k, v in match_dist.items() if k >= 4):,} ({sum(v for k, v in match_dist.items() if k >= 4) / total_s1 * 100:.2f}%)")
    log(f"Max matches for a single S1 entity: {max(match_counts)}")
    
    total_pos_pairs = sum(match_counts)
    total_s2_matches = sum(s2_counts)
    total_s3_matches = sum(s3_counts)
    log(f"Total positive pairs (S1 -> S2/S3): {total_pos_pairs:,}")
    log(f"  S2 matches: {total_s2_matches:,} ({total_s2_matches / total_pos_pairs * 100:.2f}%)")
    log(f"  S3 matches: {total_s3_matches:,} ({total_s3_matches / total_pos_pairs * 100:.2f}%)")
    log(f"S1 matching BOTH S2 and S3: {both_sources_count:,} ({both_sources_count / total_s1 * 100:.2f}% of S1, {both_sources_count / (total_s1 - match_dist[0]) * 100:.2f}% of non-singletons)")
    log(f"S1 matching >1 record from S2: {multi_s2_count:,} ({multi_s2_count / total_s1 * 100:.4f}%)")
    log(f"S1 matching >1 record from S3: {multi_s3_count:,} ({multi_s3_count / total_s1 * 100:.4f}%)")
    
    return gt, match_dist

def sample_pairs_and_noise(gt):
    log("\n=== PART 3: PAIR SAMPLES & NOISE PATTERNS ===")
    # Load S1, S2, S3 for training
    log("Loading train sources for pair lookup...")
    s1 = pd.read_csv(os.path.join(TRAIN_DIR, "train_source1.tsv"), sep="\t", dtype=str, keep_default_na=False).set_index("entity_id")
    s2 = pd.read_csv(os.path.join(TRAIN_DIR, "train_source2.tsv"), sep="\t", dtype=str, keep_default_na=False).set_index("entity_id")
    s3 = pd.read_csv(os.path.join(TRAIN_DIR, "train_source3.tsv"), sep="\t", dtype=str, keep_default_na=False).set_index("entity_id")
    
    # Expand ground truth into (s1_id, candidate_id)
    pairs = []
    for _, row in gt.iterrows():
        s1_id = row["source1_entity_id"]
        raw = row["matched_entity_ids"].strip()
        if raw:
            for m in raw.split(","):
                m = m.strip()
                if m:
                    pairs.append((s1_id, m))
                    
    log(f"Total positive pairs expanded: {len(pairs):,}")
    
    # Let's inspect country cross-over
    cross_country = 0
    sampled_indices = random.sample(range(len(pairs)), min(10000, len(pairs)))
    for idx in sampled_indices:
        s1_id, m_id = pairs[idx]
        s1_rec = s1.loc[s1_id]
        m_rec = s2.loc[m_id] if m_id.startswith("S2-") else s3.loc[m_id]
        if s1_rec["country"] != m_rec["country"]:
            cross_country += 1
    log(f"Cross-country matches in sample of 10,000: {cross_country} (Must be 0 if country is 100% consistent)")
    
    # Sample 20 random positive pairs
    random_indices = random.sample(range(len(pairs)), 20)
    random_pairs_data = []
    for idx in random_indices:
        s1_id, m_id = pairs[idx]
        s1_rec = s1.loc[s1_id]
        m_rec = s2.loc[m_id] if m_id.startswith("S2-") else s3.loc[m_id]
        name_sim = fuzz.token_sort_ratio(s1_rec["business_name"], m_rec["business_name"])
        addr_sim = fuzz.token_sort_ratio(s1_rec["business_address"], m_rec["business_address"])
        random_pairs_data.append({
            "s1_id": s1_id, "s1_name": s1_rec["business_name"], "s1_addr": s1_rec["business_address"], "country": s1_rec["country"],
            "m_id": m_id, "m_name": m_rec["business_name"], "m_addr": m_rec["business_address"],
            "name_sim": name_sim, "addr_sim": addr_sim
        })
        
    # Search for 10 "hard-looking" positive pairs (low token_sort_ratio on name or address)
    log("Scanning for hard positive pairs...")
    hard_pairs_data = []
    candidate_indices = random.sample(range(len(pairs)), min(50000, len(pairs)))
    scored_candidates = []
    for idx in candidate_indices:
        s1_id, m_id = pairs[idx]
        s1_rec = s1.loc[s1_id]
        m_rec = s2.loc[m_id] if m_id.startswith("S2-") else s3.loc[m_id]
        name_sim = fuzz.token_sort_ratio(s1_rec["business_name"], m_rec["business_name"])
        addr_sim = fuzz.token_sort_ratio(s1_rec["business_address"], m_rec["business_address"])
        combined_score = 0.6 * name_sim + 0.4 * addr_sim
        scored_candidates.append((combined_score, name_sim, addr_sim, s1_id, m_id, s1_rec, m_rec))
        
    # Sort ascending by combined score to get lowest similarity true matches
    scored_candidates.sort(key=lambda x: x[0])
    for item in scored_candidates[:10]:
        comb, n_sim, a_sim, s1_id, m_id, s1_rec, m_rec = item
        hard_pairs_data.append({
            "s1_id": s1_id, "s1_name": s1_rec["business_name"], "s1_addr": s1_rec["business_address"], "country": s1_rec["country"],
            "m_id": m_id, "m_name": m_rec["business_name"], "m_addr": m_rec["business_address"],
            "name_sim": n_sim, "addr_sim": a_sim
        })
        
    # Check ID leakage:
    log("\n=== PART 4: ENTITY ID LEAKAGE TEST ===")
    s1_nums = []
    m_nums = []
    diffs = []
    for idx in candidate_indices[:5000]:
        s1_id, m_id = pairs[idx]
        try:
            s1_num = int(s1_id.split("-")[1])
            m_num = int(m_id.split("-")[1])
            s1_nums.append(s1_num)
            m_nums.append(m_num)
            diffs.append(abs(s1_num - m_num))
        except Exception:
            pass
            
    corr = np.corrcoef(s1_nums, m_nums)[0, 1] if len(s1_nums) > 1 else 0
    # Also compare with random fake pairs
    fake_s2_nums = random.sample(m_nums, len(m_nums))
    fake_diffs = [abs(a - b) for a, b in zip(s1_nums, fake_s2_nums)]
    log(f"Correlation between S1 numeric ID and matched ID: {corr:.4f}")
    log(f"Mean |S1_num - Match_num| for true matches: {np.mean(diffs):.1f} (std: {np.std(diffs):.1f})")
    log(f"Mean |S1_num - Random_num| for random pairs: {np.mean(fake_diffs):.1f} (std: {np.std(fake_diffs):.1f})")
    min_diff = min(diffs) if diffs else 0
    exact_id_matches = sum(1 for d in diffs if d == 0)
    log(f"Exact numerical ID matches in 5,000 true pairs: {exact_id_matches} ({exact_id_matches/5000*100:.2f}%)")

    # Sample France records from Test
    log("\n=== PART 5: FRANCE TEST SAMPLES ===")
    test_s1 = pd.read_csv(os.path.join(TEST_DIR, "test_source1.tsv"), sep="\t", dtype=str, keep_default_na=False)
    france_s1 = test_s1[test_s1["country"] == "France"].head(10).to_dict(orient="records")
    del test_s1
    gc.collect()

    test_s2 = pd.read_csv(os.path.join(TEST_DIR, "test_source2.tsv"), sep="\t", dtype=str, keep_default_na=False)
    france_s2 = test_s2[test_s2["country"] == "France"].head(5).to_dict(orient="records")
    del test_s2
    gc.collect()

    test_s3 = pd.read_csv(os.path.join(TEST_DIR, "test_source3.tsv"), sep="\t", dtype=str, keep_default_na=False)
    france_s3 = test_s3[test_s3["country"] == "France"].head(5).to_dict(orient="records")
    del test_s3
    gc.collect()

    # Save all output to a json file for detailed reporting
    out = {
        "random_pairs": random_pairs_data,
        "hard_pairs": hard_pairs_data,
        "id_leakage": {
            "corr": float(corr),
            "mean_true_diff": float(np.mean(diffs)),
            "mean_random_diff": float(np.mean(fake_diffs)),
            "exact_id_matches": exact_id_matches
        },
        "france_samples": {
            "s1": france_s1,
            "s2": france_s2,
            "s3": france_s3
        }
    }
    with open("phase0_results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    log("Saved detailed samples to phase0_results.json")

def main():
    log("Starting Phase 0 EDA...")
    analyze_files()
    gt, match_dist = analyze_ground_truth()
    sample_pairs_and_noise(gt)
    log("Phase 0 EDA completed successfully!")

if __name__ == "__main__":
    main()
