"""End-to-End Test Inference Pipeline for Business Entity Resolution.

Loads test sets across US, India, and France, generates candidates via blocking,
extracts pairwise features, scores candidates using the trained LightGBM model,
applies the optimal F0.5 decision threshold, and streams formatted TSV deliverables:
  - output/matching_results.tsv
  - output/candidate_pairs.tsv
"""

import argparse
import gc
import json
import os
import pickle
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

SRC_DIR = os.path.dirname(__file__)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from config import (
    BASE_DIR,
    OUTPUT_CANDIDATES_PATH,
    OUTPUT_MATCHING_PATH,
    TEST_SOURCE1_PATH,
    TEST_SOURCE2_PATH,
    TEST_SOURCE3_PATH,
    NormalizationConfig,
)
from blocking import BlockingConfig, MultiSourceBlocker
from features import compute_features_for_candidates
from normalize import normalize_dataframe


def load_test_pool_for_country(
    country: str,
    max_cands: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load and normalize candidate records from test S2 and S3 for a specific country."""
    norm_cfg = NormalizationConfig()
    print(f"[Predict] Loading candidate pool for country '{country}'...", flush=True)

    def read_country_subset(path: str, label: str) -> pd.DataFrame:
        chunks = []
        for chunk in pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, chunksize=500000):
            filtered = chunk[chunk["country"] == country]
            if len(filtered) > 0:
                chunks.append(filtered)
            if max_cands and sum(len(c) for c in chunks) >= max_cands:
                break
        df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=["entity_id", "business_name", "business_address", "country"])
        if max_cands and len(df) > max_cands:
            df = df.head(max_cands).reset_index(drop=True)
        print(f"[Predict]   - {label} ({country}): {len(df):,} records loaded.", flush=True)
        return normalize_dataframe(df, norm_cfg)

    s2_df = read_country_subset(TEST_SOURCE2_PATH, "Source 2")
    s3_df = read_country_subset(TEST_SOURCE3_PATH, "Source 3")
    return s2_df, s3_df


def run_inference(
    sample_limit_per_country: Optional[int] = None,
    chunk_size_s1: int = 50000,
    max_pool_records: Optional[int] = None,
):
    """Run full test inference and stream outputs to matching_results.tsv and candidate_pairs.tsv."""
    t_start = time.time()
    norm_cfg = NormalizationConfig()

    # 1. Load trained model & configuration
    model_path = os.path.join(BASE_DIR, "artifacts", "model", "matcher_lgb.pkl")
    cfg_path = os.path.join(BASE_DIR, "artifacts", "model", "model_config.json")

    if not os.path.isfile(model_path) or not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"Trained model not found at {model_path}. Run train.py first.")

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(cfg_path, "r", encoding="utf-8") as f:
        model_config = json.load(f)

    threshold = model_config.get("optimal_threshold", 0.69)
    feature_cols = model_config["feature_cols"]
    print(f"[Predict] Loaded LightGBM model. Decision threshold: tau = {threshold:.2f}, Features: {len(feature_cols)}", flush=True)

    # 2. Read full test S1 reference entities
    print(f"[Predict] Loading Test Source 1 reference entities from {TEST_SOURCE1_PATH}...", flush=True)
    t0_s1 = time.time()
    s1_all = pd.read_csv(TEST_SOURCE1_PATH, sep="\t", dtype=str, keep_default_na=False)
    original_s1_order = list(s1_all["entity_id"])
    print(f"[Predict] Loaded {len(s1_all):,} Test S1 entities across {s1_all['country'].nunique()} countries in {time.time() - t0_s1:.1f}s.", flush=True)

    # 3. Prepare Output Files with exact headers
    os.makedirs(os.path.dirname(OUTPUT_MATCHING_PATH), exist_ok=True)
    with open(OUTPUT_MATCHING_PATH, "w", encoding="utf-8") as f_match:
        f_match.write("source1_entity_id\tmatched_entity_ids\n")

    with open(OUTPUT_CANDIDATES_PATH, "w", encoding="utf-8") as f_cand:
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")

    # Countries to process
    countries = ["France", "US", "India"]
    total_processed_s1 = 0
    total_matches_predicted = 0
    total_candidates_generated = 0
    total_singletons = 0

    cfg_block = BlockingConfig(
        use_exact_core_name=True,
        use_exact_full_name=True,
        use_token_signature=True,
        use_rare_tokens=True,
        rare_token_max_df=1000,
        max_cands_per_rare_token=15,
        use_postal_code=True,
        postal_code_max_df=1000,
        max_cands_per_postal=20,
        use_house_number=True,
        house_number_max_df=1000,
        max_cands_per_house=20,
        use_char_ngram=False,
        use_combined_tfidf=False,
        max_total_candidates_per_entity=25,
    )


    for country in countries:
        print(f"\n{'=' * 60}", flush=True)
        print(f"[Predict] PROCESSING COUNTRY: {country.upper()}", flush=True)
        print(f"{'=' * 60}", flush=True)

        # Filter S1 entities for country
        s1_country_all = s1_all[s1_all["country"] == country]
        if sample_limit_per_country:
            s1_country_all = s1_country_all.head(sample_limit_per_country).copy()

        n_s1_country = len(s1_country_all)
        print(f"[Predict] S1 entities to resolve for {country}: {n_s1_country:,}", flush=True)
        if n_s1_country == 0:
            continue

        # Load candidate pool
        t0 = time.time()
        s2_pool, s3_pool = load_test_pool_for_country(country, max_cands=max_pool_records)

        # Build candidate lookup
        cand_lookup = {}
        if len(s2_pool) > 0:
            cand_lookup.update(s2_pool.set_index("entity_id").to_dict(orient="index"))
        if len(s3_pool) > 0:
            cand_lookup.update(s3_pool.set_index("entity_id").to_dict(orient="index"))

        # Build blocker index
        blocker = MultiSourceBlocker(cfg_block)
        if len(s2_pool) > 0 or len(s3_pool) > 0:
            blocker.build_indexes(s2_pool, s3_pool)
        print(f"[Predict] Candidate indexing for {country} complete in {time.time() - t0:.1f}s.", flush=True)

        # Process Test S1 in chunks
        country_s1_processed = 0
        n_chunks = (n_s1_country + chunk_size_s1 - 1) // chunk_size_s1

        for chunk_idx in range(n_chunks):
            s1_chunk = s1_country_all.iloc[chunk_idx * chunk_size_s1 : (chunk_idx + 1) * chunk_size_s1].copy()

            t_chunk_start = time.time()
            s1_country_norm = normalize_dataframe(s1_chunk, norm_cfg)
            s1_lookup = s1_country_norm.set_index("entity_id").to_dict(orient="index")
            s1_ids = list(s1_chunk["entity_id"])

            # 1. Blocking
            if len(cand_lookup) > 0:
                cand_dict, _ = blocker.block_dataframe(s1_country_norm)
            else:
                cand_dict = {sid: [] for sid in s1_ids}

            # 2. Features & Scoring
            matched_dict = {sid: [] for sid in s1_ids}

            # Filter non-empty candidates for feature computation
            entities_with_cands = {sid: cands for sid, cands in cand_dict.items() if len(cands) > 0}
            if entities_with_cands and len(cand_lookup) > 0:
                feat_df = compute_features_for_candidates(entities_with_cands, s1_lookup, cand_lookup)
                if len(feat_df) > 0:
                    X = feat_df[feature_cols].values
                    probs = model.predict_proba(X)[:, 1]
                    mask = probs >= threshold
                    if np.any(mask):
                        hits_s1 = feat_df["s1_id"].values[mask]
                        hits_cand = feat_df["candidate_id"].values[mask]
                        hits_p = probs[mask]
                        s1_match_scores = defaultdict(list)
                        for sid, cid, p in zip(hits_s1, hits_cand, hits_p):
                            s1_match_scores[sid].append((cid, p))
                        for sid, cand_tuples in s1_match_scores.items():
                            cand_tuples.sort(key=lambda x: x[1], reverse=True)
                            matched_dict[sid] = list(dict.fromkeys([c for c, _ in cand_tuples]))

            # 3. Stream to Output Files
            with open(OUTPUT_MATCHING_PATH, "a", encoding="utf-8") as f_match, \
                 open(OUTPUT_CANDIDATES_PATH, "a", encoding="utf-8") as f_cand:
                for sid in s1_ids:
                    cands = cand_dict.get(sid, [])
                    matches = matched_dict.get(sid, [])
                    # Enforce subset guarantee: matches must be subset of candidates
                    cands_set = set(cands)
                    valid_matches = [m for m in matches if m in cands_set]

                    cand_str = ",".join(cands)
                    match_str = ",".join(valid_matches)

                    f_match.write(f"{sid}\t{match_str}\n")
                    f_cand.write(f"{sid}\t{cand_str}\n")

                    total_candidates_generated += len(cands)
                    total_matches_predicted += len(valid_matches)
                    if len(valid_matches) == 0:
                        total_singletons += 1

            country_s1_processed += len(s1_chunk)
            total_processed_s1 += len(s1_chunk)
            print(f"[Predict]   Chunk {chunk_idx + 1}/{n_chunks}: Processed {len(s1_chunk):,} entities in {time.time() - t_chunk_start:.1f}s (Country Total: {country_s1_processed:,} / {n_s1_country:,})", flush=True)

        # Free country memory
        del s2_pool, s3_pool, cand_lookup, blocker
        gc.collect()

    # If full inference (no sample limit), re-order outputs to match exact test_source1.tsv row order
    if not sample_limit_per_country and total_processed_s1 == len(original_s1_order):
        print("\n[Predict] Re-ordering output TSVs to match exact test_source1.tsv entity order...", flush=True)
        t0_sort = time.time()
        order_index = {eid: idx for idx, eid in enumerate(original_s1_order)}

        for file_path, header in [(OUTPUT_MATCHING_PATH, "source1_entity_id\tmatched_entity_ids"),
                                  (OUTPUT_CANDIDATES_PATH, "source1_entity_id\tcandidate_entity_ids")]:
            with open(file_path, "r", encoding="utf-8") as f:
                next(f)  # skip header
                lines = [line.rstrip("\n").split("\t", 1) for line in f if line.strip()]
            # Sort according to original order
            lines.sort(key=lambda x: order_index.get(x[0], 999999999))
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(f"{header}\n")
                for s1_id, val in lines:
                    f.write(f"{s1_id}\t{val}\n")
        print(f"[Predict] Re-ordering complete in {time.time() - t0_sort:.1f}s.", flush=True)

    t_total = time.time() - t_start
    print(f"\n{'=' * 60}", flush=True)
    print(f"[Predict] INFERENCE COMPLETED SUCCESSFULLY IN {t_total:.1f}s", flush=True)
    print(f"{'=' * 60}", flush=True)
    print(f"  - Total Source 1 entities processed: {total_processed_s1:,}")
    print(f"  - Total candidate pairs produced: {total_candidates_generated:,} ({total_candidates_generated / max(total_processed_s1, 1):.2f} cands/S1)")
    print(f"  - Total matches predicted: {total_matches_predicted:,} ({total_matches_predicted / max(total_processed_s1, 1):.2f} matches/S1)")
    print(f"  - Total predicted singletons: {total_singletons:,} ({total_singletons / max(total_processed_s1, 1) * 100:.2f}%)")
    print(f"  - Outputs saved to:")
    print(f"      1. {OUTPUT_MATCHING_PATH}")
    print(f"      2. {OUTPUT_CANDIDATES_PATH}")

    # 4. Automatically run validation
    validator_path = os.path.join(BASE_DIR, "student_resource", "utils", "validate_submission.py")
    test_dir = os.path.join(BASE_DIR, "student_resource", "dataset", "test")
    if os.path.isfile(validator_path):
        print(f"\n[Predict] Executing official submission validator...", flush=True)
        import subprocess
        res = subprocess.run([
            sys.executable, validator_path,
            "--matching", OUTPUT_MATCHING_PATH,
            "--candidate", OUTPUT_CANDIDATES_PATH,
            "--test-dir", test_dir
        ], capture_output=True, text=True)
        print(res.stdout, flush=True)
        if res.stderr:
            print(res.stderr, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Test Inference for Business Entity Resolution")
    parser.add_argument("--sample-limit", type=int, default=None, help="Limit per country for quick validation test")
    parser.add_argument("--chunk-size", type=int, default=50000, help="Chunk size for S1 processing")
    parser.add_argument("--max-candidates", type=int, default=None, help="Limit on S2/S3 candidate pool per country")
    args = parser.parse_args()

    run_inference(
        sample_limit_per_country=args.sample_limit,
        chunk_size_s1=args.chunk_size,
        max_pool_records=args.max_candidates,
    )

