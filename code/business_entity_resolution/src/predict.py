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

    # 2. Prepare Output Files with exact headers
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
        rare_token_max_df=500,
        max_cands_per_rare_token=10,
        use_postal_code=True,
        postal_code_max_df=500,
        max_cands_per_postal=15,
        use_house_number=True,
        house_number_max_df=500,
        max_cands_per_house=15,
        use_char_ngram=True,
        char_ngram_top_k=5,
        char_ngram_min_sim=0.50,
        use_combined_tfidf=True,
        combined_top_k=3,
        combined_min_sim=0.40,
        max_total_candidates_per_entity=50,
    )

    for country in countries:
        print(f"\n{'=' * 60}", flush=True)
        print(f"[Predict] PROCESSING COUNTRY: {country.upper()}", flush=True)
        print(f"{'=' * 60}", flush=True)

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

        # Read and process Test S1 in chunks
        country_s1_processed = 0
        s1_reader = pd.read_csv(TEST_SOURCE1_PATH, sep="\t", dtype=str, keep_default_na=False, chunksize=chunk_size_s1)

        for chunk_idx, s1_chunk in enumerate(s1_reader):
            s1_country = s1_chunk[s1_chunk["country"] == country]
            if len(s1_country) == 0:
                continue

            if sample_limit_per_country and (country_s1_processed + len(s1_country) > sample_limit_per_country):
                rem = sample_limit_per_country - country_s1_processed
                if rem <= 0:
                    break
                s1_country = s1_country.head(rem).copy()

            t_chunk_start = time.time()
            s1_country_norm = normalize_dataframe(s1_country, norm_cfg)
            s1_lookup = s1_country_norm.set_index("entity_id").to_dict(orient="index")
            s1_ids = list(s1_country["entity_id"])

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
                    feat_df["pred_prob"] = probs

                    # Filter predictions by optimal threshold
                    hits_df = feat_df[feat_df["pred_prob"] >= threshold]
                    for s1_id, group in hits_df.groupby("s1_id"):
                        # Keep matches in order of probability descending
                        sorted_matches = group.sort_values(by="pred_prob", ascending=False)["candidate_id"].tolist()
                        # Strictly guarantee no duplicates and subset invariant
                        matched_dict[s1_id] = list(dict.fromkeys(sorted_matches))

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

            country_s1_processed += len(s1_country)
            total_processed_s1 += len(s1_country)
            print(f"[Predict]   Chunk {chunk_idx + 1}: Processed {len(s1_country):,} entities in {time.time() - t_chunk_start:.1f}s (Country Total: {country_s1_processed:,})", flush=True)

            if sample_limit_per_country and country_s1_processed >= sample_limit_per_country:
                break

        # Free country memory
        del s2_pool, s3_pool, cand_lookup, blocker
        gc.collect()

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
