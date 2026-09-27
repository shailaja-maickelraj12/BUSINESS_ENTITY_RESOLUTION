"""Evaluate normalization on representative samples and Phase 0 hard pairs.

Collects:
  - 10 US examples
  - 10 India examples
  - 10 France test examples
  - Phase 0 hard/noisy examples
Measures throughput/timing, checks for any unintended distortions,
and outputs before/after comparisons.
"""

import sys
import os
import time
import json
import pandas as pd

SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "code", "business_entity_resolution", "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from config import NormalizationConfig, TRAIN_SOURCE1_PATH, TEST_SOURCE1_PATH
from normalize import normalize_dataframe, normalize_business_name, normalize_address


def run_evaluation():
    config = NormalizationConfig()
    
    # 1. Load samples
    print("[Phase 1 Eval] Loading source samples...")
    train_s1 = pd.read_csv(TRAIN_SOURCE1_PATH, sep="\t", dtype=str, keep_default_na=False, nrows=50000)
    test_s1 = pd.read_csv(TEST_SOURCE1_PATH, sep="\t", dtype=str, keep_default_na=False, nrows=50000)
    
    us_samples = train_s1[train_s1["country"] == "US"].head(10).copy()
    india_samples = train_s1[train_s1["country"] == "India"].head(10).copy()
    france_samples = test_s1[test_s1["country"] == "France"].head(10).copy()
    
    # 2. Load Phase 0 hard pairs from phase0_results.json
    with open("phase0_results.json", "r", encoding="utf-8") as f:
        phase0_data = json.load(f)
    hard_pairs = phase0_data.get("hard_pairs", [])
    
    # 3. Benchmark throughput on 10,000 records
    benchmark_df = train_s1.head(10000).copy()
    t0 = time.perf_counter()
    _ = normalize_dataframe(benchmark_df, config)
    t1 = time.perf_counter()
    elapsed = t1 - t0
    records_per_sec = len(benchmark_df) / elapsed
    print(f"[Phase 1 Eval] Normalization benchmark: {len(benchmark_df):,} records in {elapsed:.3f}s ({records_per_sec:,.0f} records/sec)")
    print(f"[Phase 1 Eval] Extrapolated time for 1M records: {(1_000_000 / records_per_sec):.1f}s (~{(1_000_000 / records_per_sec / 60):.2f} mins)")
    
    # 4. Normalize the target sample sets
    norm_us = normalize_dataframe(us_samples, config)
    norm_india = normalize_dataframe(india_samples, config)
    norm_france = normalize_dataframe(france_samples, config)
    
    # Prepare before/after records
    def to_summary_table(df_norm):
        rows = []
        for _, r in df_norm.iterrows():
            rows.append({
                "entity_id": r["entity_id"],
                "country": r["country"],
                "raw_name": r["business_name"],
                "name_core": r["name_core"],
                "name_suffix": r["name_legal_suffix"],
                "raw_address": r["business_address"],
                "address_norm": r["address_normalized"],
                "postal_code": r["address_postal_code"],
                "house_number": r["address_house_number"],
            })
        return rows

    us_summary = to_summary_table(norm_us)
    india_summary = to_summary_table(norm_india)
    france_summary = to_summary_table(norm_france)
    
    # Process Phase 0 hard pairs
    hard_summary = []
    for item in hard_pairs:
        # S1 normalization
        s1_n_norm, s1_core, s1_sfx = normalize_business_name(item["s1_name"], config)
        s1_a_norm, s1_p, s1_h, _, s1_empty = normalize_address(item["s1_addr"], item["country"], config)
        
        # Match normalization
        m_n_norm, m_core, m_sfx = normalize_business_name(item["m_name"], config)
        m_a_norm, m_p, m_h, _, m_empty = normalize_address(item["m_addr"], item["country"], config)
        
        hard_summary.append({
            "s1_id": item["s1_id"],
            "m_id": item["m_id"],
            "country": item["country"],
            "s1_raw_name": item["s1_name"],
            "m_raw_name": item["m_name"],
            "s1_core_name": s1_core,
            "m_core_name": m_core,
            "s1_raw_addr": item["s1_addr"],
            "m_raw_addr": item["m_addr"],
            "s1_addr_norm": s1_a_norm,
            "m_addr_norm": m_a_norm,
            "s1_postal": s1_p,
            "m_postal": m_p,
            "s1_house": s1_h,
            "m_house": m_h,
            "m_addr_empty": m_empty,
        })
        
    out_results = {
        "benchmark": {
            "records": len(benchmark_df),
            "time_sec": elapsed,
            "throughput_rec_per_sec": records_per_sec,
        },
        "us_samples": us_summary,
        "india_samples": india_summary,
        "france_samples": france_summary,
        "hard_pairs": hard_summary,
        "output_columns": list(norm_us.columns),
    }
    
    with open("phase1_eval_results.json", "w", encoding="utf-8") as f:
        json.dump(out_results, f, indent=2)
    print("[Phase 1 Eval] Results saved to phase1_eval_results.json")

if __name__ == "__main__":
    run_evaluation()
