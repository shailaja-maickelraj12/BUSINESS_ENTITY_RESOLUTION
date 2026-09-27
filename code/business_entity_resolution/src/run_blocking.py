"""Pipeline orchestrator for Phase 2: Candidate Generation / Blocking.

Executes Iteration A, Iteration B, and Iteration C on the grouped validation split,
evaluates recall ceilings, candidate statistics, and reduction ratios,
and exports diagnostic reports and missed match logs.
"""

import argparse
import os
import sys
import time
from typing import Dict, List, Optional, Set, Tuple
import pandas as pd

SRC_DIR = os.path.dirname(__file__)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from config import (
    RANDOM_SEED,
    REPORTS_DIR,
    TRAIN_GROUND_TRUTH_PATH,
    TRAIN_SOURCE1_PATH,
    TRAIN_SOURCE2_PATH,
    TRAIN_SOURCE3_PATH,
    NormalizationConfig,
)
from data_loader import load_ground_truth, load_source_records
from normalize import normalize_dataframe
from split import generate_validation_split, load_validation_ids
from blocking import BlockingConfig, MultiSourceBlocker
from blocking_diagnostics import evaluate_blocking


def load_candidate_pool_for_countries(
    countries: Set[str],
    target_match_ids: Optional[Set[str]] = None,
    max_records_per_country: Optional[int] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load and normalize candidate records from S2 and S3 for specified countries.

    Args:
        countries: Set of countries to load (e.g. {'US', 'India'}).
        target_match_ids: Optional set of ground-truth target entity IDs to guarantee inclusion.
        max_records_per_country: Optional cap on distractor records per country.

    Returns:
        Tuple of (s2_df, s3_df) normalized DataFrames.
    """
    norm_cfg = NormalizationConfig()
    print(f"[RunBlocking] Loading S2 & S3 records for countries: {sorted(countries)} (guaranteeing {len(target_match_ids or set()):,} target matches)...", flush=True)

    def read_filtered(path: str, src_label: str) -> pd.DataFrame:
        target_rows = []
        distractor_rows = []
        distractor_cap_total = (max_records_per_country * len(countries)) if max_records_per_country else None

        for chunk in pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, chunksize=500000):
            filtered = chunk[chunk["country"].isin(countries)]
            if len(filtered) == 0:
                continue

            if target_match_ids:
                targets = filtered[filtered["entity_id"].isin(target_match_ids)]
                if len(targets) > 0:
                    target_rows.append(targets)

                if distractor_cap_total is None or sum(len(d) for d in distractor_rows) < distractor_cap_total:
                    distractors = filtered[~filtered["entity_id"].isin(target_match_ids)]
                    if len(distractors) > 0:
                        distractor_rows.append(distractors)
            else:
                distractor_rows.append(filtered)
                if distractor_cap_total and sum(len(d) for d in distractor_rows) >= distractor_cap_total:
                    break

        all_dfs = []
        if target_rows:
            all_dfs.extend(target_rows)
        if distractor_rows:
            dist_df = pd.concat(distractor_rows, ignore_index=True)
            if distractor_cap_total:
                dist_df = dist_df.groupby("country").head(max_records_per_country).reset_index(drop=True)
            all_dfs.append(dist_df)

        df = pd.concat(all_dfs, ignore_index=True).drop_duplicates(subset=["entity_id"]).reset_index(drop=True) if all_dfs else pd.DataFrame()
        print(f"[RunBlocking] Loaded {len(df):,} records for {src_label} ({sum(len(t) for t in target_rows):,} targets retained).", flush=True)
        return normalize_dataframe(df, norm_cfg)

    t0 = time.time()
    s2_df = read_filtered(TRAIN_SOURCE2_PATH, "Source 2")
    s3_df = read_filtered(TRAIN_SOURCE3_PATH, "Source 3")
    print(f"[RunBlocking] Loaded and normalized candidate pool in {time.time() - t0:.2f}s.", flush=True)
    return s2_df, s3_df


def run_blocking_experiment(
    val_sample_size: int = 5000,
    max_cand_records: Optional[int] = None
) -> dict:
    """Execute Iterations A, B, and C and export diagnostics reports."""
    t_start = time.time()
    norm_cfg = NormalizationConfig()

    # 1. Load ground truth
    print("[RunBlocking] Loading ground truth...", flush=True)
    _, gt_mapping = load_ground_truth(TRAIN_GROUND_TRUTH_PATH)

    # 2. Ensure validation split is generated
    val_ids_path = os.path.join(REPORTS_DIR, "val_s1_ids.txt")
    if not os.path.isfile(val_ids_path):
        print("[RunBlocking] Validation split not found. Generating split...", flush=True)
        generate_validation_split()
    val_ids = load_validation_ids(val_ids_path)
    print(f"[RunBlocking] Total validation S1 entities available: {len(val_ids):,}", flush=True)

    # 3. Load Source 1 records and select representative validation evaluation sample
    print(f"[RunBlocking] Loading Source 1 validation sample (n={val_sample_size:,})...", flush=True)
    s1_all = load_source_records(TRAIN_SOURCE1_PATH)
    s1_val_all = s1_all[s1_all["entity_id"].isin(val_ids)].copy()

    # Stratified sample across countries
    if val_sample_size and len(s1_val_all) > val_sample_size:
        sample_indices = []
        for _, group in s1_val_all.groupby("country"):
            k = max(1, int(round(val_sample_size * len(group) / len(s1_val_all))))
            sample_indices.extend(group.sample(n=min(k, len(group)), random_state=RANDOM_SEED).index)
        s1_val = s1_val_all.loc[sample_indices].reset_index(drop=True)
    else:
        s1_val = s1_val_all.reset_index(drop=True)

    print(f"[RunBlocking] Evaluating on {len(s1_val):,} S1 validation entities:", flush=True)
    for c, cnt in s1_val["country"].value_counts().items():
        print(f"  - {c}: {cnt:,} entities", flush=True)

    s1_val_norm = normalize_dataframe(s1_val, norm_cfg)
    active_countries = set(s1_val["country"])

    # Collect target match IDs for evaluated sample
    eval_target_ids = set()
    total_eval_pairs = 0
    for s1_id in s1_val["entity_id"]:
        matches = gt_mapping.get(s1_id, set())
        eval_target_ids.update(matches)
        total_eval_pairs += len(matches)
    print(f"[RunBlocking] Identified {total_eval_pairs:,} true matches ({len(eval_target_ids):,} unique S2/S3 IDs) for evaluated sample.", flush=True)

    # 4. Load candidate pool for active countries
    s2_df, s3_df = load_candidate_pool_for_countries(
        active_countries,
        target_match_ids=eval_target_ids,
        max_records_per_country=max_cand_records
    )

    # Build lookup dictionaries for missed matches diagnostics
    print("[RunBlocking] Building lookup dicts for error diagnostics...", flush=True)
    s2_lookup = s2_df.set_index("entity_id").to_dict(orient="index")
    s3_lookup = s3_df.set_index("entity_id").to_dict(orient="index")

    total_possible_pairs = len(s1_val) * (len(s2_df) + len(s3_df))

    # =========================================================================
    # ITERATION A: Baseline Rules (Exact normalized + Rare tokens + Structured address)
    # =========================================================================
    print("\n" + "=" * 60)
    print("[RunBlocking] EXECUTING ITERATION A: Baseline (Exact + Token + Address)")
    print("=" * 60)
    cfg_a = BlockingConfig(
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
        use_char_ngram=False,
        use_combined_tfidf=False,
        max_total_candidates_per_entity=50,
    )
    t0 = time.time()
    blocker_a = MultiSourceBlocker(cfg_a)
    blocker_a.build_indexes(s2_df, s3_df)
    cands_a, audit_a = blocker_a.block_dataframe(s1_val_norm)
    t_a = time.time() - t0

    sum_a, stats_df_a, missed_df_a = evaluate_blocking(
        candidate_pairs=cands_a,
        audit_trail=audit_a,
        ground_truth=gt_mapping,
        s1_df=s1_val_norm,
        s2_lookup=s2_lookup,
        s3_lookup=s3_lookup,
        total_possible_pairs=total_possible_pairs,
    )
    print(f"Iteration A Finished in {t_a:.2f}s:")
    print(f"  - Pair Recall Ceiling: {sum_a['pair_recall_ceiling'] * 100:.2f}% ({sum_a['retained_true_pairs']:,}/{sum_a['total_true_pairs']:,})")
    print(f"  - Entity Recall Ceiling: {sum_a['entity_recall_ceiling'] * 100:.2f}%")
    print(f"  - Mean Candidates/S1: {sum_a['candidates_per_entity']['mean']:.2f} (p95: {sum_a['candidates_per_entity']['p95']:.0f})")
    print(f"  - Missed Matches: {sum_a['missed_true_pairs']:,}")

    # =========================================================================
    # ITERATION B: Add Character N-Gram and Combined Retrieval
    # =========================================================================
    print("\n" + "=" * 60)
    print("[RunBlocking] EXECUTING ITERATION B: Adding Char N-Gram & Combined TF-IDF")
    print("=" * 60)
    cfg_b = BlockingConfig(
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
    t0 = time.time()
    blocker_b = MultiSourceBlocker(cfg_b)
    blocker_b.build_indexes(s2_df, s3_df)
    cands_b, audit_b = blocker_b.block_dataframe(s1_val_norm)
    t_b = time.time() - t0

    sum_b, stats_df_b, missed_df_b = evaluate_blocking(
        candidate_pairs=cands_b,
        audit_trail=audit_b,
        ground_truth=gt_mapping,
        s1_df=s1_val_norm,
        s2_lookup=s2_lookup,
        s3_lookup=s3_lookup,
        total_possible_pairs=total_possible_pairs,
    )
    print(f"Iteration B Finished in {t_b:.2f}s:")
    print(f"  - Pair Recall Ceiling: {sum_b['pair_recall_ceiling'] * 100:.2f}% ({sum_b['retained_true_pairs']:,}/{sum_b['total_true_pairs']:,})")
    print(f"  - Entity Recall Ceiling: {sum_b['entity_recall_ceiling'] * 100:.2f}%")
    print(f"  - Mean Candidates/S1: {sum_b['candidates_per_entity']['mean']:.2f} (p95: {sum_b['candidates_per_entity']['p95']:.0f})")
    print(f"  - Missed Matches: {sum_b['missed_true_pairs']:,}")

    # =========================================================================
    # ITERATION C: Tuned Precision-Optimized Configuration
    # =========================================================================
    print("\n" + "=" * 60)
    print("[RunBlocking] EXECUTING ITERATION C: Tuned High-Recall Lean Configuration")
    print("=" * 60)
    cfg_c = BlockingConfig(
        use_exact_core_name=True,
        use_exact_full_name=True,
        use_token_signature=True,
        use_rare_tokens=True,
        rare_token_min_len=3,
        rare_token_max_df=400,
        max_cands_per_rare_token=8,
        use_postal_code=True,
        postal_code_max_df=300,
        max_cands_per_postal=10,
        use_house_number=True,
        house_number_max_df=300,
        max_cands_per_house=10,
        use_char_ngram=True,
        char_ngram_top_k=4,
        char_ngram_min_sim=0.55,
        use_combined_tfidf=True,
        combined_top_k=2,
        combined_min_sim=0.45,
        max_total_candidates_per_entity=30,  # Lean cap
    )
    t0 = time.time()
    blocker_c = MultiSourceBlocker(cfg_c)
    blocker_c.build_indexes(s2_df, s3_df)
    cands_c, audit_c = blocker_c.block_dataframe(s1_val_norm)
    t_c = time.time() - t0

    sum_c, stats_df_c, missed_df_c = evaluate_blocking(
        candidate_pairs=cands_c,
        audit_trail=audit_c,
        ground_truth=gt_mapping,
        s1_df=s1_val_norm,
        s2_lookup=s2_lookup,
        s3_lookup=s3_lookup,
        total_possible_pairs=total_possible_pairs,
    )
    print(f"Iteration C Finished in {t_c:.2f}s:")
    print(f"  - Pair Recall Ceiling: {sum_c['pair_recall_ceiling'] * 100:.2f}% ({sum_c['retained_true_pairs']:,}/{sum_c['total_true_pairs']:,})")
    print(f"  - Entity Recall Ceiling: {sum_c['entity_recall_ceiling'] * 100:.2f}%")
    print(f"  - Mean Candidates/S1: {sum_c['candidates_per_entity']['mean']:.2f} (p95: {sum_c['candidates_per_entity']['p95']:.0f})")
    print(f"  - Missed Matches: {sum_c['missed_true_pairs']:,}")

    # =========================================================================
    # EXPORT REPORTS & ARTIFACTS
    # =========================================================================
    print("\n[RunBlocking] Exporting diagnostics and missed match files...")
    # 1. Export candidate statistics TSV for best model (Iteration C or B)
    best_summary = sum_b if sum_b["pair_recall_ceiling"] > sum_c["pair_recall_ceiling"] and (sum_b["pair_recall_ceiling"] - sum_c["pair_recall_ceiling"] > 0.01) else sum_c
    best_stats_df = stats_df_b if best_summary == sum_b else stats_df_c
    best_missed_df = missed_df_b if best_summary == sum_b else missed_df_c
    best_cands = cands_b if best_summary == sum_b else cands_c

    stats_tsv_path = os.path.join(REPORTS_DIR, "phase2_candidate_statistics.tsv")
    best_stats_df.to_csv(stats_tsv_path, sep="\t", index=False)
    print(f"[RunBlocking] Saved candidate statistics to {stats_tsv_path}")

    # 2. Export missed matches TSV
    missed_tsv_path = os.path.join(REPORTS_DIR, "phase2_missed_matches.tsv")
    best_missed_df.to_csv(missed_tsv_path, sep="\t", index=False)
    print(f"[RunBlocking] Saved {len(best_missed_df):,} missed matches to {missed_tsv_path}")

    # 3. Generate phase2_blocking_report.md
    report_path = os.path.join(REPORTS_DIR, "phase2_blocking_report.md")
    write_blocking_report(
        report_path=report_path,
        sum_a=sum_a, sum_b=sum_b, sum_c=sum_c,
        t_a=t_a, t_b=t_b, t_c=t_c,
        best_missed_df=best_missed_df,
        total_runtime=time.time() - t_start
    )
    print(f"[RunBlocking] Saved complete Phase 2 report to {report_path}")

    return {
        "iteration_a": sum_a,
        "iteration_b": sum_b,
        "iteration_c": sum_c,
        "best_iteration": "Iteration C" if best_summary == sum_c else "Iteration B",
        "total_runtime": time.time() - t_start
    }


def write_blocking_report(
    report_path: str,
    sum_a: dict, sum_b: dict, sum_c: dict,
    t_a: float, t_b: float, t_c: float,
    best_missed_df: pd.DataFrame,
    total_runtime: float
) -> None:
    """Generate comprehensive markdown report comparing all blocking iterations."""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Phase 2: Candidate Generation & Blocking Report\n\n")
        f.write("## 1. Executive Summary & Iteration Comparison\n\n")
        f.write("| Iteration | Pair Recall Ceiling | Entity Recall Ceiling | Mean Cands/S1 | Median | p95 | Max | Missed Matches | Reduction Ratio | Runtime |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
        
        for label, s, t in [
            ("Iteration A (Baseline)", sum_a, t_a),
            ("Iteration B (+Char & Combined TF-IDF)", sum_b, t_b),
            ("Iteration C (Tuned Lean)", sum_c, t_c),
        ]:
            p_rec = s["pair_recall_ceiling"] * 100
            e_rec = s["entity_recall_ceiling"] * 100
            m_c = s["candidates_per_entity"]["mean"]
            med = s["candidates_per_entity"]["median"]
            p95 = s["candidates_per_entity"]["p95"]
            max_c = s["candidates_per_entity"]["max"]
            missed = s["missed_true_pairs"]
            rr = s["reduction_ratio"] * 100
            f.write(f"| **{label}** | **{p_rec:.2f}%** | {e_rec:.2f}% | {m_c:.2f} | {med:.0f} | {p95:.0f} | {max_c} | {missed:,} | {rr:.4f}% | {t:.1f}s |\n")

        best = sum_c if sum_c["pair_recall_ceiling"] >= 0.97 else sum_b
        f.write("\n---\n\n")
        f.write("## 2. Segmented Breakdown (Best Configuration)\n\n")
        f.write("### A. By Country\n\n")
        f.write("| Country | S1 Count | True Pairs | Retained Pairs | Pair Recall Ceiling | Entity Recall Ceiling | Mean Candidates/S1 |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for c, c_stat in best["country_breakdown"].items():
            f.write(
                f"| **{c}** | {c_stat['s1_count']:,} | {c_stat['true_pairs']:,} | {c_stat['retained_pairs']:,} | "
                f"**{c_stat['pair_recall_ceiling'] * 100:.2f}%** | {c_stat['entity_recall_ceiling'] * 100:.2f}% | {c_stat['mean_candidates']:.2f} |\n"
            )

        f.write("\n### B. By Candidate Source (S2 vs S3)\n\n")
        f.write("| Source | True Pairs | Retained True Pairs | Recall Ceiling |\n")
        f.write("| :--- | :--- | :--- | :--- |\n")
        for src, s_stat in best["source_breakdown"].items():
            f.write(f"| **{src}** | {s_stat['true_pairs']:,} | {s_stat['retained']:,} | **{s_stat['recall'] * 100:.2f}%** |\n")

        f.write("\n### C. Rule Attribution (True Matches Triggered by Rule)\n\n")
        f.write("| Blocking Rule Channel | True Matches Triggered |\n")
        f.write("| :--- | :--- |\n")
        for r_name, r_cnt in sorted(best["rule_attribution"].items(), key=lambda x: x[1], reverse=True):
            f.write(f"| `{r_name}` | {r_cnt:,} |\n")

        f.write("\n---\n\n")
        f.write("## 3. Analysis of Top 20 Missed Matches\n\n")
        f.write("Below are the 20 most difficult true matches missed during candidate generation:\n\n")
        f.write("| S1 ID | Missed ID | Country | S1 Business Name | Match Business Name | Name Sim | Reason Missed |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")

        for _, r in best_missed_df.head(20).iterrows():
            s1_id = r["s1_id"]
            m_id = r["missed_id"]
            c = r["country"]
            s1_n = r["s1_raw_name"]
            m_n = r["missed_raw_name"]
            sim = r["name_token_sort_sim"]
            # Classify reason
            reason = "Severe alias/rebrand" if sim < 25 else ("Spelling/transposition beyond floor" if sim < 50 else "High name overlap but rare token pruned")
            f.write(f"| `{s1_id}` | `{m_id}` | {c} | {s1_n} | {m_n} | {sim:.1f}% | {reason} |\n")

        f.write(f"\n\n*Total diagnostic runtime: {total_runtime:.1f}s*\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Phase 2 Candidate Generation / Blocking")
    parser.add_argument("--sample-size", type=int, default=5000, help="Validation sample size")
    parser.add_argument("--max-candidates", type=int, default=100000, help="Max candidate records per country for testing")
    args = parser.parse_args()

    run_blocking_experiment(
        val_sample_size=args.sample_size,
        max_cand_records=args.max_candidates if args.max_candidates > 0 else None
    )
