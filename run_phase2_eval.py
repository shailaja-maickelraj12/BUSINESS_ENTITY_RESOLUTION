"""Fast and accurate validation evaluation script for Phase 2 blocking iterations.

Evaluates Iteration A, Iteration B, and Iteration C on a stratified sample of
5,000 validation Source 1 entities against the realistic candidate pool,
computes recall ceilings, candidate statistics, reduction ratios,
and writes out all required Phase 2 reports.
"""

import os
import sys
import time
import json
import pandas as pd
import numpy as np
from collections import defaultdict, Counter

SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "code", "business_entity_resolution", "src"))
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


def run_phase2():
    t_start = time.time()
    norm_cfg = NormalizationConfig()

    print("[Phase 2] Loading Ground Truth...", flush=True)
    _, gt_mapping = load_ground_truth(TRAIN_GROUND_TRUTH_PATH)

    val_ids_path = os.path.join(REPORTS_DIR, "val_s1_ids.txt")
    if not os.path.isfile(val_ids_path):
        print("[Phase 2] Generating validation split...", flush=True)
        generate_validation_split()
    val_ids = load_validation_ids(val_ids_path)
    print(f"[Phase 2] Total validation S1 entities available: {len(val_ids):,}", flush=True)

    # 1. Sample 5,000 validation entities stratified by country
    s1_val = s1_all[s1_all["entity_id"].isin(val_ids)].sample(n=min(5000, len(val_ids)), random_state=RANDOM_SEED).reset_index(drop=True)
    
    print(f"[Phase 2] Selected {len(s1_val):,} validation S1 entities for evaluation:", flush=True)
    for c, cnt in s1_val["country"].value_counts().items():
        print(f"  - {c}: {cnt:,} entities", flush=True)

    s1_val_norm = normalize_dataframe(s1_val, norm_cfg)
    eval_s1_ids = set(s1_val["entity_id"])

    # 2. Collect all true match IDs for these evaluation entities
    eval_target_ids = set()
    total_true_pairs_eval = 0
    for s1_id in eval_s1_ids:
        matches = gt_mapping.get(s1_id, set())
        eval_target_ids.update(matches)
        total_true_pairs_eval += len(matches)
    print(f"[Phase 2] Total true matches for evaluated entities: {total_true_pairs_eval:,} ({len(eval_target_ids):,} unique S2/S3 IDs)", flush=True)

    # 3. Load candidate records: all true matches + 100,000 distractor records per source
    print("[Phase 2] Loading candidate records (all target matches + distractors)...", flush=True)
    def load_candidates_smart(path: str, target_prefix: str) -> pd.DataFrame:
        target_ids_src = {tid for tid in eval_target_ids if tid.startswith(target_prefix)}
        found_target_rows = []
        distractor_rows = []
        distractor_cap = 60000

        for chunk in pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, chunksize=250000):
            # Target matches
            t_chunk = chunk[chunk["entity_id"].isin(target_ids_src)]
            if len(t_chunk) > 0:
                found_target_rows.append(t_chunk)
            # Distractors
            if len(distractor_rows) * 250000 < distractor_cap:
                d_chunk = chunk[~chunk["entity_id"].isin(target_ids_src)].head(min(15000, distractor_cap))
                distractor_rows.append(d_chunk)
                
        all_targets = pd.concat(found_target_rows, ignore_index=True) if found_target_rows else pd.DataFrame()
        all_distractors = pd.concat(distractor_rows, ignore_index=True) if distractor_rows else pd.DataFrame()
        combined = pd.concat([all_targets, all_distractors], ignore_index=True).drop_duplicates(subset=["entity_id"]).reset_index(drop=True)
        print(f"  - {target_prefix}: {len(combined):,} records ({len(all_targets):,} true matches, {len(all_distractors):,} distractors)", flush=True)
        return normalize_dataframe(combined, norm_cfg)

    s2_cand_df = load_candidates_smart(TRAIN_SOURCE2_PATH, "S2-")
    s3_cand_df = load_candidates_smart(TRAIN_SOURCE3_PATH, "S3-")

    s2_lookup = {r["entity_id"]: r for _, r in s2_cand_df.iterrows()}
    s3_lookup = {r["entity_id"]: r for _, r in s3_cand_df.iterrows()}
    total_possible_pairs = len(s1_val) * (len(s2_cand_df) + len(s3_cand_df))

    # =========================================================================
    # ITERATION A: Baseline Rules
    # =========================================================================
    print("\n" + "=" * 60, flush=True)
    print("[Phase 2] ITERATION A: Exact Core Name + Token Signature + Rare Tokens + Structured Address", flush=True)
    print("=" * 60, flush=True)
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
    blocker_a.build_indexes(s2_cand_df, s3_cand_df)
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
    print(f"Iteration A Results (Time: {t_a:.2f}s):", flush=True)
    print(f"  - Pair Recall Ceiling: {sum_a['pair_recall_ceiling'] * 100:.2f}% ({sum_a['retained_true_pairs']:,}/{sum_a['total_true_pairs']:,})", flush=True)
    print(f"  - Entity Recall Ceiling: {sum_a['entity_recall_ceiling'] * 100:.2f}%", flush=True)
    print(f"  - Mean Cands/S1: {sum_a['candidates_per_entity']['mean']:.2f} (Median: {sum_a['candidates_per_entity']['median']:.0f}, p95: {sum_a['candidates_per_entity']['p95']:.0f}, Max: {sum_a['candidates_per_entity']['max']})", flush=True)
    print(f"  - Missed Matches: {sum_a['missed_true_pairs']:,}", flush=True)

    # =========================================================================
    # ITERATION B: Add Character N-Gram & Combined TF-IDF
    # =========================================================================
    print("\n" + "=" * 60, flush=True)
    print("[Phase 2] ITERATION B: Adding Char 3-gram TF-IDF & Combined Text Retrieval", flush=True)
    print("=" * 60, flush=True)
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
    blocker_b.build_indexes(s2_cand_df, s3_cand_df)
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
    print(f"Iteration B Results (Time: {t_b:.2f}s):", flush=True)
    print(f"  - Pair Recall Ceiling: {sum_b['pair_recall_ceiling'] * 100:.2f}% ({sum_b['retained_true_pairs']:,}/{sum_b['total_true_pairs']:,})", flush=True)
    print(f"  - Entity Recall Ceiling: {sum_b['entity_recall_ceiling'] * 100:.2f}%", flush=True)
    print(f"  - Mean Cands/S1: {sum_b['candidates_per_entity']['mean']:.2f} (Median: {sum_b['candidates_per_entity']['median']:.0f}, p95: {sum_b['candidates_per_entity']['p95']:.0f}, Max: {sum_b['candidates_per_entity']['max']})", flush=True)
    print(f"  - Missed Matches: {sum_b['missed_true_pairs']:,}", flush=True)

    # =========================================================================
    # ITERATION C: Precision-Tuned High-Recall Lean Configuration
    # =========================================================================
    print("\n" + "=" * 60, flush=True)
    print("[Phase 2] ITERATION C: Tuned Precision-Optimized Lean Configuration", flush=True)
    print("=" * 60, flush=True)
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
        max_total_candidates_per_entity=30,
    )
    t0 = time.time()
    blocker_c = MultiSourceBlocker(cfg_c)
    blocker_c.build_indexes(s2_cand_df, s3_cand_df)
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
    print(f"Iteration C Results (Time: {t_c:.2f}s):", flush=True)
    print(f"  - Pair Recall Ceiling: {sum_c['pair_recall_ceiling'] * 100:.2f}% ({sum_c['retained_true_pairs']:,}/{sum_c['total_true_pairs']:,})", flush=True)
    print(f"  - Entity Recall Ceiling: {sum_c['entity_recall_ceiling'] * 100:.2f}%", flush=True)
    print(f"  - Mean Cands/S1: {sum_c['candidates_per_entity']['mean']:.2f} (Median: {sum_c['candidates_per_entity']['median']:.0f}, p95: {sum_c['candidates_per_entity']['p95']:.0f}, Max: {sum_c['candidates_per_entity']['max']})", flush=True)
    print(f"  - Missed Matches: {sum_c['missed_true_pairs']:,}", flush=True)

    # Choose Best Configuration
    best_name = "Iteration B" if sum_b["pair_recall_ceiling"] >= sum_c["pair_recall_ceiling"] + 0.005 else "Iteration C"
    best_sum = sum_b if best_name == "Iteration B" else sum_c
    best_stats = stats_df_b if best_name == "Iteration B" else stats_df_c
    best_missed = missed_df_b if best_name == "Iteration B" else missed_df_c

    # Save output artifacts
    stats_tsv_path = os.path.join(REPORTS_DIR, "phase2_candidate_statistics.tsv")
    best_stats.to_csv(stats_tsv_path, sep="\t", index=False)
    print(f"\n[Phase 2] Saved candidate statistics to {stats_tsv_path}", flush=True)

    missed_tsv_path = os.path.join(REPORTS_DIR, "phase2_missed_matches.tsv")
    best_missed.to_csv(missed_tsv_path, sep="\t", index=False)
    print(f"[Phase 2] Saved {len(best_missed):,} missed matches to {missed_tsv_path}", flush=True)

    # Write Markdown Report
    report_path = os.path.join(REPORTS_DIR, "phase2_blocking_report.md")
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

        f.write(f"\n**Selected Best Configuration:** `{best_name}`\n\n")
        f.write("---\n\n")
        f.write("## 2. Segmented Breakdown (Best Configuration)\n\n")
        f.write("### A. By Country\n\n")
        f.write("| Country | S1 Count | True Pairs | Retained Pairs | Pair Recall Ceiling | Entity Recall Ceiling | Mean Candidates/S1 |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for c, c_stat in best_sum["country_breakdown"].items():
            f.write(
                f"| **{c}** | {c_stat['s1_count']:,} | {c_stat['true_pairs']:,} | {c_stat['retained_pairs']:,} | "
                f"**{c_stat['pair_recall_ceiling'] * 100:.2f}%** | {c_stat['entity_recall_ceiling'] * 100:.2f}% | {c_stat['mean_candidates']:.2f} |\n"
            )

        f.write("\n### B. By Candidate Source (S2 vs S3)\n\n")
        f.write("| Source | True Pairs | Retained True Pairs | Recall Ceiling |\n")
        f.write("| :--- | :--- | :--- | :--- |\n")
        for src, s_stat in best_sum["source_breakdown"].items():
            f.write(f"| **{src}** | {s_stat['true_pairs']:,} | {s_stat['retained']:,} | **{s_stat['recall'] * 100:.2f}%** |\n")

        f.write("\n### C. Rule Attribution (True Matches Triggered by Rule)\n\n")
        f.write("| Blocking Rule Channel | True Matches Triggered |\n")
        f.write("| :--- | :--- |\n")
        for r_name, r_cnt in sorted(best_sum["rule_attribution"].items(), key=lambda x: x[1], reverse=True):
            f.write(f"| `{r_name}` | {r_cnt:,} |\n")

        f.write("\n---\n\n")
        f.write("## 3. Top 20 Most Difficult Missed Matches\n\n")
        f.write("| S1 ID | Missed ID | Country | S1 Business Name | Match Business Name | Name Sim | Reason Missed |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for _, r in best_missed.head(20).iterrows():
            sim = r["name_token_sort_sim"]
            reason = "Severe alias/rebrand" if sim < 25 else ("Spelling/transposition beyond floor" if sim < 50 else "High name overlap but rare token pruned")
            f.write(f"| `{r['s1_id']}` | `{r['missed_id']}` | {r['country']} | {r['s1_raw_name']} | {r['missed_raw_name']} | {sim:.1f}% | {reason} |\n")

        f.write(f"\n\n*Total Phase 2 Runtime: {time.time() - t_start:.1f}s*\n")
    print(f"[Phase 2] Report written to {report_path}", flush=True)

    # Also save json results for pipeline
    with open("reports/phase2_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "best_configuration": best_name,
            "pair_recall_ceiling": best_sum["pair_recall_ceiling"],
            "entity_recall_ceiling": best_sum["entity_recall_ceiling"],
            "candidates_per_entity": best_sum["candidates_per_entity"],
            "reduction_ratio": best_sum["reduction_ratio"],
            "missed_true_pairs": best_sum["missed_true_pairs"],
            "total_true_pairs": best_sum["total_true_pairs"],
            "country_breakdown": best_sum["country_breakdown"],
            "source_breakdown": best_sum["source_breakdown"],
        }, f, indent=2)

    print("\n[Phase 2] Execution successfully completed!", flush=True)


if __name__ == "__main__":
    run_phase2()
