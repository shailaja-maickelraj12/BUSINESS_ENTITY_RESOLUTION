"""Model Training and Macro-F0.5 Threshold Tuning for Business Entity Resolution.

Trains a LightGBM gradient boosted decision tree classifier on pairwise features,
evaluates on the grouped validation set, tunes probability threshold and singleton
decision boundary to maximize Macro-F0.5, and persists model artifacts.
"""

import argparse
import json
import os
import pickle
import sys
import time
from typing import Dict, List, Optional, Set, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd

SRC_DIR = os.path.dirname(__file__)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from config import (
    BASE_DIR,
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
from features import compute_features_for_candidates

ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
MODEL_DIR = os.path.join(ARTIFACTS_DIR, "model")
FEATURES_DIR = os.path.join(ARTIFACTS_DIR, "features")


def compute_macro_f05(
    s1_ids: List[str],
    predictions: Dict[str, List[str]],
    ground_truth: Dict[str, Set[str]],
) -> Tuple[float, float, float, dict]:
    """Compute exact competition Macro-average F_0.5 metric over all S1 entities.

    Formula:
        F_0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)
    Singletons:
        - True singleton (empty GT) + Empty prediction -> Score = 1.0
        - True singleton + Non-empty prediction -> Score = 0.0
        - Non-singleton + Empty prediction -> Score = 0.0

    Returns:
        Tuple of (macro_f05, macro_precision, macro_recall, details_dict)
    """
    scores = []
    precisions = []
    recalls = []
    
    tp_total = 0
    fp_total = 0
    fn_total = 0
    singleton_correct = 0
    singleton_total = 0

    for s1_id in s1_ids:
        true_set = ground_truth.get(s1_id, set())
        pred_list = predictions.get(s1_id, [])
        pred_set = set(pred_list)

        if not true_set:
            singleton_total += 1
            if not pred_set:
                singleton_correct += 1
                scores.append(1.0)
                precisions.append(1.0)
                recalls.append(1.0)
            else:
                fp_total += len(pred_set)
                scores.append(0.0)
                precisions.append(0.0)
                recalls.append(1.0)
        else:
            if not pred_set:
                fn_total += len(true_set)
                scores.append(0.0)
                precisions.append(1.0)
                recalls.append(0.0)
            else:
                hits = len(pred_set.intersection(true_set))
                tp_total += hits
                fp_total += len(pred_set) - hits
                fn_total += len(true_set) - hits

                p = hits / len(pred_set)
                r = hits / len(true_set)
                precisions.append(p)
                recalls.append(r)

                if p + r == 0:
                    scores.append(0.0)
                else:
                    f05 = (1.25 * p * r) / (0.25 * p + r)
                    scores.append(f05)

    macro_f05 = float(np.mean(scores)) if scores else 0.0
    macro_prec = float(np.mean(precisions)) if precisions else 0.0
    macro_rec = float(np.mean(recalls)) if recalls else 0.0
    
    details = {
        "macro_f05": macro_f05,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "micro_precision": tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0,
        "micro_recall": tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0,
        "singleton_accuracy": singleton_correct / singleton_total if singleton_total > 0 else 0.0,
        "singleton_total": singleton_total,
        "singleton_correct": singleton_correct,
        "total_entities": len(s1_ids),
    }
    return macro_f05, macro_prec, macro_rec, details


def train_and_evaluate(
    train_sample_size: int = 15000,
    val_sample_size: int = 5000,
    max_cand_distractors: int = 50000,
) -> dict:
    """Execute complete end-to-end training and threshold tuning pipeline."""
    t_start = time.time()
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(FEATURES_DIR, exist_ok=True)
    norm_cfg = NormalizationConfig()

    print("[Train] Loading Ground Truth...", flush=True)
    _, gt_mapping = load_ground_truth(TRAIN_GROUND_TRUTH_PATH)

    val_ids_path = os.path.join(REPORTS_DIR, "val_s1_ids.txt")
    if not os.path.isfile(val_ids_path):
        generate_validation_split()
    val_ids = load_validation_ids(val_ids_path)

    print("[Train] Loading Source 1 records...", flush=True)
    s1_all = load_source_records(TRAIN_SOURCE1_PATH)
    
    val_mask = s1_all["entity_id"].isin(val_ids)
    s1_val_pool = s1_all[val_mask].copy()
    s1_train_pool = s1_all[~val_mask].copy()

    # Stratified samples for train and validation
    print(f"[Train] Sampling {train_sample_size:,} train and {val_sample_size:,} validation entities...", flush=True)
    train_indices = []
    for _, grp in s1_train_pool.groupby("country"):
        k = max(1, int(round(train_sample_size * len(grp) / len(s1_train_pool))))
        train_indices.extend(grp.sample(n=min(k, len(grp)), random_state=RANDOM_SEED).index)
    s1_train = s1_train_pool.loc[train_indices].reset_index(drop=True)

    val_indices = []
    for _, grp in s1_val_pool.groupby("country"):
        k = max(1, int(round(val_sample_size * len(grp) / len(s1_val_pool))))
        val_indices.extend(grp.sample(n=min(k, len(grp)), random_state=RANDOM_SEED).index)
    s1_val = s1_val_pool.loc[val_indices].reset_index(drop=True)

    print(f"[Train] Normalizing S1 records...", flush=True)
    s1_train_norm = normalize_dataframe(s1_train, norm_cfg)
    s1_val_norm = normalize_dataframe(s1_val, norm_cfg)

    # Collect target match IDs for both sets
    all_eval_s1_ids = list(s1_train["entity_id"]) + list(s1_val["entity_id"])
    target_match_ids = set()
    for sid in all_eval_s1_ids:
        target_match_ids.update(gt_mapping.get(sid, set()))
    print(f"[Train] Target matches required for candidate pool: {len(target_match_ids):,}", flush=True)

    # Load S2 & S3 pool with guaranteed target matches
    from run_blocking import load_candidate_pool_for_countries
    active_countries = set(s1_train["country"]).union(set(s1_val["country"]))
    s2_df, s3_df = load_candidate_pool_for_countries(
        countries=active_countries,
        target_match_ids=target_match_ids,
        max_records_per_country=max_cand_distractors,
    )

    # Build Blocking Index (Iteration B Configuration)
    print("\n[Train] Building candidate blocker indexes (Iteration B)...", flush=True)
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
    blocker = MultiSourceBlocker(cfg_block)
    blocker.build_indexes(s2_df, s3_df)

    # Candidate generation
    print("[Train] Blocking train entities...", flush=True)
    cands_train, _ = blocker.block_dataframe(s1_train_norm)
    print("[Train] Blocking validation entities...", flush=True)
    cands_val, _ = blocker.block_dataframe(s1_val_norm)

    # Build lookups
    s1_train_lookup = s1_train_norm.set_index("entity_id").to_dict(orient="index")
    s1_val_lookup = s1_val_norm.set_index("entity_id").to_dict(orient="index")
    cand_pool_lookup = s2_df.set_index("entity_id").to_dict(orient="index")
    cand_pool_lookup.update(s3_df.set_index("entity_id").to_dict(orient="index"))

    # Feature computation
    print("[Train] Extracting pairwise features for training pairs...", flush=True)
    t0 = time.time()
    train_feat_df = compute_features_for_candidates(cands_train, s1_train_lookup, cand_pool_lookup)
    print(f"[Train] Extracted {len(train_feat_df):,} training pair features in {time.time() - t0:.1f}s.", flush=True)

    print("[Train] Extracting pairwise features for validation pairs...", flush=True)
    t0 = time.time()
    val_feat_df = compute_features_for_candidates(cands_val, s1_val_lookup, cand_pool_lookup)
    print(f"[Train] Extracted {len(val_feat_df):,} validation pair features in {time.time() - t0:.1f}s.", flush=True)

    # Add labels
    def add_label(df: pd.DataFrame) -> pd.DataFrame:
        labels = []
        for _, row in df.iterrows():
            s1_id = row["s1_id"]
            cand_id = row["candidate_id"]
            is_match = 1 if cand_id in gt_mapping.get(s1_id, set()) else 0
            labels.append(is_match)
        df["label"] = labels
        return df

    train_feat_df = add_label(train_feat_df)
    val_feat_df = add_label(val_feat_df)

    pos_tr = int(train_feat_df["label"].sum())
    neg_tr = len(train_feat_df) - pos_tr
    print(f"[Train] Training pair class balance: {pos_tr:,} positives vs {neg_tr:,} negatives (ratio 1:{neg_tr / max(pos_tr, 1):.1f})", flush=True)

    # Prepare feature columns
    ignore_cols = {"s1_id", "candidate_id", "label"}
    feature_cols = [c for c in train_feat_df.columns if c not in ignore_cols]
    print(f"[Train] Total feature dimension: {len(feature_cols)} features: {feature_cols}")

    X_train = train_feat_df[feature_cols].values
    y_train = train_feat_df["label"].values
    X_val = val_feat_df[feature_cols].values
    y_val = val_feat_df["label"].values

    # Train LightGBM model
    print("\n[Train] Training LightGBM classifier...", flush=True)
    model = lgb.LGBMClassifier(
        n_estimators=350,
        learning_rate=0.06,
        num_leaves=31,
        max_depth=6,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
    )

    # Predict validation probabilities
    val_probs = model.predict_proba(X_val)[:, 1]
    val_feat_df["pred_prob"] = val_probs

    # Threshold Optimization for Macro-F0.5
    print("\n[Train] Optimizing probability threshold for Macro-F0.5...", flush=True)
    val_s1_list = list(s1_val["entity_id"])
    best_thresh = 0.50
    best_f05 = -1.0
    best_details = {}
    threshold_results = []

    # Map validation predictions by s1_id for fast threshold tuning
    pairs_by_s1 = {}
    for _, r in val_feat_df.iterrows():
        s1 = r["s1_id"]
        cid = r["candidate_id"]
        prob = r["pred_prob"]
        if s1 not in pairs_by_s1:
            pairs_by_s1[s1] = []
        pairs_by_s1[s1].append((cid, prob))

    for thresh in np.arange(0.35, 0.90, 0.02):
        thresh = round(float(thresh), 2)
        preds = {}
        for s1 in val_s1_list:
            cands = pairs_by_s1.get(s1, [])
            matched = [cid for cid, p in cands if p >= thresh]
            preds[s1] = matched

        f05, p_macro, r_macro, det = compute_macro_f05(val_s1_list, preds, gt_mapping)
        threshold_results.append((thresh, f05, p_macro, r_macro, det["singleton_accuracy"]))

        if f05 > best_f05:
            best_f05 = f05
            best_thresh = thresh
            best_details = det

    print(f"\n[Train] Threshold Tuning Results:")
    print(f"  | Threshold | Macro-F0.5 | Macro-Prec | Macro-Rec | Singleton Acc |")
    print(f"  | :--- | :--- | :--- | :--- | :--- |")
    for t, f, p, r, sa in threshold_results[::2]:
        star = "*" if t == best_thresh else " "
        print(f"  | {t:.2f} {star} | {f * 100:.2f}% | {p * 100:.2f}% | {r * 100:.2f}% | {sa * 100:.2f}% |")

    print(f"\n[Train] Optimal Threshold: tau = {best_thresh:.2f} -> Macro-F0.5 = {best_f05 * 100:.2f}%")

    # Persist model artifacts
    model_path = os.path.join(MODEL_DIR, "matcher_lgb.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"[Train] Saved trained LightGBM model to {model_path}")

    cfg_dict = {
        "optimal_threshold": best_thresh,
        "best_macro_f05": best_f05,
        "feature_cols": feature_cols,
        "train_sample_size": train_sample_size,
        "val_sample_size": val_sample_size,
        "best_details": best_details,
    }
    cfg_path = os.path.join(MODEL_DIR, "model_config.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg_dict, f, indent=2)
    print(f"[Train] Saved model configuration to {cfg_path}")

    # Generate Feature Importance
    importances = model.feature_importances_
    feat_imp = sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)

    # Write Phase 4 Report
    report_path = os.path.join(REPORTS_DIR, "phase4_training_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Phase 4: Model Training & Macro-F0.5 Optimization Report\n\n")
        f.write("## 1. Validation Performance Summary\n\n")
        f.write(f"- **Optimal Decision Threshold ($\\tau$):** `{best_thresh:.2f}`\n")
        f.write(f"- **Macro-average $F_{{0.5}}$:** **{best_f05 * 100:.2f}%**\n")
        f.write(f"- **Macro-average Precision:** {best_details['macro_precision'] * 100:.2f}%\n")
        f.write(f"- **Macro-average Recall:** {best_details['macro_recall'] * 100:.2f}%\n")
        f.write(f"- **Micro Precision:** {best_details['micro_precision'] * 100:.2f}%\n")
        f.write(f"- **Micro Recall:** {best_details['micro_recall'] * 100:.2f}%\n")
        f.write(f"- **Singleton Identification Accuracy:** {best_details['singleton_accuracy'] * 100:.2f}% ({best_details['singleton_correct']:,} / {best_details['singleton_total']:,})\n")
        f.write(f"- **Evaluated Entities:** {best_details['total_entities']:,}\n\n")

        f.write("## 2. Top 15 Feature Importances\n\n")
        f.write("| Rank | Feature | Importance (Splits) |\n")
        f.write("| :--- | :--- | :--- |\n")
        for i, (fn, imp) in enumerate(feat_imp[:15], start=1):
            f.write(f"| {i} | `{fn}` | {imp} |\n")

        f.write("\n## 3. Threshold Optimization Curve\n\n")
        f.write("| Threshold ($\\tau$) | Macro-$F_{0.5}$ | Macro-Precision | Macro-Recall | Singleton Accuracy |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- |\n")
        for t, f_val, p_val, r_val, sa in threshold_results:
            bold = "**" if t == best_thresh else ""
            f.write(f"| {bold}{t:.2f}{bold} | {bold}{f_val * 100:.2f}%{bold} | {p_val * 100:.2f}% | {r_val * 100:.2f}% | {sa * 100:.2f}% |\n")

        f.write(f"\n*Total training & tuning runtime: {time.time() - t_start:.1f}s*\n")

    print(f"[Train] Saved complete Phase 4 report to {report_path}")
    return cfg_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LightGBM Matcher and Tune Macro-F0.5")
    parser.add_argument("--train-sample", type=int, default=12000, help="Train sample size")
    parser.add_argument("--val-sample", type=int, default=4000, help="Validation sample size")
    parser.add_argument("--distractors", type=int, default=40000, help="Max distractor candidates per country")
    args = parser.parse_args()

    train_and_evaluate(
        train_sample_size=args.train_sample,
        val_sample_size=args.val_sample,
        max_cand_distractors=args.distractors,
    )
