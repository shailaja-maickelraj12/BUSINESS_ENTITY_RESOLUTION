"""Deterministic entity-grouped dataset splitting module.

Splits Source 1 entities into train (80%) and validation (20%) grouped by entity_id,
ensuring all linked records stay strictly together without data leakage.
"""

import os
import json
from typing import Dict, Set, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split

import sys
SRC_DIR = os.path.dirname(__file__)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from config import RANDOM_SEED, TRAIN_SOURCE1_PATH, REPORTS_DIR
from data_loader import load_source_records


def generate_validation_split(
    s1_path: str = TRAIN_SOURCE1_PATH,
    val_size: float = 0.20,
    seed: int = RANDOM_SEED,
    output_metadata_path: str = os.path.join(REPORTS_DIR, "split_metadata.json"),
    output_val_ids_path: str = os.path.join(REPORTS_DIR, "val_s1_ids.txt")
) -> Tuple[Set[str], Set[str]]:
    """Create and persist a reproducible grouped validation split of Source 1 entities.

    Stratified by country so train and validation sets have identical country distributions.

    Args:
        s1_path: Path to train_source1.tsv.
        val_size: Validation fraction (default 0.20).
        seed: Random seed.
        output_metadata_path: Where to save JSON metadata.
        output_val_ids_path: Where to save validation S1 IDs list.

    Returns:
        Tuple of (train_s1_ids_set, val_s1_ids_set).
    """
    print(f"[Split] Loading Source 1 records from {s1_path}...")
    s1_df = load_source_records(s1_path)
    total_entities = len(s1_df)

    # Deterministic split stratified by country
    train_df, val_df = train_test_split(
        s1_df,
        test_size=val_size,
        random_state=seed,
        stratify=s1_df["country"] if "country" in s1_df.columns else None
    )

    train_ids = set(train_df["entity_id"])
    val_ids = set(val_df["entity_id"])

    # Collect distribution statistics
    country_train = train_df["country"].value_counts().to_dict()
    country_val = val_df["country"].value_counts().to_dict()

    metadata = {
        "random_seed": seed,
        "val_size": val_size,
        "total_s1_entities": total_entities,
        "train_s1_count": len(train_ids),
        "val_s1_count": len(val_ids),
        "country_distribution_train": country_train,
        "country_distribution_val": country_val,
    }

    os.makedirs(os.path.dirname(output_metadata_path), exist_ok=True)
    with open(output_metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # Save validation IDs to text file (one ID per line)
    with open(output_val_ids_path, "w", encoding="utf-8") as f:
        for s1_id in sorted(val_ids):
            f.write(f"{s1_id}\n")

    print(f"[Split] Split complete: {len(train_ids):,} train, {len(val_ids):,} validation.")
    print(f"[Split] Metadata saved to {output_metadata_path}")
    print(f"[Split] Validation IDs saved to {output_val_ids_path}")

    return train_ids, val_ids


def load_validation_ids(
    val_ids_path: str = os.path.join(REPORTS_DIR, "val_s1_ids.txt")
) -> Set[str]:
    """Load pre-computed validation S1 entity IDs.

    Args:
        val_ids_path: Path to val_s1_ids.txt.

    Returns:
        Set of validation S1 IDs.
    """
    if not os.path.isfile(val_ids_path):
        raise FileNotFoundError(f"Validation IDs file not found: {val_ids_path}. Run generate_validation_split first.")

    with open(val_ids_path, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


if __name__ == "__main__":
    generate_validation_split()
