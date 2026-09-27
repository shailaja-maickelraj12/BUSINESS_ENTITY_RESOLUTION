"""Data loading and dataset splitting module for Business Entity Resolution.

Handles tab-separated value (TSV) file loading deterministically, parses ground truth
into pair labels, and creates entity-grouped train/validation splits to prevent leakage.
"""

import os
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import (
    RANDOM_SEED,
    TRAIN_SOURCE1_PATH,
    TRAIN_SOURCE2_PATH,
    TRAIN_SOURCE3_PATH,
    TRAIN_GROUND_TRUTH_PATH,
    TEST_SOURCE1_PATH,
    TEST_SOURCE2_PATH,
    TEST_SOURCE3_PATH,
)


def load_tsv(path: str) -> pd.DataFrame:
    """Load a TSV file deterministically as string dtype without dropping empty values.

    Args:
        path: Path to the .tsv file.

    Returns:
        DataFrame with all columns as str and empty cells preserved as empty strings.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"TSV file not found: {path}")

    # Explicit tab separator, string dtype for all columns, empty strings not cast to NaN
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    return df


def load_source_records(
    source_path: str,
    nrows: Optional[int] = None
) -> pd.DataFrame:
    """Load an entity source file (Source 1, 2, or 3).

    Args:
        source_path: Path to source TSV file.
        nrows: Optional limit on number of rows to load (for testing or debugging).

    Returns:
        DataFrame containing entity_id, business_name, business_address, country.
    """
    if nrows is not None:
        df = pd.read_csv(source_path, sep="\t", dtype=str, keep_default_na=False, nrows=nrows)
    else:
        df = load_tsv(source_path)

    expected_cols = {"entity_id", "business_name", "business_address", "country"}
    if not expected_cols.issubset(set(df.columns)):
        raise ValueError(
            f"File {source_path} is missing expected columns. "
            f"Found: {list(df.columns)}, expected at least: {expected_cols}"
        )
    return df


def load_ground_truth(
    gt_path: str = TRAIN_GROUND_TRUTH_PATH
) -> Tuple[pd.DataFrame, Dict[str, Set[str]]]:
    """Load ground truth mappings and return both DataFrame and dictionary.

    Args:
        gt_path: Path to train_ground_truth.tsv.

    Returns:
        Tuple of:
          - DataFrame with source1_entity_id and matched_entity_ids.
          - Dict mapping source1_entity_id -> set of matched S2/S3 entity IDs.
    """
    df = load_tsv(gt_path)
    mapping: Dict[str, Set[str]] = {}

    for _, row in df.iterrows():
        s1_id = row["source1_entity_id"].strip()
        raw_matches = row["matched_entity_ids"].strip()
        if not raw_matches:
            mapping[s1_id] = set()
        else:
            matches = {m.strip() for m in raw_matches.split(",") if m.strip()}
            mapping[s1_id] = matches

    return df, mapping


def create_grouped_validation_split(
    s1_df: pd.DataFrame,
    val_size: float = 0.2,
    seed: int = RANDOM_SEED
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create a leak-free train/validation split strictly grouped by Source 1 entity.

    Stratified by country so that both Train and Validation have proportional
    representation of US and India.

    Args:
        s1_df: DataFrame of Source 1 entities.
        val_size: Fraction of S1 entities to hold out (default 0.20).
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_s1_df, val_s1_df).
    """
    # Stratify by country
    train_s1, val_s1 = train_test_split(
        s1_df,
        test_size=val_size,
        random_state=seed,
        stratify=s1_df["country"] if "country" in s1_df.columns else None
    )
    return train_s1.reset_index(drop=True), val_s1.reset_index(drop=True)
