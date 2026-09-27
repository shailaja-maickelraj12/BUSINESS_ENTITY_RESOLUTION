"""Pairwise Feature Engineering Module for Business Entity Resolution.

Extracts discriminative string similarity, token overlap, numeric/postal matching,
and group-level ranking features for candidate pairs (Source 1, Candidate).
"""

from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd
from rapidfuzz import fuzz


def char_ngrams(s: str, n: int = 3) -> Set[str]:
    """Generate set of character n-grams from a string."""
    if len(s) < n:
        return {s} if s else set()
    return {s[i:i + n] for i in range(len(s) - n + 1)}


def jaccard_similarity(s1: Set[str], s2: Set[str]) -> float:
    """Compute Jaccard similarity between two sets."""
    if not s1 and not s2:
        return 0.0
    intersection = len(s1.intersection(s2))
    union = len(s1.union(s2))
    return intersection / union if union > 0 else 0.0


def compute_pair_features(
    s1_row: dict,
    cand_row: dict,
) -> dict:
    """Compute single pair features between an S1 entity and a candidate entity.

    Args:
        s1_row: Dict of S1 normalized entity attributes.
        cand_row: Dict of candidate (S2 or S3) normalized entity attributes.

    Returns:
        Dict of numerical feature values.
    """
    s1_name = s1_row.get("name_normalized", "")
    cand_name = cand_row.get("name_normalized", "")
    s1_core = s1_row.get("name_core", "")
    cand_core = cand_row.get("name_core", "")

    s1_addr = s1_row.get("address_normalized", "")
    cand_addr = cand_row.get("address_normalized", "")

    # Token sets
    s1_name_tokens = set(s1_name.split())
    cand_name_tokens = set(cand_name.split())
    s1_addr_tokens = set(s1_addr.split())
    cand_addr_tokens = set(cand_addr.split())

    # --- 1. Name Features ---
    name_lev = fuzz.ratio(s1_name, cand_name) / 100.0
    name_partial = fuzz.partial_ratio(s1_name, cand_name) / 100.0
    name_token_sort = fuzz.token_sort_ratio(s1_name, cand_name) / 100.0
    name_token_set = fuzz.token_set_ratio(s1_name, cand_name) / 100.0
    name_core_sort = fuzz.token_sort_ratio(s1_core, cand_core) / 100.0

    name_jaccard = jaccard_similarity(s1_name_tokens, cand_name_tokens)
    name_3gram_jaccard = jaccard_similarity(char_ngrams(s1_name, 3), char_ngrams(cand_name, 3))
    name_4gram_jaccard = jaccard_similarity(char_ngrams(s1_name, 4), char_ngrams(cand_name, 4))

    # Prefix and first token
    prefix_3_match = 1.0 if s1_name[:3] == cand_name[:3] and len(s1_name) >= 3 else 0.0
    prefix_5_match = 1.0 if s1_name[:5] == cand_name[:5] and len(s1_name) >= 5 else 0.0
    
    first_token_match = 0.0
    if s1_name_tokens and cand_name_tokens:
        s1_first = s1_name.split()[0]
        cand_first = cand_name.split()[0]
        first_token_match = 1.0 if s1_first == cand_first else 0.0

    # Legal suffix match
    s1_suffix = s1_row.get("name_legal_suffix", "")
    cand_suffix = cand_row.get("name_legal_suffix", "")
    if s1_suffix and cand_suffix:
        suffix_match = 1.0 if s1_suffix == cand_suffix else 0.0
    elif not s1_suffix and not cand_suffix:
        suffix_match = 0.5
    else:
        suffix_match = 0.0

    len_s1_n = len(s1_name)
    len_cand_n = len(cand_name)
    name_len_diff = float(abs(len_s1_n - len_cand_n))
    name_len_ratio = min(len_s1_n, len_cand_n) / max(len_s1_n, len_cand_n, 1)

    # --- 2. Address Features ---
    s1_addr_empty = 1.0 if not s1_addr.strip() else 0.0
    cand_addr_empty = 1.0 if not cand_addr.strip() else 0.0
    both_addr_empty = 1.0 if (s1_addr_empty and cand_addr_empty) else 0.0

    if not s1_addr_empty and not cand_addr_empty:
        addr_lev = fuzz.ratio(s1_addr, cand_addr) / 100.0
        addr_token_sort = fuzz.token_sort_ratio(s1_addr, cand_addr) / 100.0
        addr_token_set = fuzz.token_set_ratio(s1_addr, cand_addr) / 100.0
        addr_jaccard = jaccard_similarity(s1_addr_tokens, cand_addr_tokens)
    else:
        addr_lev = 0.0
        addr_token_sort = 0.0
        addr_token_set = 0.0
        addr_jaccard = 0.0

    # Postal code match: 1.0 match, 0.0 mismatch, -1.0 missing
    s1_post = s1_row.get("address_postal_code", "")
    cand_post = cand_row.get("address_postal_code", "")
    if s1_post and cand_post:
        postal_match = 1.0 if s1_post == cand_post else 0.0
    else:
        postal_match = -1.0

    # House number match: 1.0 match, 0.0 mismatch, -1.0 missing
    s1_house = s1_row.get("address_house_number", "")
    cand_house = cand_row.get("address_house_number", "")
    if s1_house and cand_house:
        house_match = 1.0 if s1_house == cand_house else 0.0
    else:
        house_match = -1.0

    # Numeric tokens overlap (e.g. plot numbers, street numbers, pin codes)
    s1_nums = set(s1_row.get("address_numeric_tokens") or [])
    cand_nums = set(cand_row.get("address_numeric_tokens") or [])
    numeric_jaccard = jaccard_similarity(s1_nums, cand_nums)

    len_s1_a = len(s1_addr)
    len_cand_a = len(cand_addr)
    addr_len_ratio = min(len_s1_a, len_cand_a) / max(len_s1_a, len_cand_a, 1) if (len_s1_a > 0 and len_cand_a > 0) else 0.0

    # --- 3. Combined Features ---
    s1_full = f"{s1_name} {s1_addr}".strip()
    cand_full = f"{cand_name} {cand_addr}".strip()
    full_token_set = fuzz.token_set_ratio(s1_full, cand_full) / 100.0

    cand_id = cand_row.get("entity_id", "")
    is_s2 = 1.0 if cand_id.startswith("S2-") else 0.0
    is_s3 = 1.0 if cand_id.startswith("S3-") else 0.0

    country = s1_row.get("country_normalized", "")
    country_is_us = 1.0 if country == "US" else 0.0
    country_is_in = 1.0 if country == "India" else 0.0
    country_is_fr = 1.0 if country == "France" else 0.0

    return {
        "name_lev": name_lev,
        "name_partial": name_partial,
        "name_token_sort": name_token_sort,
        "name_token_set": name_token_set,
        "name_core_sort": name_core_sort,
        "name_jaccard": name_jaccard,
        "name_3gram_jaccard": name_3gram_jaccard,
        "name_4gram_jaccard": name_4gram_jaccard,
        "prefix_3_match": prefix_3_match,
        "prefix_5_match": prefix_5_match,
        "first_token_match": first_token_match,
        "suffix_match": suffix_match,
        "name_len_diff": name_len_diff,
        "name_len_ratio": name_len_ratio,
        "addr_lev": addr_lev,
        "addr_token_sort": addr_token_sort,
        "addr_token_set": addr_token_set,
        "addr_jaccard": addr_jaccard,
        "postal_match": postal_match,
        "house_match": house_match,
        "numeric_jaccard": numeric_jaccard,
        "s1_addr_empty": s1_addr_empty,
        "cand_addr_empty": cand_addr_empty,
        "both_addr_empty": both_addr_empty,
        "addr_len_ratio": addr_len_ratio,
        "full_token_set": full_token_set,
        "is_s2": is_s2,
        "is_s3": is_s3,
        "country_is_us": country_is_us,
        "country_is_in": country_is_in,
        "country_is_fr": country_is_fr,
    }


def compute_features_for_candidates(
    candidate_dict: Dict[str, List[str]],
    s1_lookup: Dict[str, dict],
    candidate_pool_lookup: Dict[str, dict],
) -> pd.DataFrame:
    """Compute pairwise feature dataframe with entity group ranking features.

    Args:
        candidate_dict: Mapping {s1_id: [candidate_ids]}
        s1_lookup: Mapping {s1_id: s1_record_dict}
        candidate_pool_lookup: Mapping {cand_id: cand_record_dict}

    Returns:
        DataFrame containing s1_id, candidate_id, all pairwise features,
        and group-level ranking features (rank, margin to best).
    """
    records = []
    
    for s1_id, cand_ids in candidate_dict.items():
        s1_row = s1_lookup.get(s1_id)
        if not s1_row or not cand_ids:
            continue

        cands_for_s1 = []
        for cand_id in cand_ids:
            cand_row = candidate_pool_lookup.get(cand_id)
            if not cand_row:
                continue
            
            feat = compute_pair_features(s1_row, cand_row)
            feat["s1_id"] = s1_id
            feat["candidate_id"] = cand_id
            cands_for_s1.append(feat)

        if not cands_for_s1:
            continue

        # Add group-level ranking features per S1 entity
        # Sort candidates descending by token_sort ratio
        cands_for_s1.sort(key=lambda x: x["name_token_sort"], reverse=True)
        best_name_sort = cands_for_s1[0]["name_token_sort"]
        n_cands = len(cands_for_s1)

        for rank, c_feat in enumerate(cands_for_s1, start=1):
            c_feat["candidate_rank"] = float(rank)
            c_feat["name_sort_diff_to_best"] = float(best_name_sort - c_feat["name_token_sort"])
            c_feat["total_candidates_for_s1"] = float(n_cands)
            records.append(c_feat)

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    # Ensure ID columns are first
    cols = ["s1_id", "candidate_id"] + [c for c in df.columns if c not in ("s1_id", "candidate_id")]
    return df[cols]
