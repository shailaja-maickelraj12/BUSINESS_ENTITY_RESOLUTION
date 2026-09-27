"""Scalable Candidate Generation and Blocking Module for Business Entity Resolution.

Implements multi-channel blocking within countries across Source 2 and Source 3:
  Rule 1: Exact or near-exact normalized name blocking (core name, full name, token signature)
  Rule 2: Informative/rare name-token inverted indexing with frequency-based pruning
  Rule 3: Structured address blocking (postal code and house/plot numbers with weak name overlap)
  Rule 4: Character n-gram TF-IDF cosine retrieval (top-k)
  Rule 5: Combined name-address TF-IDF retrieval (top-k)

Preserves audit trails of generating rules and enforces strict candidate constraints:
  - Candidates generated independently for S2 and S3, then unified
  - S1 IDs strictly excluded
  - No duplicate candidates per S1 entity
  - Country-restricted open-set matching
"""

import os
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from config import NormalizationConfig
from normalize import clean_basic_text, normalize_dataframe


@dataclass
class BlockingConfig:
    """Hyperparameters and channel toggles for candidate generation."""
    # Channel 1: Exact and signature name matching
    use_exact_core_name: bool = True
    use_exact_full_name: bool = True
    use_token_signature: bool = True

    # Channel 2: Rare token inverted index
    use_rare_tokens: bool = True
    rare_token_min_len: int = 3
    rare_token_max_df: int = 500       # Prune tokens occurring in > 500 records
    rare_token_min_df: int = 1         # Minimum document frequency
    max_cands_per_rare_token: int = 10  # Cap candidates returned per token

    # Channel 3: Structured address indexing
    use_postal_code: bool = True
    postal_code_max_df: int = 500      # Cap dense postal codes
    max_cands_per_postal: int = 15     # Max candidates per postal code
    use_house_number: bool = True
    house_number_max_df: int = 500     # Cap common house numbers (e.g., '1')
    max_cands_per_house: int = 15      # Max candidates per house number

    # Channel 4: Character n-gram TF-IDF cosine retrieval
    use_char_ngram: bool = False
    char_ngram_range: Tuple[int, int] = (3, 3)
    char_ngram_top_k: int = 5
    char_ngram_min_sim: float = 0.50

    # Channel 5: Combined name + address TF-IDF retrieval
    use_combined_tfidf: bool = False
    combined_top_k: int = 3
    combined_min_sim: float = 0.40

    # Fallback and pre-filter caps
    max_total_candidates_per_entity: int = 30  # Safety cap per S1 entity
    min_name_token_overlap_for_address: bool = True  # Require >= 1 shared name char/token for addr blocks


# Stop words derived strictly from business records without external lookups
COMMON_BUSINESS_STOPWORDS: Set[str] = {
    "and", "the", "of", "in", "for", "at", "to", "by", "on", "with",
    "co", "company", "inc", "corp", "corporation", "ltd", "limited",
    "pvt", "private", "llc", "llp", "services", "service", "group",
    "solutions", "enterprises", "enterprise", "center", "centre",
    "industries", "international", "holding", "holdings", "management",
    "associates", "consulting", "technologies", "technology", "systems",
    "india", "us", "usa", "france", "sarl", "sas", "sasu", "eurl", "sci",
}


def compute_token_signature(tokens: List[str]) -> str:
    """Compute sorted token signature string for word-order invariant matching."""
    informative = [t for t in tokens if t not in COMMON_BUSINESS_STOPWORDS and len(t) > 1]
    if not informative:
        informative = tokens
    return " ".join(sorted(informative))


class CountrySourceIndex:
    """Inverted index structures for a single source (S2 or S3) within a specific country."""

    def __init__(self, source_name: str, country: str):
        self.source_name = source_name
        self.country = country
        self.record_ids: List[str] = []
        self.core_names: List[str] = []
        self.addresses: List[str] = []

        # Inverted index mappings: key -> list of record row indices
        self.core_name_index: Dict[str, List[int]] = defaultdict(list)
        self.full_name_index: Dict[str, List[int]] = defaultdict(list)
        self.token_sig_index: Dict[str, List[int]] = defaultdict(list)
        self.rare_token_index: Dict[str, List[int]] = defaultdict(list)
        self.postal_index: Dict[str, List[int]] = defaultdict(list)
        self.house_index: Dict[str, List[int]] = defaultdict(list)

        # TF-IDF vectorizers and matrices
        self.char_vectorizer: Optional[TfidfVectorizer] = None
        self.char_matrix: Optional[csr_matrix] = None
        self.combined_vectorizer: Optional[TfidfVectorizer] = None
        self.combined_matrix: Optional[csr_matrix] = None

    def build(self, df: pd.DataFrame, config: BlockingConfig) -> None:
        """Build all active inverted indexes for this source and country partition.

        Args:
            df: Normalized DataFrame for candidate source records.
            config: BlockingConfig instance.
        """
        n_records = len(df)
        self.record_ids = list(df["entity_id"])
        self.core_names = list(df["name_core"])
        self.addresses = list(df["address_normalized"])

        # Compute document frequencies of all name tokens to identify rare informative tokens
        token_df_counter: Counter = Counter()
        token_to_rows: Dict[str, List[int]] = defaultdict(list)

        for row_idx, (r_id, core_name, full_name, tokens, postal, house) in enumerate(
            zip(
                self.record_ids,
                self.core_names,
                df["name_normalized"],
                df["name_tokens"],
                df["address_postal_code"],
                df["address_house_number"],
            )
        ):
            # 1. Exact core and full name
            if core_name:
                self.core_name_index[core_name].append(row_idx)
            if full_name and full_name != core_name:
                self.full_name_index[full_name].append(row_idx)

            # 2. Token signature
            sig = compute_token_signature(tokens)
            if sig and sig != core_name:
                self.token_sig_index[sig].append(row_idx)

            # 3. Token document frequencies
            unique_tokens = set(tokens)
            for tok in unique_tokens:
                if tok not in COMMON_BUSINESS_STOPWORDS and len(tok) >= config.rare_token_min_len:
                    token_df_counter[tok] += 1
                    token_to_rows[tok].append(row_idx)

            # 4. Postal code
            if postal:
                self.postal_index[postal].append(row_idx)

            # 5. House number
            if house:
                self.house_index[house].append(row_idx)

        # Build rare token index filtered by frequency thresholds
        if config.use_rare_tokens:
            for tok, count in token_df_counter.items():
                if config.rare_token_min_df <= count <= config.rare_token_max_df:
                    self.rare_token_index[tok] = token_to_rows[tok]

        # Build optional TF-IDF matrices
        if config.use_char_ngram and n_records > 0:
            self.char_vectorizer = TfidfVectorizer(
                analyzer="char",
                ngram_range=config.char_ngram_range,
                min_df=2,
                sublinear_tf=True
            )
            # Use core_names for char n-grams
            self.char_matrix = self.char_vectorizer.fit_transform(self.core_names)

        if config.use_combined_tfidf and n_records > 0:
            combined_texts = [
                f"{n} {a}".strip() for n, a in zip(self.core_names, self.addresses)
            ]
            self.combined_vectorizer = TfidfVectorizer(
                analyzer="word",
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.5,
                sublinear_tf=True
            )
            self.combined_matrix = self.combined_vectorizer.fit_transform(combined_texts)

    def query_entity(
        self,
        core_name: str,
        full_name: str,
        name_tokens: List[str],
        postal_code: str,
        house_number: str,
        address_text: str,
        config: BlockingConfig
    ) -> Dict[str, List[str]]:
        """Retrieve candidate IDs and the rules that triggered them for a single S1 record.

        Args:
            core_name: S1 core business name.
            full_name: S1 normalized full name.
            name_tokens: S1 name tokens list.
            postal_code: S1 extracted postal code.
            house_number: S1 extracted house number.
            address_text: S1 normalized address text.
            config: BlockingConfig.

        Returns:
            Dictionary mapping candidate_id -> list of rule names.
        """
        candidates: Dict[str, List[str]] = defaultdict(list)
        name_token_set = set(name_tokens) - COMMON_BUSINESS_STOPWORDS

        # Rule 1a: Exact core name
        if config.use_exact_core_name and core_name in self.core_name_index:
            for row_idx in self.core_name_index[core_name]:
                candidates[self.record_ids[row_idx]].append("exact_core_name")

        # Rule 1b: Exact full name
        if config.use_exact_full_name and full_name in self.full_name_index:
            for row_idx in self.full_name_index[full_name]:
                candidates[self.record_ids[row_idx]].append("exact_full_name")

        # Rule 1c: Token signature
        if config.use_token_signature:
            sig = compute_token_signature(name_tokens)
            if sig and sig in self.token_sig_index:
                for row_idx in self.token_sig_index[sig]:
                    candidates[self.record_ids[row_idx]].append("token_signature")

        # Rule 2: Rare name tokens
        if config.use_rare_tokens:
            for tok in name_tokens:
                if tok in self.rare_token_index:
                    matched_rows = self.rare_token_index[tok]
                    # Cap candidates from a single rare token
                    for row_idx in matched_rows[:config.max_cands_per_rare_token]:
                        candidates[self.record_ids[row_idx]].append(f"rare_token:{tok}")

        # Rule 3a: Postal code block (gated by weak name condition)
        if config.use_postal_code and postal_code and postal_code in self.postal_index:
            matched_rows = self.postal_index[postal_code]
            if len(matched_rows) <= config.postal_code_max_df:
                for row_idx in matched_rows[:config.max_cands_per_postal]:
                    cand_name = self.core_names[row_idx]
                    cand_tokens = set(cand_name.split()) - COMMON_BUSINESS_STOPWORDS
                    # Require weak name agreement: share at least one name token or first 2 chars
                    if not config.min_name_token_overlap_for_address or (
                        bool(name_token_set & cand_tokens) or (core_name[:2] == cand_name[:2] if len(core_name) >= 2 else False)
                    ):
                        candidates[self.record_ids[row_idx]].append("postal_code_block")

        # Rule 3b: House number block (gated by weak name condition)
        if config.use_house_number and house_number and house_number in self.house_index:
            matched_rows = self.house_index[house_number]
            if len(matched_rows) <= config.house_number_max_df:
                for row_idx in matched_rows[:config.max_cands_per_house]:
                    cand_name = self.core_names[row_idx]
                    cand_tokens = set(cand_name.split()) - COMMON_BUSINESS_STOPWORDS
                    if not config.min_name_token_overlap_for_address or (
                        bool(name_token_set & cand_tokens) or (core_name[:2] == cand_name[:2] if len(core_name) >= 2 else False)
                    ):
                        candidates[self.record_ids[row_idx]].append("house_number_block")

        # Rule 4: Character n-gram TF-IDF top-k
        if config.use_char_ngram and self.char_vectorizer and self.char_matrix is not None and core_name:
            try:
                q_vec = self.char_vectorizer.transform([core_name])
                sims = self.char_matrix.dot(q_vec.T).toarray().ravel()
                top_indices = np.argpartition(sims, -config.char_ngram_top_k)[-config.char_ngram_top_k:]
                for row_idx in top_indices:
                    sim_val = sims[row_idx]
                    if sim_val >= config.char_ngram_min_sim:
                        candidates[self.record_ids[row_idx]].append(f"char_tfidf:{sim_val:.2f}")
            except Exception:
                pass

        # Rule 5: Combined name-address TF-IDF top-k
        if config.use_combined_tfidf and self.combined_vectorizer and self.combined_matrix is not None:
            comb_text = f"{core_name} {address_text}".strip()
            if comb_text:
                try:
                    q_vec = self.combined_vectorizer.transform([comb_text])
                    sims = self.combined_matrix.dot(q_vec.T).toarray().ravel()
                    top_indices = np.argpartition(sims, -config.combined_top_k)[-config.combined_top_k:]
                    for row_idx in top_indices:
                        sim_val = sims[row_idx]
                        if sim_val >= config.combined_min_sim:
                            candidates[self.record_ids[row_idx]].append(f"combined_tfidf:{sim_val:.2f}")
                except Exception:
                    pass

        return candidates


class MultiSourceBlocker:
    """Coordinates candidate generation across Source 2 and Source 3 partitioned by country."""

    def __init__(self, config: Optional[BlockingConfig] = None):
        self.config = config or BlockingConfig()
        # Nested dict: indexes[country][source_label] = CountrySourceIndex
        self.indexes: Dict[str, Dict[str, CountrySourceIndex]] = defaultdict(dict)

    def build_indexes(self, s2_df: pd.DataFrame, s3_df: pd.DataFrame) -> None:
        """Build independent inverted indexes for S2 and S3 for each country in the data.

        Args:
            s2_df: Normalized Source 2 DataFrame.
            s3_df: Normalized Source 3 DataFrame.
        """
        # Collect all unique countries present across sources
        countries = set(s2_df["country_normalized"]).union(set(s3_df["country_normalized"]))
        print(f"[Blocker] Building blocking indexes for countries: {sorted(countries)}")

        for c in sorted(countries):
            t0 = time.time()
            s2_c = s2_df[s2_df["country_normalized"] == c]
            s3_c = s3_df[s3_df["country_normalized"] == c]

            idx_s2 = CountrySourceIndex("S2", c)
            idx_s2.build(s2_c, self.config)
            self.indexes[c]["S2"] = idx_s2

            idx_s3 = CountrySourceIndex("S3", c)
            idx_s3.build(s3_c, self.config)
            self.indexes[c]["S3"] = idx_s3

            print(
                f"[Blocker] Country {c}: Indexed {len(s2_c):,} S2 records and "
                f"{len(s3_c):,} S3 records in {time.time() - t0:.2f}s."
            )

    def generate_candidates_for_entity(
        self,
        s1_row: pd.Series
    ) -> Tuple[List[str], Dict[str, List[str]]]:
        """Generate candidate IDs for one Source 1 entity across both S2 and S3."""
        return self.fast_generate_candidates(
            country=s1_row["country_normalized"],
            s1_id=s1_row["entity_id"],
            core_name=s1_row["name_core"],
            full_name=s1_row["name_normalized"],
            name_tokens=s1_row["name_tokens"],
            postal=s1_row["address_postal_code"],
            house=s1_row["address_house_number"],
            addr=s1_row["address_normalized"],
        )

    def fast_generate_candidates(
        self,
        country: str,
        s1_id: str,
        core_name: str,
        full_name: str,
        name_tokens: List[str],
        postal: str,
        house: str,
        addr: str,
    ) -> Tuple[List[str], Dict[str, List[str]]]:
        """Fast candidate generation avoiding pd.Series overhead."""
        candidates_map: Dict[str, List[str]] = {}

        if country not in self.indexes:
            return [], {}

        # Query S2 and S3 indexes independently
        for src in ("S2", "S3"):
            if src in self.indexes[country]:
                src_cands = self.indexes[country][src].query_entity(
                    core_name=core_name,
                    full_name=full_name,
                    name_tokens=name_tokens,
                    postal_code=postal,
                    house_number=house,
                    address_text=addr,
                    config=self.config
                )
                for cand_id, rules in src_cands.items():
                    # Sanity constraint: Never include S1 ID
                    if cand_id == s1_id or cand_id.startswith("S1-"):
                        continue
                    # Sanity constraint: Ensure proper source prefix
                    if not (cand_id.startswith("S2-") or cand_id.startswith("S3-")):
                        continue
                    if cand_id not in candidates_map:
                        candidates_map[cand_id] = []
                    candidates_map[cand_id].extend(rules)

        # Enforce maximum candidate cap if configured
        if self.config.max_total_candidates_per_entity and len(candidates_map) > self.config.max_total_candidates_per_entity:
            # Sort candidates by number of triggering rules descending (higher confidence first)
            sorted_cands = sorted(candidates_map.items(), key=lambda x: len(x[1]), reverse=True)
            candidates_map = dict(sorted_cands[:self.config.max_total_candidates_per_entity])

        candidate_ids = list(candidates_map.keys())
        return candidate_ids, candidates_map

    def block_dataframe(
        self,
        s1_df: pd.DataFrame
    ) -> Tuple[Dict[str, List[str]], Dict[str, Dict[str, List[str]]]]:
        """Generate candidate sets for all S1 entities in a DataFrame.

        Args:
            s1_df: Normalized Source 1 DataFrame.

        Returns:
            Tuple of:
              - candidate_pairs: Dict mapping s1_id -> list of candidate_ids.
              - audit_trail: Dict mapping s1_id -> {candidate_id: list of triggering rules}.
        """
        candidate_pairs: Dict[str, List[str]] = {}
        audit_trail: Dict[str, Dict[str, List[str]]] = {}

        for country, s1_id, core_name, full_name, name_tokens, postal, house, addr in zip(
            s1_df["country_normalized"].values,
            s1_df["entity_id"].values,
            s1_df["name_core"].values,
            s1_df["name_normalized"].values,
            s1_df["name_tokens"].values,
            s1_df["address_postal_code"].values,
            s1_df["address_house_number"].values,
            s1_df["address_normalized"].values,
        ):
            cands, rules_map = self.fast_generate_candidates(
                country=country,
                s1_id=s1_id,
                core_name=core_name,
                full_name=full_name,
                name_tokens=name_tokens,
                postal=postal,
                house=house,
                addr=addr,
            )
            candidate_pairs[s1_id] = cands
            audit_trail[s1_id] = rules_map

        return candidate_pairs, audit_trail

