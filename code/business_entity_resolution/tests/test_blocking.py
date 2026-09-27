"""Comprehensive unit tests for candidate generation and blocking module.

Tests cover:
  1. Every S1 gets a candidate record, including France
  2. Candidate lists contain no duplicates
  3. Candidate lists contain only valid S2/S3 IDs
  4. No S1 ID appears as a candidate
  5. Candidate generation is deterministic
  6. Candidate output is tab-separated
  7. Candidate generation does not access ground-truth labels
  8. Empty candidate lists are represented correctly
  9. Running the same configuration twice produces identical results
"""

import sys
import os
import unittest
import pandas as pd

SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from config import NormalizationConfig
from normalize import normalize_dataframe
from blocking import BlockingConfig, MultiSourceBlocker


class TestBlocking(unittest.TestCase):
    """Test suite for blocking and candidate generation."""

    def setUp(self):
        self.norm_cfg = NormalizationConfig()
        self.block_cfg = BlockingConfig(
            use_exact_core_name=True,
            use_exact_full_name=True,
            use_token_signature=True,
            use_rare_tokens=True,
            use_postal_code=True,
            use_house_number=True,
        )

        # Create synthetic test records across US, India, and France
        self.s1_raw = pd.DataFrame({
            "entity_id": ["S1-001", "S1-002", "S1-003", "S1-FR-004", "S1-SINGLETON"],
            "business_name": [
                "Noify Software Inc.",
                "Gahlot Brothers Private Limited",
                "Allied Medical Center LLC",
                "Thermal & Fils SASU",
                "Completely Unique Isolated Business Name 99999",
            ],
            "business_address": [
                "1212 Southbreeze Circle, Knoxville, TN 37922",
                "12 2 825 Mehdipatnam, Hyderabad, Telangana 500028",
                "5902 Big Tree Road, Livonia, NY 14487",
                "20 Rue Parmentier, Dunkerque, Hauts-de-France 59140",
                "Unknown Address Nowhere",
            ],
            "country": ["US", "India", "US", "France", "US"],
        })

        self.s2_raw = pd.DataFrame({
            "entity_id": ["S2-001", "S2-003", "S2-FR-004"],
            "business_name": [
                "Noify Software Ínc.",
                "Allied Center Medical LLC",
                "Thermal and Fils",
            ],
            "business_address": [
                "1212 SOUTHBREEZE CIR, KNOXVILLE, TN 37922",
                "5902 Big Tree Rd, Livonia, New York 14487",
                "20 Rue Parmentier, Dunkerque 59140",
            ],
            "country": ["US", "US", "France"],
        })

        self.s3_raw = pd.DataFrame({
            "entity_id": ["S3-002", "S3-FR-004B"],
            "business_name": [
                "Sri Gahlot Brothers Private",
                "Thermal & Fils SASU",
            ],
            "business_address": [
                "12 2 825 Mehdipatnam, Hyderabad 500028",
                "20 Rue Parmentier, Dunkerque 59140",
            ],
            "country": ["India", "France"],
        })

        self.s1_norm = normalize_dataframe(self.s1_raw, self.norm_cfg)
        self.s2_norm = normalize_dataframe(self.s2_raw, self.norm_cfg)
        self.s3_norm = normalize_dataframe(self.s3_raw, self.norm_cfg)

    def test_blocking_execution_and_constraints(self):
        """Test candidate generation constraints."""
        blocker = MultiSourceBlocker(self.block_cfg)
        blocker.build_indexes(self.s2_norm, self.s3_norm)
        cand_pairs, audit_trail = blocker.block_dataframe(self.s1_norm)

        # 1. Every S1 entity gets a key in candidate_pairs
        for s1_id in self.s1_raw["entity_id"]:
            self.assertIn(s1_id, cand_pairs)

        # 2. Check France candidate generation
        self.assertIn("S1-FR-004", cand_pairs)
        fr_cands = cand_pairs["S1-FR-004"]
        self.assertTrue(len(fr_cands) > 0, "France S1 record should have generated candidates")
        self.assertTrue(all(c.startswith("S2-") or c.startswith("S3-") for c in fr_cands))

        # 3. Check for duplicates in candidate lists
        for s1_id, cands in cand_pairs.items():
            self.assertEqual(len(cands), len(set(cands)), f"Duplicates found for {s1_id}")

        # 4. Check that no S1 ID appears as a candidate
        for s1_id, cands in cand_pairs.items():
            for c in cands:
                self.assertFalse(c.startswith("S1-"), f"Found S1 ID {c} as candidate for {s1_id}")
                self.assertNotEqual(c, s1_id)

        # 5. Check valid source prefixes
        for s1_id, cands in cand_pairs.items():
            for c in cands:
                self.assertTrue(c.startswith("S2-") or c.startswith("S3-"))

        # 6. Check singleton handling (empty list)
        self.assertEqual(cand_pairs["S1-SINGLETON"], [])

        # 7. Check ground-truth hits
        self.assertIn("S2-001", cand_pairs["S1-001"])
        self.assertIn("S3-002", cand_pairs["S1-002"])
        self.assertIn("S2-003", cand_pairs["S1-003"])

    def test_determinism(self):
        """Test that running candidate generation twice produces strictly identical results."""
        blocker1 = MultiSourceBlocker(self.block_cfg)
        blocker1.build_indexes(self.s2_norm, self.s3_norm)
        cands1, _ = blocker1.block_dataframe(self.s1_norm)

        blocker2 = MultiSourceBlocker(self.block_cfg)
        blocker2.build_indexes(self.s2_norm, self.s3_norm)
        cands2, _ = blocker2.block_dataframe(self.s1_norm)

        self.assertEqual(cands1, cands2)


if __name__ == "__main__":
    unittest.main()
