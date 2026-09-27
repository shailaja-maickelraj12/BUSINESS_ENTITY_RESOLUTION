"""Comprehensive unit tests for normalization module.

Tests cover:
  1. Empty and null-like values
  2. Punctuation and '&' normalization
  3. Legal suffix extraction and canonicalization (US, India, France)
  4. Accented French text
  5. Indian address patterns and landmarks
  6. US ZIP and ZIP+4 postal codes
  7. Indian PIN codes
  8. French postal codes
  9. Business names containing meaningful numbers
  10. Idempotency: normalize(normalize(x)) == normalize(x)
  11. Unchanged entity IDs and preserved raw columns
  12. Unseen country values handling
"""

import sys
import os
import unittest
import pandas as pd

# Add src directory to path
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from config import NormalizationConfig
from normalize import (
    clean_basic_text,
    extract_legal_suffix,
    normalize_business_name,
    expand_address_abbreviations,
    extract_postal_code,
    extract_house_number,
    extract_numeric_tokens,
    normalize_address,
    normalize_country,
    normalize_dataframe,
)


class TestNormalize(unittest.TestCase):
    """Test suite for data normalization and component extraction."""

    def setUp(self):
        self.config = NormalizationConfig()

    def test_empty_and_null_values(self):
        """Test handling of empty, None, and whitespace strings."""
        self.assertEqual(clean_basic_text(""), "")
        self.assertEqual(clean_basic_text(None), "")
        self.assertEqual(clean_basic_text("   "), "")

        norm_name, core, sfx = normalize_business_name("")
        self.assertEqual((norm_name, core, sfx), ("", "", ""))

        norm_name, core, sfx = normalize_business_name(None)
        self.assertEqual((norm_name, core, sfx), ("", "", ""))

        addr, postal, house, nums, is_empty = normalize_address("")
        self.assertEqual((addr, postal, house, nums, is_empty), ("", "", "", [], 1))

        addr, postal, house, nums, is_empty = normalize_address(None)
        self.assertEqual((addr, postal, house, nums, is_empty), ("", "", "", [], 1))

    def test_punctuation_and_ampersand(self):
        """Test normalization of '&', '+', and punctuation marks."""
        raw = "AT&T + Verizon, Inc. (Global) - 100%!"
        cleaned = clean_basic_text(raw)
        self.assertEqual(cleaned, "at and t and verizon inc global 100")

        # Test legal suffix extraction on '&' expanded name
        norm_name, core, sfx = normalize_business_name("Barnes & Noble, Inc.", self.config)
        self.assertEqual(norm_name, "barnes and noble inc")
        self.assertEqual(core, "barnes and noble")
        self.assertEqual(sfx, "inc")

    def test_legal_suffixes(self):
        """Test legal suffix canonicalization across US, India, and France."""
        cases = [
            ("Alpha Corp.", "alpha", "corp"),
            ("Alpha Corporation", "alpha", "corp"),
            ("Beta LLC", "beta", "llc"),
            ("Beta L.L.C.", "beta", "llc"),
            ("Gamma Inc.", "gamma", "inc"),
            ("Delta Pvt. Ltd.", "delta", "pvt ltd"),
            ("Delta Private Limited", "delta", "pvt ltd"),
            ("Epsilon SARL", "epsilon", "sarl"),
            ("Epsilon S.A.R.L.", "epsilon", "sarl"),
            ("Zeta SASU", "zeta", "sasu"),
            ("Eta EURL", "eta", "eurl"),
            ("Theta SCI", "theta", "sci"),
            # Leading suffix
            ("LLP Turning Point Complex", "turning point complex", "llp"),
            ("SCI Ptit Amicale", "ptit amicale", "sci"),
        ]
        for raw, expected_core, expected_sfx in cases:
            with self.subTest(raw=raw):
                _, core, sfx = normalize_business_name(raw, self.config)
                self.assertEqual(core, expected_core)
                self.assertEqual(sfx, expected_sfx)

    def test_accented_french_text(self):
        """Test accent transliteration and French address cleaning."""
        raw_name = "Thermal & Fils Épicerie SASU"
        norm_name, core, sfx = normalize_business_name(raw_name, self.config)
        self.assertEqual(norm_name, "thermal and fils epicerie sasu")
        self.assertEqual(core, "thermal and fils epicerie")
        self.assertEqual(sfx, "sasu")

        raw_addr = "175 Boulevard du Président Franklin Roosevelt, Bordeaux"
        norm_addr, postal, house, _, _ = normalize_address(raw_addr, country="France", config=self.config)
        self.assertIn("president", norm_addr)
        self.assertEqual(house, "175")

    def test_indian_addresses(self):
        """Test Indian addresses with landmarks, complex numbering, and abbreviations."""
        addr = "Shop No. 25, Plot No.15/A, Opp. Railway St., Near SBI ATM, Navi Mumbai 400703"
        norm_addr, postal, house, nums, is_empty = normalize_address(addr, country="India", config=self.config)
        self.assertEqual(postal, "400703")
        self.assertEqual(house, "25")
        self.assertIn("opposite", norm_addr)
        self.assertIn("near", norm_addr)
        self.assertEqual(is_empty, 0)

    def test_us_zip_and_zip_plus_4(self):
        """Test US 5-digit ZIP and ZIP+4 extraction."""
        addr1 = "5902 Big Tree Road, Livonia, NY 14487"
        postal1 = extract_postal_code(addr1, country="US")
        self.assertEqual(postal1, "14487")

        addr2 = "1200 Main St, Dallas, TX 75201-1234"
        postal2 = extract_postal_code(addr2, country="US")
        self.assertEqual(postal2, "75201-1234")

    def test_indian_pin_codes(self):
        """Test Indian 6-digit PIN code extraction."""
        addr = "D - 7, Press Apartment, 23, I.P. Extension, Patparganj, Delhi 110092"
        postal = extract_postal_code(addr, country="India")
        self.assertEqual(postal, "110092")

    def test_french_postal_codes(self):
        """Test French 5-digit postal code extraction."""
        addr = "20 Rue Parmentier, Dunkerque, Hauts-de-France 59140"
        postal = extract_postal_code(addr, country="France")
        self.assertEqual(postal, "59140")

    def test_business_names_with_meaningful_numbers(self):
        """Test that numbers in business names are strictly preserved."""
        cases = [
            ("3M Company", "3m", "co"),
            ("7-Eleven Inc", "7 eleven", "inc"),
            ("Studio 54 LLC", "studio 54", "llc"),
            ("24/7 Express Pvt Ltd", "24 7 express", "pvt ltd"),
        ]
        for raw, expected_core, expected_sfx in cases:
            with self.subTest(raw=raw):
                norm_name, core, sfx = normalize_business_name(raw, self.config)
                self.assertEqual(core, expected_core)
                self.assertEqual(sfx, expected_sfx)

    def test_idempotency(self):
        """Test that normalize(normalize(x)) == normalize(x)."""
        sample_names = [
            "L&t Wéalth Pvt Ltd #11741",
            "M/s Gv Exports Limited Services",
            "Noify Software Ínc.",
            "Allied Medical Center LLC",
            "Thermal & Fils SASU",
            "7-Eleven Inc",
        ]
        for name in sample_names:
            norm1, core1, sfx1 = normalize_business_name(name, self.config)
            norm2, core2, sfx2 = normalize_business_name(norm1, self.config)
            self.assertEqual(norm1, norm2, f"Idempotency failed on full name for {name}")
            self.assertEqual(core1, core2, f"Idempotency failed on core name for {name}")

        sample_addrs = [
            "1212 Southbreeze Circle, Knoxville, TN 37922",
            "175 Boulevard du Président Franklin Roosevelt, Bordeaux 33000",
            "Shop No. 25, Opp. Railway St., Mumbai 400703",
        ]
        for addr in sample_addrs:
            norm1, p1, h1, n1, e1 = normalize_address(addr, "US", self.config)
            norm2, p2, h2, n2, e2 = normalize_address(norm1, "US", self.config)
            self.assertEqual(norm1, norm2, f"Idempotency failed on address for {addr}")

    def test_unchanged_entity_ids(self):
        """Test that normalize_dataframe preserves original fields without modification."""
        df = pd.DataFrame({
            "entity_id": ["S1-001", "S2-002"],
            "business_name": ["3M Company", "ZNB Club SARL"],
            "business_address": ["1200 Main St, Dallas, TX 75201", "5 bis Rue Pierre Dignac, 33260"],
            "country": ["US", "France"],
        })
        orig_copy = df.copy()
        result = normalize_dataframe(df, self.config)

        # Ensure raw columns remain exactly identical
        for col in ["entity_id", "business_name", "business_address", "country"]:
            pd.testing.assert_series_equal(df[col], orig_copy[col])
            pd.testing.assert_series_equal(result[col], orig_copy[col])

        # Verify added columns exist
        expected_added_cols = [
            "country_normalized",
            "name_normalized",
            "name_core",
            "name_legal_suffix",
            "name_tokens",
            "address_normalized",
            "address_postal_code",
            "address_house_number",
            "address_numeric_tokens",
            "address_is_empty",
            "address_tokens",
        ]
        for col in expected_added_cols:
            self.assertIn(col, result.columns)

    def test_unseen_country_values(self):
        """Test that open-set / unseen country values pass through safely."""
        self.assertEqual(normalize_country("US"), "US")
        self.assertEqual(normalize_country("India"), "India")
        self.assertEqual(normalize_country("France"), "France")
        self.assertEqual(normalize_country("germany"), "Germany")
        self.assertEqual(normalize_country("BRAZIL"), "Brazil")
        self.assertEqual(normalize_country("jp"), "JP")
        self.assertEqual(normalize_country(""), "UNKNOWN")
        self.assertEqual(normalize_country(None), "UNKNOWN")


if __name__ == "__main__":
    unittest.main()
