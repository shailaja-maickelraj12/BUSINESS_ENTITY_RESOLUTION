import os
import sys
import pytest
import numpy as np

SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from features import char_ngrams, jaccard_similarity, compute_pair_features, compute_features_for_candidates


def test_char_ngrams_and_jaccard():
    ng = char_ngrams("apple", 3)
    assert ng == {"app", "ppl", "ple"}
    
    sim = jaccard_similarity({"a", "b"}, {"a", "c"})
    assert sim == pytest.approx(1.0 / 3.0)
    assert jaccard_similarity(set(), set()) == 0.0


def test_compute_pair_features_exact():
    s1 = {
        "entity_id": "S1-100",
        "name_normalized": "acme supply corp",
        "name_core": "acme supply",
        "name_legal_suffix": "corp",
        "address_normalized": "123 main st seattle wa 98101",
        "address_postal_code": "98101",
        "address_house_number": "123",
        "address_numeric_tokens": ["123", "98101"],
        "country_normalized": "US",
    }
    cand = {
        "entity_id": "S2-200",
        "name_normalized": "acme supply corp",
        "name_core": "acme supply",
        "name_legal_suffix": "corp",
        "address_normalized": "123 main st seattle wa 98101",
        "address_postal_code": "98101",
        "address_house_number": "123",
        "address_numeric_tokens": ["123", "98101"],
        "country_normalized": "US",
    }
    feat = compute_pair_features(s1, cand)
    assert feat["name_lev"] == pytest.approx(1.0)
    assert feat["name_token_sort"] == pytest.approx(1.0)
    assert feat["addr_token_sort"] == pytest.approx(1.0)
    assert feat["postal_match"] == 1.0
    assert feat["house_match"] == 1.0
    assert feat["is_s2"] == 1.0
    assert feat["is_s3"] == 0.0
    assert feat["country_is_us"] == 1.0


def test_compute_pair_features_missing_address():
    s1 = {
        "entity_id": "S1-101",
        "name_normalized": "global tech inc",
        "name_core": "global tech",
        "name_legal_suffix": "inc",
        "address_normalized": "",
        "address_postal_code": "",
        "address_house_number": "",
        "address_numeric_tokens": [],
        "country_normalized": "India",
    }
    cand = {
        "entity_id": "S3-300",
        "name_normalized": "global tech pvt ltd",
        "name_core": "global tech",
        "name_legal_suffix": "pvt ltd",
        "address_normalized": "45 brigade road bangalore",
        "address_postal_code": "",
        "address_house_number": "45",
        "address_numeric_tokens": ["45"],
        "country_normalized": "India",
    }
    feat = compute_pair_features(s1, cand)
    assert feat["s1_addr_empty"] == 1.0
    assert feat["cand_addr_empty"] == 0.0
    assert feat["postal_match"] == -1.0  # Missing
    assert feat["name_core_sort"] == pytest.approx(1.0)
    assert feat["is_s3"] == 1.0
    assert feat["country_is_in"] == 1.0


def test_compute_features_for_candidates():
    s1_dict = {
        "S1-1": {
            "entity_id": "S1-1",
            "name_normalized": "alpha pharmaceuticals",
            "name_core": "alpha pharmaceuticals",
            "name_legal_suffix": "",
            "address_normalized": "100 central ave",
            "address_postal_code": "10001",
            "address_house_number": "100",
            "address_numeric_tokens": ["100"],
            "country_normalized": "US",
        }
    }
    cand_dict = {
        "S2-1": {
            "entity_id": "S2-1",
            "name_normalized": "alpha pharmaceuticals inc",
            "name_core": "alpha pharmaceuticals",
            "name_legal_suffix": "inc",
            "address_normalized": "100 central avenue",
            "address_postal_code": "10001",
            "address_house_number": "100",
            "address_numeric_tokens": ["100"],
            "country_normalized": "US",
        },
        "S3-1": {
            "entity_id": "S3-1",
            "name_normalized": "beta logistics",
            "name_core": "beta logistics",
            "name_legal_suffix": "",
            "address_normalized": "200 side st",
            "address_postal_code": "20002",
            "address_house_number": "200",
            "address_numeric_tokens": ["200"],
            "country_normalized": "US",
        },
    }
    cands = {"S1-1": ["S2-1", "S3-1"]}
    df = compute_features_for_candidates(cands, s1_dict, cand_dict)
    
    assert len(df) == 2
    assert "candidate_rank" in df.columns
    assert "name_sort_diff_to_best" in df.columns
    # Check that S2-1 (close match) is ranked 1st
    top_cand = df.iloc[0]
    assert top_cand["candidate_id"] == "S2-1"
    assert top_cand["candidate_rank"] == 1.0
    assert top_cand["name_sort_diff_to_best"] == 0.0
