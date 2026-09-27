"""Global configuration settings for Business Entity Resolution pipeline.

Defines deterministic seeds, file paths, normalization dictionaries,
and pipeline runtime parameters.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

# Random seed for reproducibility across all stages
RANDOM_SEED: int = 42

# Base directories
BASE_DIR: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DATA_DIR: str = os.path.join(BASE_DIR, "student_resource", "dataset")
TRAIN_DIR: str = os.path.join(DATA_DIR, "train")
TEST_DIR: str = os.path.join(DATA_DIR, "test")
OUTPUT_DIR: str = os.path.join(BASE_DIR, "output")
REPORTS_DIR: str = os.path.join(BASE_DIR, "reports")

# Canonical file names
TRAIN_SOURCE1_PATH: str = os.path.join(TRAIN_DIR, "train_source1.tsv")
TRAIN_SOURCE2_PATH: str = os.path.join(TRAIN_DIR, "train_source2.tsv")
TRAIN_SOURCE3_PATH: str = os.path.join(TRAIN_DIR, "train_source3.tsv")
TRAIN_GROUND_TRUTH_PATH: str = os.path.join(TRAIN_DIR, "train_ground_truth.tsv")

TEST_SOURCE1_PATH: str = os.path.join(TEST_DIR, "test_source1.tsv")
TEST_SOURCE2_PATH: str = os.path.join(TEST_DIR, "test_source2.tsv")
TEST_SOURCE3_PATH: str = os.path.join(TEST_DIR, "test_source3.tsv")

OUTPUT_MATCHING_PATH: str = os.path.join(OUTPUT_DIR, "matching_results.tsv")
OUTPUT_CANDIDATES_PATH: str = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")

# Evaluation metric parameters
F_BETA: float = 0.5

@dataclass
class NormalizationConfig:
    """Configuration for name and address normalization."""
    # Whether to expand ambiguous abbreviations like 'st' -> 'street' vs 'saint'
    expand_ambiguous_street_saint: bool = False
    
    # Strip accents / diacritics using Unicode decomposition + unidecode
    strip_accents: bool = True
    
    # Character n-gram range for representation
    char_ngram_range: Tuple[int, int] = (2, 4)
    
    # Canonical legal suffixes dictionary: maps variant lower-case suffix to canonical form
    # Covers US, India, and France
    legal_suffixes: Dict[str, str] = field(default_factory=lambda: {
        # US / General English
        "inc": "inc",
        "incorporated": "inc",
        "corp": "corp",
        "corporation": "corp",
        "co": "co",
        "company": "co",
        "ltd": "ltd",
        "limited": "ltd",
        "llc": "llc",
        "l.l.c.": "llc",
        "llp": "llp",
        "l.l.p.": "llp",
        "lp": "lp",
        "plc": "plc",
        "pc": "pc",
        "p.c.": "pc",
        
        # India specific
        "pvt ltd": "pvt ltd",
        "private limited": "pvt ltd",
        "pvt limited": "pvt ltd",
        "private ltd": "pvt ltd",
        "pvt": "pvt",
        "private": "pvt",
        
        # France specific
        "sarl": "sarl",
        "s.a.r.l.": "sarl",
        "sas": "sas",
        "s.a.s.": "sas",
        "sasu": "sasu",
        "s.a.s.u.": "sasu",
        "eurl": "eurl",
        "e.u.r.l.": "eurl",
        "sa": "sa",
        "s.a.": "sa",
        "sci": "sci",
        "s.c.i.": "sci",
        "snc": "snc",
        "s.n.c.": "snc",
        "gie": "gie",
        "g.i.e.": "gie",
    })

    # Unambiguous address abbreviations: mapped to standard expanded tokens
    address_abbreviations: Dict[str, str] = field(default_factory=lambda: {
        # General / US
        "rd": "road",
        "ave": "avenue",
        "av": "avenue",
        "blvd": "boulevard",
        "bd": "boulevard",
        "dr": "drive",
        "ln": "lane",
        "cir": "circle",
        "ct": "court",
        "pl": "place",
        "sq": "square",
        "ste": "suite",
        "apt": "apartment",
        "fl": "floor",
        "bldg": "building",
        "dept": "department",
        "pkwy": "parkway",
        "hwy": "highway",
        "expwy": "expressway",
        
        # India
        "opp": "opposite",
        "nr": "near",
        "mkt": "market",
        "ngr": "nagar",
        "clny": "colony",
        "sec": "sector",
        "soc": "society",
        "apt": "apartment",
        "apts": "apartments",
        "comp": "complex",
        "cplx": "complex",
        "ind": "industrial",
        "est": "estate",
        "b/h": "behind",
        
        # France
        "r": "rue",
        "rte": "route",
        "imp": "impasse",
        "all": "allee",
        "chem": "chemin",
        "qu": "quai",
        "pass": "passage",
    })
