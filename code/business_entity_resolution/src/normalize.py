"""Text cleaning and normalization module for Business Entity Resolution.

Provides country-agnostic, rule-based, and conservative normalization for:
  - Business names (accent stripping, case folding, '&' expansion, legal suffix stripping)
  - Addresses (conservative abbreviation expansion, postal code extraction, house number extraction)
  - Country labels (open-set string normalization)

Preserves all raw fields and guarantees idempotency: normalize(normalize(x)) == normalize(x).
"""

import re
import unicodedata
from typing import Dict, List, Optional, Set, Tuple
import pandas as pd
import unidecode

from config import NormalizationConfig

# Pre-compiled regex patterns for performance across millions of rows
RE_ACRONYM_DOTS = re.compile(r"(?<=\b[a-zA-Z])\.(?=[a-zA-Z]\b|[^\w]|$)", re.UNICODE)
RE_AMPERSAND = re.compile(r"[\&\+]", re.UNICODE)
RE_PUNCTUATION = re.compile(r"[^\w\s]", re.UNICODE)
RE_WHITESPACE = re.compile(r"\s+", re.UNICODE)
RE_NUMBERS = re.compile(r"\b\d+\b")

# Postal code patterns
RE_POSTAL_INDIA = re.compile(r"\b([1-9]\d{5})\b")
RE_POSTAL_US = re.compile(r"\b(\d{5}(?:-\d{4})?)\b")
RE_POSTAL_FRANCE = re.compile(r"\b((?:0[1-9]|[1-8]\d|9[0-8])\d{3})\b")

# House / plot / door number patterns
RE_DOOR_PREFIX = re.compile(
    r"\b(?:door\s*no\.?|shop\s*no\.?|plot\s*no\.?|h\.?\s*no\.?|no\.?|unit|#)\s*"
    r"([a-zA-Z0-9\-/\s]+?)(?:,|$|\s+(?:rue|street|road|ave|avenue|blvd|lane|st|rd|dr))",
    re.IGNORECASE
)
RE_LEAD_HOUSE = re.compile(
    r"^\s*\(?\s*([a-zA-Z]?-?\d+[a-zA-Z]?(?:[\-\s/]+\d+[a-zA-Z]?)?(?:\s+bis|\s+ter)?)\s*\)?\b",
    re.IGNORECASE
)
RE_MID_HOUSE = re.compile(
    r"\b(\d+(?:[\-\s]+\d+)?(?:\s+bis|\s+ter)?)\s*,?\s*"
    r"(?:rue|bd|boulevard|ave|avenue|road|rd|street|st|lane|ln|drive|dr|circle|cir|all[eé]e|impasse)\b",
    re.IGNORECASE
)


def clean_basic_text(text: Optional[str]) -> str:
    """Perform baseline Unicode NFKC, accent stripping, lowercasing, and whitespace collapse.

    Guarantees idempotency: clean_basic_text(clean_basic_text(x)) == clean_basic_text(x).

    Args:
        text: Input string or None.

    Returns:
        Cleaned, lowercased ASCII-compatible string with single spaces.
    """
    if not text:
        return ""

    # 1. Unicode NFKC normalization
    s = unicodedata.normalize("NFKC", str(text))

    # 3. Transliterate accents and diacritics to ASCII (unidecode handles Latin, Cyrillic, Greek, etc.)
    s = unidecode.unidecode(s)

    # 4. Remove dots from acronyms/abbreviations before replacing punctuation (e.g. L.L.C. -> LLC, S.A.R.L. -> SARL)
    s = RE_ACRONYM_DOTS.sub("", s)

    # 5. Case folding / lowercasing
    s = s.lower()

    # 6. Standardize '&' and '+' to 'and'
    s = RE_AMPERSAND.sub(" and ", s)

    # 7. Remove punctuation symbols while preserving alphanumerics
    s = RE_PUNCTUATION.sub(" ", s)

    # 8. Collapse consecutive whitespace and strip leading/trailing spaces
    s = RE_WHITESPACE.sub(" ", s).strip()

    return s


_SUFFIX_CACHE: Dict[int, List[Tuple[re.Pattern, re.Pattern, str]]] = {}


def get_compiled_suffixes(suffixes_dict: Dict[str, str]) -> List[Tuple[re.Pattern, re.Pattern, str]]:
    dict_id = id(suffixes_dict)
    if dict_id not in _SUFFIX_CACHE:
        sorted_suffixes = sorted(suffixes_dict.items(), key=lambda x: len(x[0]), reverse=True)
        compiled = [
            (
                re.compile(r"^" + re.escape(sfx) + r"\b\s*"),
                re.compile(r"\s*\b" + re.escape(sfx) + r"$"),
                canon
            )
            for sfx, canon in sorted_suffixes
        ]
        _SUFFIX_CACHE[dict_id] = compiled
    return _SUFFIX_CACHE[dict_id]


def extract_legal_suffix(
    normalized_name: str,
    suffixes_dict: Dict[str, str]
) -> Tuple[str, str]:
    """Extract and remove legal entity suffixes from both ends of a normalized name.

    Handles leading suffixes (e.g. 'llp turning point...') and trailing suffixes
    (e.g. 'noify software inc', 'gahlot brothers private limited').
    Multi-word suffixes are tested first to prevent partial matches.

    Args:
        normalized_name: Lowercased, cleaned business name.
        suffixes_dict: Dictionary mapping suffix variants to canonical forms.

    Returns:
        Tuple of (core_name, canonical_suffix).
        If no suffix is found, returns (normalized_name, "").
    """
    if not normalized_name:
        return "", ""

    core = normalized_name
    found_suffix = ""

    compiled_suffixes = get_compiled_suffixes(suffixes_dict)

    # 1. Check leading suffix (e.g., 'llp ...', 'sci ...')
    for lead_pat, _, canon in compiled_suffixes:
        if lead_pat.search(core):
            found_suffix = canon
            core = lead_pat.sub("", core).strip()
            break

    # 2. Check trailing suffix (e.g., '... inc', '... private limited')
    for _, trail_pat, canon in compiled_suffixes:
        if trail_pat.search(core):
            if not found_suffix:
                found_suffix = canon
            core = trail_pat.sub("", core).strip()
            break

    # If removing the suffix empties the core name (e.g., single-word name 'inc'), restore original
    if not core:
        core = normalized_name

    return core, found_suffix


def normalize_business_name(
    name: Optional[str],
    config: Optional[NormalizationConfig] = None
) -> Tuple[str, str, str]:
    """Normalize a business name into full normalized name, core name, and legal suffix.

    Args:
        name: Raw business name string.
        config: NormalizationConfig instance (defaults to standard config).

    Returns:
        Tuple of:
          - name_normalized: Full cleaned name.
          - name_core: Core business name without legal suffixes.
          - name_legal_suffix: Canonical legal suffix or empty string.
    """
    if not name:
        return "", "", ""

    cfg = config or NormalizationConfig()
    norm_name = clean_basic_text(name)
    core_name, suffix = extract_legal_suffix(norm_name, cfg.legal_suffixes)

    return norm_name, core_name, suffix


def expand_address_abbreviations(
    address_text: str,
    abbrev_dict: Dict[str, str],
    expand_street_saint: bool = False
) -> str:
    """Expand unambiguous address abbreviations using word-boundary matching.

    Ambiguous abbreviations like 'st' (street vs saint) are preserved as 'st'
    unless expand_street_saint is explicitly enabled.

    Args:
        address_text: Cleaned address string.
        abbrev_dict: Dictionary mapping abbreviations to expanded tokens.
        expand_street_saint: Flag controlling expansion of 'st'.

    Returns:
        Address string with abbreviations expanded.
    """
    if not address_text:
        return ""

    tokens = address_text.split()
    expanded_tokens: List[str] = []

    for i, tok in enumerate(tokens):
        if tok in abbrev_dict:
            expanded_tokens.append(abbrev_dict[tok])
        elif tok == "st":
            if expand_street_saint:
                expanded_tokens.append("street")
            else:
                # Keep 'st' intact to avoid wrongly converting 'saint' to 'street'
                expanded_tokens.append("st")
        else:
            expanded_tokens.append(tok)

    return " ".join(expanded_tokens)


def extract_postal_code(address: str, country: str = "") -> str:
    """Extract postal code with country-aware regexes and open-set fallback.

    Supports:
      - US: 5-digit ZIP or ZIP+4 (e.g., 90210 or 90210-1234)
      - India: 6-digit PIN code (e.g., 500028)
      - France: 5-digit postal code (e.g., 33000, 75008)
      - Unseen countries: check 6-digit first, then 5-digit.

    Args:
        address: Raw or normalized address text.
        country: Country label (e.g. 'US', 'India', 'France', or other).

    Returns:
        Extracted postal code string or empty string.
    """
    if not address:
        return ""

    c = country.strip().upper() if country else ""

    if c == "INDIA":
        m = RE_POSTAL_INDIA.search(address)
        return m.group(1) if m else ""
    elif c == "US":
        m = RE_POSTAL_US.search(address)
        return m.group(1) if m else ""
    elif c == "FRANCE":
        m = RE_POSTAL_FRANCE.search(address)
        return m.group(1) if m else ""
    else:
        # Open-set country fallback: test 6-digit first to prevent partial 5-digit match
        m6 = RE_POSTAL_INDIA.search(address)
        if m6:
            return m6.group(1)
        m5 = RE_POSTAL_US.search(address)
        if m5:
            return m5.group(1)
        return ""


def extract_house_number(address: str) -> str:
    """Extract house, plot, door, or building identifier from address text.

    Args:
        address: Address text.

    Returns:
        Extracted house number string or empty string.
    """
    if not address:
        return ""

    s = address.strip()

    # 1. Check explicit prefixes like 'Door No', 'Shop No', 'Plot No', 'H. No.', 'Unit', '#'
    pref = RE_DOOR_PREFIX.search(s)
    if pref:
        val = pref.group(1).strip()
        tokens = val.split()
        if tokens and len(tokens[0]) <= 10:
            return " ".join(tokens[:2]).lower()

    # 2. Check leading house number like '1212', 'G-21', '5 bis', '(41)', 'D - 7'
    m_lead = RE_LEAD_HOUSE.match(s)
    if m_lead:
        val = m_lead.group(1).strip()
        val = re.sub(r"\s*-\s*", "-", val)
        if any(c.isdigit() for c in val) and len(val) <= 15:
            return val.lower()

    # 3. Check standalone number preceding a street type
    m_mid = RE_MID_HOUSE.search(s)
    if m_mid:
        val = m_mid.group(1).strip()
        val = re.sub(r"\s*-\s*", "-", val)
        return val.lower()

    return ""


def extract_numeric_tokens(text: str) -> List[str]:
    """Extract all standalone numeric tokens from text.

    Args:
        text: Input string.

    Returns:
        List of numeric token strings.
    """
    if not text:
        return []
    return RE_NUMBERS.findall(text)


def normalize_address(
    address: Optional[str],
    country: str = "",
    config: Optional[NormalizationConfig] = None
) -> Tuple[str, str, str, List[str], int]:
    """Normalize address and extract structured components.

    Args:
        address: Raw business address.
        country: Country string label.
        config: NormalizationConfig instance.

    Returns:
        Tuple of:
          - address_normalized: Cleaned, abbreviation-expanded address.
          - postal_code: Extracted postal code.
          - house_number: Extracted house/door number.
          - numeric_tokens: List of numeric strings.
          - is_empty: 1 if raw address was empty, 0 otherwise.
    """
    if not address or not address.strip():
        return "", "", "", [], 1

    cfg = config or NormalizationConfig()
    raw_addr = address.strip()

    # Extract postal code and house number before aggressive punctuation stripping
    postal_code = extract_postal_code(raw_addr, country)
    house_num = extract_house_number(raw_addr)

    # Clean text and expand conservative abbreviations
    cleaned = clean_basic_text(raw_addr)
    expanded = expand_address_abbreviations(
        cleaned,
        cfg.address_abbreviations,
        expand_street_saint=cfg.expand_ambiguous_street_saint
    )
    numeric_tokens = extract_numeric_tokens(expanded)

    return expanded, postal_code, house_num, numeric_tokens, 0


def normalize_country(country: Optional[str]) -> str:
    """Normalize country string conservatively without filtering or one-hot encoding.

    Allows unseen countries (e.g. France, Germany) to pass through cleanly.

    Args:
        country: Raw country string.

    Returns:
        Standardized country string.
    """
    if not country:
        return "UNKNOWN"

    c = country.strip()
    c_lower = c.lower()

    if c_lower in {"us", "usa", "united states", "united states of america"}:
        return "US"
    elif c_lower in {"india", "ind", "bharat"}:
        return "India"
    elif c_lower in {"france", "fra", "fr"}:
        return "France"
    else:
        # Open-set fallback: preserve original casing/title
        return c.title() if len(c) > 3 else c.upper()


def generate_char_ngrams(
    text: str,
    n_min: int = 2,
    n_max: int = 4
) -> List[str]:
    """Generate character n-grams from text within specified length range.

    Args:
        text: Input string.
        n_min: Minimum n-gram length (default 2).
        n_max: Maximum n-gram length (default 4).

    Returns:
        List of character n-gram strings.
    """
    if not text:
        return []

    # Pad with boundary markers for prefix/suffix sensitivity
    padded = f" {text.strip()} "
    length = len(padded)
    ngrams: List[str] = []

    for n in range(n_min, min(n_max + 1, length + 1)):
        for i in range(length - n + 1):
            ngrams.append(padded[i:i + n])

    return ngrams


def normalize_dataframe(
    df: pd.DataFrame,
    config: Optional[NormalizationConfig] = None
) -> pd.DataFrame:
    """Apply complete normalization to an entity DataFrame while preserving raw fields.

    Produces new standardized columns without altering original raw columns.

    Args:
        df: Input DataFrame containing entity_id, business_name, business_address, country.
        config: NormalizationConfig instance.

    Returns:
        Enriched DataFrame with normalized features.
    """
    cfg = config or NormalizationConfig()
    out = df.copy()

    # 1. Normalize Country
    out["country_normalized"] = out["country"].apply(normalize_country)

    # 2. Normalize Business Name
    name_results = [
        normalize_business_name(n, cfg)
        for n in out["business_name"]
    ]
    out["name_normalized"] = [r[0] for r in name_results]
    out["name_core"] = [r[1] for r in name_results]
    out["name_legal_suffix"] = [r[2] for r in name_results]
    out["name_tokens"] = out["name_normalized"].apply(lambda s: s.split())

    # 3. Normalize Business Address
    addr_results = [
        normalize_address(addr, c, cfg)
        for addr, c in zip(out["business_address"], out["country_normalized"])
    ]
    out["address_normalized"] = [r[0] for r in addr_results]
    out["address_postal_code"] = [r[1] for r in addr_results]
    out["address_house_number"] = [r[2] for r in addr_results]
    out["address_numeric_tokens"] = [r[3] for r in addr_results]
    out["address_is_empty"] = [r[4] for r in addr_results]
    out["address_tokens"] = out["address_normalized"].apply(lambda s: s.split())

    return out
