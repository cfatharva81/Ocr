"""Value normalizers: currency, dates, phone, SSN, ZIP, state codes, ints, booleans.

All functions accept messy strings (including common OCR confusions such as
O/0, l/1, S/$) and return None when nothing usable is found, never raise.
"""
from __future__ import annotations

import re
from datetime import date, datetime

from rapidfuzz import fuzz, process

US_STATES = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California",
    "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware", "DC": "District of Columbia",
    "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois",
    "IN": "Indiana", "IA": "Iowa", "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana",
    "ME": "Maine", "MD": "Maryland", "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota",
    "MS": "Mississippi", "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada",
    "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York",
    "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma", "OR": "Oregon",
    "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina", "SD": "South Dakota",
    "TN": "Tennessee", "TX": "Texas", "UT": "Utah", "VT": "Vermont", "VA": "Virginia",
    "WA": "Washington", "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming",
    "PR": "Puerto Rico", "GU": "Guam", "VI": "Virgin Islands",
}
_NAME_TO_CODE = {v.lower(): k for k, v in US_STATES.items()}

# OCR confusions that only make sense inside numeric tokens.
_DIGIT_FIX = str.maketrans({"O": "0", "o": "0", "D": "0", "Q": "0", "l": "1", "I": "1", "|": "1",
                            "i": "1", "Z": "2", "z": "2", "B": "8", "g": "9", "b": "6"})

TRUE_WORDS = {"yes", "y", "true", "t", "1", "x", "checked", "on", "selected",
              "✓", "✔", "☒", "☑", "■"}
FALSE_WORDS = {"no", "n", "false", "f", "0", "unchecked", "off", "none", ""}


def clean_text(s: str | None) -> str | None:
    """Collapse whitespace and trim separator punctuation left over from labels."""
    if s is None:
        return None
    s = re.sub(r"\s+", " ", str(s)).strip(" :;|_\t")
    return s or None


def _numeric_token(s: str) -> str:
    """Fix OCR letter/digit confusions in a token that is mostly digits."""
    digits = sum(ch.isdigit() for ch in s)
    letters = sum(ch.isalpha() for ch in s)
    return s.translate(_DIGIT_FIX) if digits and digits >= letters else s


def currency(s) -> float | None:
    """'$4,750.01' -> 4750.01, '(1,200.00)' -> -1200.0, 'S6,500.00' (OCR) -> 6500.0."""
    if s is None:
        return None
    if isinstance(s, (int, float)):
        return float(s)
    t = str(s).strip().replace(" ", "")
    neg = (t.startswith("(") and t.endswith(")")) or t.startswith("-")
    t = t.strip("()-")
    # Currency symbol, including OCR reading '$' as 'S'.
    t = re.sub(r"^(?:USD|US)?[\$S](?=[\dOoIl])", "", t)
    t = _numeric_token(t)
    m = re.search(r"\d[\d,.]*", t)
    if not m:
        return None
    num = m.group(0).rstrip(".,")
    # Several dots: the last one is the decimal point ("6.500.00" -> 6500.00).
    if num.count(".") > 1:
        head, _, tail = num.rpartition(".")
        num = head.replace(".", "") + "." + tail
    # A comma followed by exactly two digits at the end is a misread decimal point.
    if "." not in num and re.search(r",\d{2}$", num):
        num = num[:-3] + "." + num[-2:]
    num = num.replace(",", "")
    try:
        v = float(num)
    except ValueError:
        return None
    return -v if neg else v


_DATE_FORMATS = ["%m/%d/%Y", "%m-%d-%Y", "%m.%d.%Y", "%m/%d/%y", "%m-%d-%y", "%Y-%m-%d",
                 "%Y/%m/%d", "%B %d, %Y", "%b %d, %Y", "%B %d %Y", "%b %d %Y", "%d %B %Y",
                 "%m%d%Y"]


def to_date(s) -> date | None:
    """Parse US-style dates to a date; returns None when unparseable."""
    if s is None:
        return None
    if isinstance(s, datetime):
        return s.date()
    if isinstance(s, date):
        return s
    t = clean_text(str(s)) or ""
    # Fix OCR confusions inside the numeric part only.
    t = re.sub(r"[\dOolI|]{1,4}(?:[/\-.][\dOolI|]{1,4}){2}", lambda m: _numeric_token(m.group(0)), t)
    m = re.search(r"\d{1,4}[/\-.]\d{1,2}[/\-.]\d{2,4}", t)
    candidates = [m.group(0)] if m else []
    candidates.append(t)
    for c in candidates:
        c = re.sub(r"\s*([/\-.])\s*", r"\1", c)
        for fmt in _DATE_FORMATS:
            try:
                d = datetime.strptime(c, fmt).date()
            except ValueError:
                continue
            if d.year < 100:
                d = d.replace(year=d.year + (1900 if d.year > 30 else 2000))
            return d
    return None


def iso_date(s) -> str | None:
    d = to_date(s)
    return d.isoformat() if d else None


def phone(s) -> str | None:
    """Digits only; drops a leading US country code. Needs >= 7 digits."""
    if s is None:
        return None
    digits = re.sub(r"\D", "", _numeric_token(str(s)))
    if len(digits) == 11 and digits.startswith("1"):
        digits = digits[1:]
    return digits if len(digits) >= 7 else None


def ssn(s) -> str | None:
    """'123 45 6789' / '123456789' -> '123-45-6789'. None unless exactly 9 digits."""
    if s is None:
        return None
    t = _numeric_token(str(s))
    m = re.search(r"(\d{3})\D{0,2}(\d{2})\D{0,2}(\d{4})", t)
    if not m:
        return None
    return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"


def zip_code(s) -> str | None:
    if s is None:
        return None
    t = _numeric_token(str(s))
    m = re.search(r"\b(\d{5})(?:\s*-\s*(\d{4}))?\b", t)
    if not m:
        return None
    return f"{m.group(1)}-{m.group(2)}" if m.group(2) else m.group(1)


def state_code(s) -> str | None:
    """'IL', 'il', 'Illinois', 'Ilinois' -> 'IL'."""
    t = clean_text(s)
    if not t:
        return None
    up = re.sub(r"[^A-Za-z ]", "", t).strip().upper()
    if up in US_STATES:
        return up
    low = up.lower()
    if low in _NAME_TO_CODE:
        return _NAME_TO_CODE[low]
    # OCR: '1L' -> 'IL' etc.
    alt = t.upper().replace("1", "I").replace("0", "O").replace("5", "S")
    alt = re.sub(r"[^A-Z]", "", alt)
    if alt in US_STATES:
        return alt
    hit = process.extractOne(low, list(_NAME_TO_CODE), scorer=fuzz.ratio, score_cutoff=85)
    return _NAME_TO_CODE[hit[0]] if hit else None


def integer(s) -> int | None:
    if s is None:
        return None
    if isinstance(s, bool):
        return int(s)
    if isinstance(s, int):
        return s
    if isinstance(s, float):
        return int(round(s))
    t = str(s).strip()
    # a lone OCR confusion in a numeric field ("O" months, "l" unit)
    if t and len(t) <= 2 and all(ch in "OoDQlI|" for ch in t):
        t = t.translate(_DIGIT_FIX)
    t = _numeric_token(t)
    m = re.search(r"-?\d[\d,]*", t)
    if not m:
        return None
    try:
        return int(m.group(0).replace(",", ""))
    except ValueError:
        return None


def boolean(s) -> bool | None:
    """checked / 'Yes' / 'X' / True -> True; 'No' / 'Off' / unchecked -> False."""
    if s is None:
        return None
    if isinstance(s, bool):
        return s
    if isinstance(s, (int, float)):
        return bool(s)
    t = str(s).strip().lower().strip("[]()")
    if t in TRUE_WORDS:
        return True
    if t in FALSE_WORDS:
        return False
    return None


def account_number(s) -> str | None:
    """Keep alphanumerics and dashes; fix O/0 and l/1 when mostly digits."""
    t = clean_text(s)
    if not t:
        return None
    t = t.replace(" ", "")
    return _numeric_token(t) if re.search(r"\d", t) else t


def email(s) -> str | None:
    t = clean_text(s)
    if not t:
        return None
    t = t.replace(" ", "").replace(",", ".").lower()
    t = re.sub(r"\(at\)|\[at\]", "@", t)
    m = re.search(r"[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}", t)
    return m.group(0) if m else t


def comparable(v) -> str:
    """Canonical string for comparing values (benchmark/tests)."""
    if v is None:
        return ""
    if isinstance(v, bool):
        return str(v).lower()
    if isinstance(v, float):
        return f"{v:.2f}"
    return re.sub(r"\s+", " ", str(v)).strip().casefold()


# ------------------------------------------------------------------ masking

_SSN_RE = re.compile(r"\b(\d{3})[- ]?(\d{2})[- ]?(\d{4})\b")
_LONG_NUM_RE = re.compile(r"\b([A-Za-z]{0,3}-?)(\d[\d\- ]{4,}\d)\b")


def mask_ssn(s: str) -> str:
    return _SSN_RE.sub(lambda m: f"***-**-{m.group(3)}", s)


def mask_account(s: str) -> str:
    def _m(m):
        digits = re.sub(r"\D", "", m.group(2))
        return m.group(1) + "*" * max(0, len(digits) - 4) + digits[-4:]
    return _LONG_NUM_RE.sub(_m, s)


def mask_sensitive(s: str) -> str:
    return mask_account(mask_ssn(s))
