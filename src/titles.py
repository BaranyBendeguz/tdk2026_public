# Officer title normalization.

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

ROLE_PRIORITY = [
    "ceo", "cfo", "president", "coo", "cio", "cto", "cao", "general_counsel",
    "evp", "svp", "vp", "chairman", "director", "ten_pct_owner",
]
ROLE_RANK = {
    "ceo": 1, "cfo": 2, "president": 3, "coo": 3,
    "cio": 4, "cto": 4, "cao": 4, "general_counsel": 4,
    "evp": 5, "svp": 5, "vp": 5,
    "chairman": 6, "director": 6, "ten_pct_owner": 7,
}
RANK_OTHER_OFFICER = 5
RANK_OTHER = 8

# Misspelled or truncated tokens observed in the real RPTOWNER_TITLE values
# (frequency-ranked near matches of the key words), mapped to the full word.
TYPO_TOKENS = {
    "chief": "cheif chie chiel cjhief chielf chf",
    "officer": "office offcr offic offier officr oficer oficr offficer ofcer offfice offcier offiicer officier ofc ofcr offr",
    "executive": "excutive executve exective executiv execut exectuive exeutive exectutive excecutive executvie execuitve eecutive exeuctive execitive exec exe exc exctv exec",
    "financial": "financ finanical fiancial financal finacial finanacial fianancial financi finanicial financl financail fincial fin finl fincl",
    "operating": "operat operatio operati opperating oprtng operatns oprting operatg opertaing operting opertating opertations opertns operatons operations oper ops",
    "president": "presiden presdient preisdent preside preident presient presdent presidnet prseident presid prsident pesident presidelnt pres presdt prez",
    "senior": "senor senioe sr snr",
    "chairman": "chariman chaiman chairwoman chairmanm chairmain chairmen chairmn chiarman chairmand charman chairm chairma chair chairperson chmn chrmn chm chrm",
    "director": "directors directo dir drctr",
    "accounting": "accting accntng accouting acccounting accnting accounti acconting accountg accountng accouning acctg acct accounting",
    "technology": "technol technolog technlgy techology techlgy techonology technolo technolgy technololgy tchnlgy tech",
    "information": "inforrmation info",
    "general": "generral gerneral genera gneral gen genl",
    "counsel": "couns counse counsl cousel consl counel councel counl consel cousnel cnsl",
    "legal": "legl",
    "principal": "principle princpl prncpl prinpl prinicipal prin",
    "investment": "investments invstmt investmt invstmnt investme investmets",
    "vice": "vic",
}
TYPO_MAP = {bad: good for good, bads in TYPO_TOKENS.items() for bad in bads.split()}
# Glued tokens: split them into words.
GLUED = {
    "vicepresident": "vice president", "executivevp": "executive vp", "seniorvp": "senior vp",
    "vicechairman": "vice chairman", "execchairman": "executive chairman",
    "chieftechnology": "chief technology", "gencounsel": "general counsel",
    "interimcfo": "interim cfo", "interimceo": "interim ceo", "andtreasurer": "and treasurer",
    "srvp": "senior vp", "exvp": "executive vp", "chairmanceo": "chairman ceo",
    "presidentceo": "president ceo", "ceopresident": "ceo president",
}

_TEN_PCT = re.compile(r"\b(?:10|ten)\s*(?:%|percent|pct|per\s+cent)", re.IGNORECASE)
_DOTS = re.compile(r"(?<=\w)\.(?=\w)|\.")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")
# Phrase fixes applied after punctuation removal: spaced initials and board abbreviations.
PHRASES = [
    (re.compile(r"\bv p\b"), "vp"),
    (re.compile(r"\bc ([efot]) o\b"), r"c\1o"),
    (re.compile(r"\bch(?:airman)? of (?:the )?(?:bd|board)\b"), "chairman"),
]

ROLE_PATTERNS = {
    "ceo": r"\bceo\b|\bchief executive\b|\bprincipal executive\b|\bpeo\b",
    "cfo": r"\bcfo\b|\bchief financial\b|\bchief finance\b|\bprincipal financial\b|\bpfo\b",
    "coo": r"\bcoo\b|\bchief operating\b",
    "cio": r"\bcio\b|\bchief information\b|\bchief investment\b",
    "cto": r"\bcto\b|\bchief technology\b|\bchief technical\b",
    "cao": r"\bcao\b|\bchief accounting\b|\bprincipal accounting\b|\bpao\b",
    "general_counsel": r"\bgeneral counsel\b|\bgc\b|\bchief legal\b",
    "evp": r"\bexecutive vice president\b|\bevp\b|\bsevp\b|\bexecutive vp\b",
    "svp": r"\bsenior vice president\b|\bsvp\b|\bsenior vp\b",
    "vp": r"\bvice president\b|\bvp\b|\bavp\b",
    "ten_pct_owner": r"\btenpctowner\b",
}
ROLE_RE = {k: re.compile(v) for k, v in ROLE_PATTERNS.items()}
_VICE_PRES = re.compile(r"\b(?:vice president|vp|evp|svp|avp)\b")
_PRESIDENT = re.compile(r"\bpresident\b")
_VICE_CHAIR = re.compile(r"\bvice chairman\b")
_CHAIRMAN = re.compile(r"\bchairman\b")
# A board seat, not a functional title such as "Managing Director" or "Director of Sales".
_BOARD_DIRECTOR = re.compile(
    r"(?:^|\band\b|\b(?:independent|lead|outside|non executive|nonexecutive|board|presiding|vice chairman)\s+)"
    r"directors?\b(?!\s+of\b)"
    r"|\bboard member\b|\bmember of (?:the )?board\b|\bboard of directors\b"
)


def normalize_title(raw: str | None) -> str:
    """Lowercase, strip punctuation and whitespace, expand typos. Returns '' for empty."""
    if not raw:
        return ""
    s = unicodedata.normalize("NFKC", str(raw)).lower()
    s = _TEN_PCT.sub(" tenpctowner ", s)
    s = s.replace("&", " and ")
    s = _DOTS.sub("", s)  # C.E.O. -> ceo, Sr. -> sr
    s = _NON_ALNUM.sub(" ", s).strip()
    words = []
    for w in s.split():
        w = GLUED.get(w, w)
        for part in w.split():
            words.append(TYPO_MAP.get(part, part))
    s = " ".join(words)
    for rx, repl in PHRASES:
        s = rx.sub(repl, s)
    # "owner" alone after the percentage token is redundant
    return s.replace("tenpctowner owner", "tenpctowner")


@lru_cache(maxsize=None)
def title_roles(raw: str | None) -> tuple[str, ...]:
    """All canonical roles named by a raw title, most senior first."""
    s = normalize_title(raw)
    if not s:
        return ()
    found = {role for role, rx in ROLE_RE.items() if rx.search(s)}
    if _PRESIDENT.search(_VICE_PRES.sub(" ", s)):
        found.add("president")
    if _CHAIRMAN.search(_VICE_CHAIR.sub(" ", s)):
        found.add("chairman")
    if _BOARD_DIRECTOR.search(s) or _VICE_CHAIR.search(s):
        found.add("director")
    return tuple(r for r in ROLE_PRIORITY if r in found)


def classify(raw_title: str | None, is_officer: int, is_director: int, is_ten_pct: int) -> tuple[str, int]:
    """Return (officer_title_clean, role_rank) for one reporting owner."""
    roles = title_roles(raw_title)
    if roles:
        best = min(roles, key=lambda r: (ROLE_RANK[r], ROLE_PRIORITY.index(r)))
        return best, ROLE_RANK[best]
    if is_officer:
        return "other", RANK_OTHER_OFFICER
    if is_director:
        return "director", ROLE_RANK["director"]
    if is_ten_pct:
        return "ten_pct_owner", ROLE_RANK["ten_pct_owner"]
    return "other", RANK_OTHER
