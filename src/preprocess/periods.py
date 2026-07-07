"""Parse Mérimée SCLE century expressions into year ranges and political periods.

SCLE encodes construction campaigns as French century phrases, semicolon-
separated, e.g. "4e quart 13e siècle;14e siècle" or "limite 19e siècle 20e
siècle". We parse each phrase to a (start, end) year range, then intersect
it with the Alsace sovereignty timeline:

    -1648  germanic_hre    (Holy Roman Empire; Strasbourg itself to 1681)
    1648-  french_ancien   (French crown / Republic / Empire)
    1871-  german_reich    (annexed, Reichsland Elsass-Lothringen)
    1918-  french_interwar
    1940-  german_occupation
    1945-  french_modern

A building whose campaign range falls inside one period gets that label; a
range straddling a boundary gets all periods it touches. This is the weak
label for cultural-strata classification: "built under X" is not the same as
"looks X", and the gap between the two is exactly what RQ2 interrogates.

Usable as a library (parse_scle, periods_for_range) or as a CLI that
annotates a matched JSONL in place:

    python src/preprocess/periods.py \
        --in data/processed/matched.jsonl \
        --out data/processed/matched_periods.jsonl
"""

import argparse
import json
import re
import sys
from pathlib import Path

PERIODS = [
    ("germanic_hre", None, 1648),
    ("french_ancien", 1648, 1871),
    ("german_reich", 1871, 1918),
    ("french_interwar", 1918, 1940),
    ("german_occupation", 1940, 1945),
    ("french_modern", 1945, None),
]

_CENTURY = re.compile(r"(\d{1,2})e\s+si[eè]cle", re.I)
_QUART = re.compile(r"(\d)(?:er|e)\s+quart", re.I)
_MOITIE = re.compile(r"(\d)(?:re|ère|e)\s+moiti[eé]", re.I)
_MILIEU = re.compile(r"milieu", re.I)
_LIMITE = re.compile(r"limite", re.I)


def parse_phrase(phrase: str) -> tuple[int, int] | None:
    """One SCLE phrase -> (start_year, end_year), or None if unparseable."""
    centuries = [int(c) for c in _CENTURY.findall(phrase)]
    if not centuries:
        return None

    if _LIMITE.search(phrase) and len(centuries) >= 2:
        # "limite 19e siècle 20e siècle" = the turn between them, ±15 years.
        turn = (min(centuries)) * 100
        return turn - 15, turn + 15

    c = centuries[0]
    c0 = (c - 1) * 100 + 1  # 19e siècle -> 1801
    if m := _QUART.search(phrase):
        q = int(m.group(1))
        return c0 + (q - 1) * 25, c0 + q * 25 - 1
    if m := _MOITIE.search(phrase):
        h = int(m.group(1))
        return c0 + (h - 1) * 50, c0 + h * 50 - 1
    if _MILIEU.search(phrase):
        return c0 + 33, c0 + 66
    return c0, c0 + 99


def parse_scle(scle: str) -> list[tuple[int, int]]:
    """Full SCLE field -> list of campaign year ranges."""
    ranges = []
    for phrase in scle.split(";"):
        r = parse_phrase(phrase.strip())
        if r:
            ranges.append(r)
    return ranges


def periods_for_range(start: int, end: int) -> list[str]:
    """Political periods a year range touches."""
    out = []
    for name, p0, p1 in PERIODS:
        lo = p0 if p0 is not None else -10_000
        hi = p1 if p1 is not None else 10_000
        if start < hi and end >= lo:
            out.append(name)
    return out


def annotate(rec: dict) -> dict:
    scle_parts = [rec["merimee"].get("SCLE"), rec["merimee"].get("SCLE2")]
    ranges = []
    for part in scle_parts:
        if part:
            ranges.extend(parse_scle(part))
    periods = sorted({p for r in ranges for p in periods_for_range(*r)})
    rec["campaign_ranges"] = ranges
    rec["periods"] = periods
    return rec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    n, labeled = 0, 0
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.inp, encoding="utf-8") as f, out.open("w", encoding="utf-8") as g:
        for line in f:
            rec = annotate(json.loads(line))
            g.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
            if rec["periods"]:
                labeled += 1
    print(f"{labeled}/{n} records got period labels -> {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
