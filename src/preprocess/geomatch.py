"""Match Wikimedia Commons photos to Mérimée heritage records by GPS proximity.

Each Commons record carries the camera position; each Mérimée notice carries
the monument position (COORD, WGS84). A photo matched to a notice inherits
that notice's architectural history — SCLE (construction centuries), HIST
(free text, often naming French/German phases), DATE, AUTR — which becomes
our weak-label source for cultural-strata classification.

Matching: nearest Mérimée record within --radius meters (haversine), using a
degree-grid index so the pairing stays O(n) instead of n_photos x n_notices.

Usage:
    python src/preprocess/geomatch.py \
        --commons data/annotations/commons_alsace.jsonl \
        --merimee data/annotations/merimee_alsace.jsonl \
        --radius 60 \
        --out data/processed/matched.jsonl
"""

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

EARTH_R = 6_371_000.0  # meters


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_R * math.asin(math.sqrt(a))


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def merimee_coords(rec: dict) -> tuple[float, float] | None:
    coord = rec.get("COORD")
    if not coord:
        return None
    try:
        lat, lon = (float(x) for x in coord.split(","))
    except ValueError:
        return None
    # Sanity box for Alsace; the dump has a few swapped/garbage coordinates.
    if not (47.0 <= lat <= 49.5 and 6.5 <= lon <= 8.5):
        return None
    return lat, lon


class GridIndex:
    """Bucket points into cells of `cell_deg` degrees; query returns the
    candidates from the 3x3 neighborhood around a coordinate."""

    def __init__(self, cell_deg: float = 0.002):  # ~220 m in latitude
        self.cell_deg = cell_deg
        self.cells: dict[tuple[int, int], list] = defaultdict(list)

    def _key(self, lat: float, lon: float) -> tuple[int, int]:
        return (int(lat / self.cell_deg), int(lon / self.cell_deg))

    def add(self, lat: float, lon: float, item) -> None:
        self.cells[self._key(lat, lon)].append((lat, lon, item))

    def near(self, lat: float, lon: float):
        ki, kj = self._key(lat, lon)
        for i in range(ki - 1, ki + 2):
            for j in range(kj - 1, kj + 2):
                yield from self.cells.get((i, j), ())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--commons", required=True)
    ap.add_argument("--merimee", required=True)
    ap.add_argument("--radius", type=float, default=60.0, help="max match distance, meters")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    photos = load_jsonl(Path(args.commons))
    notices = load_jsonl(Path(args.merimee))

    index = GridIndex()
    n_indexed = 0
    for rec in notices:
        coords = merimee_coords(rec)
        if coords:
            index.add(*coords, rec)
            n_indexed += 1

    matched, no_gps, no_neighbor = [], 0, 0
    for photo in photos:
        gps = photo.get("gps")
        if not gps:
            no_gps += 1
            continue
        best, best_d = None, args.radius
        for lat, lon, rec in index.near(gps["lat"], gps["lon"]):
            d = haversine(gps["lat"], gps["lon"], lat, lon)
            if d <= best_d:
                best, best_d = rec, d
        if best is None:
            no_neighbor += 1
            continue
        matched.append(
            {
                **photo,
                "merimee_ref": best["REF"],
                "merimee_distance_m": round(best_d, 1),
                "merimee": {
                    k: best.get(k)
                    for k in ("TICO", "DENO", "COM", "SCLE", "SCLE2", "DATE", "AUTR", "HIST", "DESC")
                },
            }
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for rec in matched:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(
        f"{len(photos)} photos: {len(matched)} matched (<= {args.radius:.0f} m), "
        f"{no_gps} without GPS, {no_neighbor} with GPS but no notice in range; "
        f"{n_indexed}/{len(notices)} notices had usable coordinates",
        file=sys.stderr,
    )
    print(f"Wrote {len(matched)} records to {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
