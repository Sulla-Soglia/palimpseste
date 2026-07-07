# Phase 1 data pipeline

```
Wikimedia Commons ──wikimedia.py──> commons_*.jsonl ─┐
                                                     ├─geomatch.py─> matched.jsonl ─periods.py─> matched_periods.jsonl
Mérimée dump ────────merimee.py───> merimee_*.jsonl ─┘
```

## Steps

1. **Collect Commons metadata** — walk a category tree, keep free-licensed
   images ≥800px, record GPS when present.

   ```
   python src/collect/wikimedia.py --category "Category:Architecture of Alsace" \
       --depth 2 --out data/annotations/commons_alsace.jsonl
   ```

2. **Collect Mérimée notices** — filter the Ministry of Culture dump to
   departments 67/68.

   ```
   python src/collect/merimee.py --departments 67 68 \
       --out data/annotations/merimee_alsace.jsonl
   ```

3. **GPS matching** — nearest notice within 60 m (haversine, grid index).
   A matched photo inherits the notice's SCLE/HIST/DATE/AUTR.

   ```
   python src/preprocess/geomatch.py --commons ... --merimee ... --radius 60 \
       --out data/processed/matched.jsonl
   ```

4. **Period labels** — parse SCLE century phrases ("4e quart 19e siècle")
   into year ranges, intersect with the sovereignty timeline:
   germanic_hre (→1648) · french_ancien (→1871) · german_reich (→1918) ·
   french_interwar (→1940) · german_occupation (→1945) · french_modern.

   ```
   python src/preprocess/periods.py --in matched.jsonl --out matched_periods.jsonl
   ```

## Run log (2026-07-07)

| Stage | Result |
|---|---|
| Mérimée 67+68 | 1,439 notices, 1,349 with usable WGS84 coords |
| Commons "Architecture of Alsace" depth 2 | 914 files → 876 pass filters |
| GPS match ≤60 m | 246 photos ↔ 124 monuments, median 26 m; 429 photos had no GPS |
| Period labels | 233/246 labeled |

Period distribution (multi-label): french_ancien 180 · germanic_hre 106 ·
german_reich 45 · french_interwar 14 · french_modern 8 · german_occupation 5.

## Known issues / decisions

- **German-period deficit.** The heritage-protected corpus skews medieval and
  18th-century; Reichsland-era architecture (1871–1918, the Neustadt) is
  underrepresented in Mérimée. Mitigation: collect
  `Category:Neustadt (Strasbourg)` separately — those photos can take a
  `german_reich` label from the category itself, not only from Mérimée.
- **"Built under X" ≠ "looks X".** Period labels are weak labels for style.
  The gap between political period and stylistic reading is not noise — it is
  the RQ2 object (a French-period building in Rhenish half-timber, a
  German-period building in Parisian Beaux-Arts).
- **Nearest-notice mislabeling.** A photo can match a *different* nearby
  monument (e.g. a convent photo matched to a Renaissance well 59 m away).
  Median distance is 26 m so most matches are right; tighten radius or add
  title-similarity checks if annotation review shows too much noise.
- **GPS-less photos (429/876).** Many sit in building-specific Commons
  categories; a category→coordinate lookup (via Wikidata P625) could rescue
  them. Deferred.
- **HIST keyword labeling is too sparse** at this scale (85/246 records have
  HIST, only 4 mention a German phase). SCLE periods are the primary weak
  label; HIST stays as qualitative evidence for the paper.
