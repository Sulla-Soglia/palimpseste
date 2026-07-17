"""Assemble the Phase 2 training dataset from matched+labeled records.

Merges the Alsace and Neustadt matched_periods files, keeps records whose
image was actually downloaded, dedups by pageid, and splits train/val/test
BY MONUMENT (merimee_ref), never by photo.

Why group-wise splitting matters: the same monument often has several photos
(different angles, years, photographers). If photos of one cathedral land in
both train and test, the model can score well by recognizing *that building*
instead of learning what a period looks like. Grouping by merimee_ref means
the test set contains only monuments the model has never seen.

Usage:
    python src/preprocess/make_dataset.py \
        --matched data/processed/matched_periods.jsonl \
                  data/processed/matched_neustadt_periods.jsonl \
        --images data/raw/images \
        --out data/processed/dataset.jsonl
"""

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

PERIODS = [
    "germanic_hre",
    "french_ancien",
    "german_reich",
    "french_interwar",
    "german_occupation",
    "french_modern",
]

SPLITS = {"train": 0.70, "val": 0.15, "test": 0.15}


def load_records(paths):
    records = {}
    for path in paths:
        with open(path, encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                # dedup across files: the same photo can match in both runs
                records.setdefault(rec["pageid"], rec)
    return list(records.values())


def assign_splits(records, seed):
    """Stratified group split: assign each monument (merimee_ref) to a split.

    Plain random grouping fails here: rare labels (german_occupation,
    french_modern) live on a handful of monuments, so a random cut can drop
    a whole class from val/test. Instead we use greedy iterative
    stratification: process monuments starting with those carrying the
    rarest labels, and put each one into the split whose deficit for that
    label (target share minus what it already has) is largest. Ties break
    by which split is furthest below its overall photo quota. Deterministic
    for a fixed seed, so the split is reproducible across sessions.
    """
    # per-monument label counts and photo counts
    photos = Counter()
    labels = {}
    for rec in records:
        ref = rec["merimee_ref"]
        photos[ref] += 1
        lab = labels.setdefault(ref, Counter())
        for p in rec["periods"]:
            lab[p] += 1

    total_label = Counter()
    for lab in labels.values():
        total_label.update(lab)
    total_photos = sum(photos.values())

    rarity = {p: total_label[p] for p in PERIODS}
    refs = sorted(labels)
    random.Random(seed).shuffle(refs)
    # rarest-label monuments first, big monuments before small ones
    refs.sort(key=lambda r: (min(rarity[p] for p in labels[r]), -photos[r]))

    have_label = {s: Counter() for s in SPLITS}
    have_photos = Counter()
    out = {}
    for ref in refs:
        rarest = min(labels[ref], key=lambda p: rarity[p])
        def deficit(s):
            want = total_label[rarest] * SPLITS[s]
            return want - have_label[s][rarest]
        def photo_deficit(s):
            return total_photos * SPLITS[s] - have_photos[s]
        split = max(SPLITS, key=lambda s: (deficit(s), photo_deficit(s)))
        out[ref] = split
        have_label[split].update(labels[ref])
        have_photos[split] += photos[ref]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matched", nargs="+", required=True)
    ap.add_argument("--images", default="data/raw/images")
    ap.add_argument("--out", default="data/processed/dataset.jsonl")
    ap.add_argument("--seed", type=int, default=67)
    args = ap.parse_args()

    records = load_records(args.matched)
    print(f"loaded {len(records)} unique photos from {len(args.matched)} files")

    image_dir = Path(args.images)
    usable = []
    dropped_no_image = dropped_no_label = 0
    for rec in records:
        if not rec.get("periods"):
            dropped_no_label += 1
            continue
        img = image_dir / f"{rec['pageid']}.jpg"
        if not img.exists():
            dropped_no_image += 1
            continue
        usable.append(rec)
    print(f"usable: {len(usable)}  (dropped {dropped_no_label} unlabeled, "
          f"{dropped_no_image} without a local image)")

    split_of = assign_splits(usable, args.seed)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    split_counts = Counter()
    label_counts = {s: Counter() for s in SPLITS}
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in usable:
            split = split_of[rec["merimee_ref"]]
            row = {
                "pageid": rec["pageid"],
                "image": f"{rec['pageid']}.jpg",
                "merimee_ref": rec["merimee_ref"],
                "title": rec["title"],
                "commune": (rec.get("merimee") or {}).get("COM"),
                "periods": rec["periods"],
                "split": split,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            split_counts[split] += 1
            for p in rec["periods"]:
                label_counts[split][p] += 1

    n_refs = Counter(split_of.values())
    print(f"\nmonuments: train {n_refs['train']} / val {n_refs['val']} / test {n_refs['test']}")
    print(f"photos:    train {split_counts['train']} / val {split_counts['val']} / test {split_counts['test']}")
    print("\nlabel counts per split (multi-label):")
    for p in PERIODS:
        print(f"  {p:20s} " + "  ".join(f"{s}:{label_counts[s][p]:4d}" for s in SPLITS))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
