"""Download training-resolution images for geo-matched records.

Reads one or more matched JSONL files, dedupes by pageid, and fetches each
image as a Commons thumbnail (Special:FilePath?width=N) rather than the
original — 1200 px is plenty for a CNN and an order of magnitude smaller.
Files land in DEST/<pageid>.jpg so the pageid is the join key back to
metadata; a manifest JSONL records pageid -> local path + source file.

Usage:
    python src/collect/download_matched.py \
        data/processed/matched_periods.jsonl \
        data/processed/matched_neustadt_periods.jsonl \
        --dest data/raw/images --width 1200
"""

import argparse
import json
import sys
import time
from pathlib import Path
from urllib.parse import quote

import requests
from tqdm import tqdm

USER_AGENT = (
    "Palimpseste/0.1 (https://github.com/Sulla-Soglia/palimpseste; "
    "chenying@trelune.dev) python-requests"
)


def thumb_url(title: str, width: int) -> str:
    name = title.removeprefix("File:")
    return f"https://commons.wikimedia.org/wiki/Special:FilePath/{quote(name)}?width={width}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("inputs", nargs="+", help="matched JSONL files")
    ap.add_argument("--dest", default="data/raw/images")
    ap.add_argument("--width", type=int, default=1200)
    args = ap.parse_args()

    records: dict[int, dict] = {}
    for path in args.inputs:
        with open(path, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                records.setdefault(rec["pageid"], {**rec, "source_file": path})

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT

    manifest, fetched, failed = [], 0, 0
    for pageid, rec in tqdm(records.items(), desc="download", unit="img"):
        path = dest / f"{pageid}.jpg"
        if not path.exists():
            try:
                r = session.get(thumb_url(rec["title"], args.width), timeout=60)
                r.raise_for_status()
                path.write_bytes(r.content)
                fetched += 1
                time.sleep(0.3)
            except requests.RequestException as e:
                failed += 1
                print(f"failed {rec['title']}: {e}", file=sys.stderr)
                continue
        manifest.append(
            {"pageid": pageid, "local_path": str(path), "source_file": rec["source_file"]}
        )

    mpath = dest / "manifest.jsonl"
    with mpath.open("w", encoding="utf-8") as f:
        for m in manifest:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    print(
        f"{len(records)} unique images: {fetched} fetched, {failed} failed, "
        f"{len(manifest)} in manifest -> {mpath}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
