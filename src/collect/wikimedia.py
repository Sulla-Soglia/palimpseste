"""Collect Alsace architecture images and metadata from Wikimedia Commons.

Walks a Commons category tree (e.g. "Category:Architecture of Alsace"),
gathers file metadata (URL, size, license, GPS, author, date) and writes it
as JSONL. Optionally downloads the image files themselves.

Usage:
    python src/collect/wikimedia.py \
        --category "Category:Architecture of Alsace" \
        --depth 2 --limit 100 \
        --out data/annotations/commons_alsace.jsonl

    # add --download data/raw to also fetch the images
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import requests
from tqdm import tqdm

API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = (
    "Palimpseste/0.1 (https://github.com/Sulla-Soglia/palimpseste; "
    "chenying@trelune.dev) python-requests"
)

# Licenses acceptable for ML training use. Everything on Commons is nominally
# free, but we keep the license string so the dataset card can report exact terms.
FREE_LICENSE_RE = re.compile(r"(cc[ -]|creative commons|public domain|pd-|fal|gfdl)", re.I)


def make_session() -> requests.Session:
    s = requests.Session()
    s.headers["User-Agent"] = USER_AGENT
    return s


def api_get(session: requests.Session, params: dict, retries: int = 4) -> dict:
    params = {"action": "query", "format": "json", **params}
    for attempt in range(retries):
        try:
            r = session.get(API, params=params, timeout=30)
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, ValueError):
            if attempt == retries - 1:
                raise
            time.sleep(2**attempt)
    raise RuntimeError("unreachable")


def iter_category_members(session, category: str, member_type: str):
    """Yield member titles of one category ('file' or 'subcat'), following continuation."""
    cont = {}
    while True:
        data = api_get(
            session,
            {
                "list": "categorymembers",
                "cmtitle": category,
                "cmtype": member_type,
                "cmlimit": "500",
                **cont,
            },
        )
        for m in data.get("query", {}).get("categorymembers", []):
            yield m["title"]
        cont = data.get("continue")
        if not cont:
            return


def collect_file_titles(session, root: str, depth: int, limit: int | None) -> list[str]:
    """BFS the category tree down to `depth`, collecting file titles."""
    titles: list[str] = []
    seen_cats = {root}
    frontier = [root]
    for level in range(depth + 1):
        next_frontier: list[str] = []
        for cat in frontier:
            for title in iter_category_members(session, cat, "file"):
                titles.append(title)
                if limit and len(titles) >= limit:
                    return titles
            if level < depth:
                for sub in iter_category_members(session, cat, "subcat"):
                    if sub not in seen_cats:
                        seen_cats.add(sub)
                        next_frontier.append(sub)
        frontier = next_frontier
        if not frontier:
            break
    return titles


def _meta_value(extmeta: dict, key: str) -> str | None:
    v = extmeta.get(key, {}).get("value")
    if isinstance(v, str):
        # Commons wraps some values in HTML; strip tags for storage.
        return re.sub(r"<[^>]+>", "", v).strip() or None
    return v


def fetch_imageinfo(session, titles: list[str]) -> list[dict]:
    """Fetch imageinfo for up to 50 titles per request."""
    records = []
    for i in range(0, len(titles), 50):
        batch = titles[i : i + 50]
        data = api_get(
            session,
            {
                "titles": "|".join(batch),
                "prop": "imageinfo",
                "iiprop": "url|size|mime|extmetadata",
            },
        )
        for page in data.get("query", {}).get("pages", {}).values():
            infos = page.get("imageinfo")
            if not infos:
                continue
            info = infos[0]
            ext = info.get("extmetadata", {})
            lat = _meta_value(ext, "GPSLatitude")
            lon = _meta_value(ext, "GPSLongitude")
            records.append(
                {
                    "title": page.get("title"),
                    "pageid": page.get("pageid"),
                    "url": info.get("url"),
                    "descriptionurl": info.get("descriptionurl"),
                    "width": info.get("width"),
                    "height": info.get("height"),
                    "mime": info.get("mime"),
                    "license": _meta_value(ext, "LicenseShortName"),
                    "usage_terms": _meta_value(ext, "UsageTerms"),
                    "artist": _meta_value(ext, "Artist"),
                    "date_original": _meta_value(ext, "DateTimeOriginal"),
                    "gps": {"lat": float(lat), "lon": float(lon)} if lat and lon else None,
                    "categories": _meta_value(ext, "Categories"),
                }
            )
    return records


def sanitize_filename(title: str) -> str:
    name = title.removeprefix("File:")
    return re.sub(r'[\\/:*?"<>|]', "_", name)


def download_images(session, records: list[dict], dest: Path) -> int:
    dest.mkdir(parents=True, exist_ok=True)
    n = 0
    for rec in tqdm(records, desc="download", unit="img"):
        if not rec.get("url"):
            continue
        path = dest / sanitize_filename(rec["title"])
        rec["local_path"] = str(path)
        if path.exists():
            continue
        r = session.get(rec["url"], timeout=60)
        r.raise_for_status()
        path.write_bytes(r.content)
        n += 1
        time.sleep(0.5)  # stay polite; Commons rate guidance
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--category", default="Category:Architecture of Alsace")
    ap.add_argument("--depth", type=int, default=1, help="subcategory recursion depth")
    ap.add_argument("--limit", type=int, default=None, help="max number of files")
    ap.add_argument("--min-width", type=int, default=800)
    ap.add_argument("--out", required=True, help="output metadata JSONL path")
    ap.add_argument("--download", metavar="DIR", default=None, help="also download images to DIR")
    args = ap.parse_args()

    session = make_session()

    print(f"Walking {args.category} (depth={args.depth}) ...", file=sys.stderr)
    titles = collect_file_titles(session, args.category, args.depth, args.limit)
    print(f"Found {len(titles)} files, fetching metadata ...", file=sys.stderr)

    records = fetch_imageinfo(session, titles)
    kept = [
        r
        for r in records
        if (r["width"] or 0) >= args.min_width
        and r["license"]
        and FREE_LICENSE_RE.search(r["license"] or "")
        and (r["mime"] or "").startswith("image/")
    ]
    print(
        f"{len(kept)}/{len(records)} pass filters (min_width={args.min_width}, free license)",
        file=sys.stderr,
    )

    if args.download:
        n = download_images(session, kept, Path(args.download))
        print(f"Downloaded {n} new images to {args.download}", file=sys.stderr)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for rec in kept:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Wrote {len(kept)} records to {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
