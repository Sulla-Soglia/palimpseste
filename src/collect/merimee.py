"""Fetch French architectural heritage metadata from the Mérimée open-data dump.

The POP production API (api.pop.culture.gouv.fr, "POPv2") no longer exposes the
Elasticsearch search routes of the old betagouv/pop codebase — only per-record
GET /notices/merimee/{REF}. Instead of scraping, we use the Ministry of
Culture's full public dump on OVH S3:

    https://ministere-culture.s3.sbg.io.cloud.ovh.net/POP/merimee.csv  (~100 MB)

This script downloads the dump once (cached under data/raw/) and filters it to
the requested departments (67 = Bas-Rhin, 68 = Haut-Rhin), writing the
architectural-history core fields as JSONL.

Usage:
    python src/collect/merimee.py --departments 67 68 \
        --out data/annotations/merimee_alsace.jsonl
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import requests
from tqdm import tqdm

DUMP_URL = "https://ministere-culture.s3.sbg.io.cloud.ovh.net/POP/merimee.csv"
USER_AGENT = (
    "Palimpseste/0.1 (https://github.com/Sulla-Soglia/palimpseste; "
    "chenying@trelune.dev) python-requests"
)

# The dump is pipe-delimited with long French column names (not the ES field
# codes of the old API). Map them to the short codes used in the literature.
# We keep the architectural-history core and drop administrative boilerplate.
FIELD_MAP = {
    "REF": "Reference",
    "TICO": "Titre_editorial_de_la_notice",
    "DENO": "Denomination_de_l_edifice",
    "COM": "Commune_forme_editoriale",
    "DPT": "Departement_format_numerique",
    "ADRS": "Adresse_forme_editoriale",
    "SCLE": "Siecle_de_la_campagne_principale_de_construction",
    "SCLE2": "Siecle_de_campagne_secondaire_de_construction",
    "DATE": "Datation_de_l_edifice",
    "AUTR": "Auteur_de_l_edifice",
    "HIST": "Historique",  # free text, often mentions French/German phases
    "DESC": "Description_de_l_edifice",
    "MATR": "Materiaux_du_gros_oeuvre",
    "PROT": "Nature_de_la_protection",
    "STAT": "Statut_juridique_de_l_edifice",
    "COORD": "coordonnees_au_format_WGS84",
}
DPT_COLUMN = "Departement_format_numerique"


def download_dump(dest: Path, retries: int = 8) -> Path:
    """Download the dump with resume support, alternating direct/proxy on failure.

    The route to OVH S3 drops TLS connections intermittently from some
    networks, so every retry resumes from the current .part size via a
    Range request instead of starting over.
    """
    if dest.exists() and dest.stat().st_size > 0:
        print(f"Using cached dump {dest} ({dest.stat().st_size / 1e6:.0f} MB)", file=sys.stderr)
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")

    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    session.trust_env = False  # start direct; flip to system proxy on failure

    total = None
    for attempt in range(retries):
        offset = tmp.stat().st_size if tmp.exists() else 0
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        try:
            with session.get(DUMP_URL, stream=True, timeout=60, headers=headers) as r:
                r.raise_for_status()
                if total is None:
                    length = int(r.headers.get("Content-Length", 0))
                    total = offset + length
                with tmp.open("ab") as f, tqdm(
                    total=total, initial=offset, unit="B", unit_scale=True, desc="merimee.csv"
                ) as bar:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
                        bar.update(len(chunk))
            if total and tmp.stat().st_size < total:
                raise IOError(f"incomplete: {tmp.stat().st_size}/{total}")
            tmp.rename(dest)
            return dest
        except (requests.RequestException, IOError) as e:
            if attempt == retries - 1:
                raise
            session.trust_env = not session.trust_env
            mode = "proxy" if session.trust_env else "direct"
            print(
                f"retry {attempt + 1} ({e.__class__.__name__}), resuming at "
                f"{tmp.stat().st_size / 1e6 if tmp.exists() else 0:.0f} MB via {mode}",
                file=sys.stderr,
            )
            time.sleep(min(2**attempt, 30))
    raise RuntimeError("unreachable")


def filter_records(dump: Path, departments: set[str], limit: int | None):
    # HIST/DESC free-text fields exceed csv's default 128 KiB field limit.
    csv.field_size_limit(1 << 27)
    n = 0
    with dump.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="|")
        for row in reader:
            if (row.get(DPT_COLUMN) or "").strip() in departments:
                yield {k: (row.get(col) or None) for k, col in FIELD_MAP.items()}
                n += 1
                if limit and n >= limit:
                    return


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--departments", nargs="+", default=["67", "68"])
    ap.add_argument("--limit", type=int, default=None, help="max total records")
    ap.add_argument("--dump", default="data/raw/merimee.csv", help="cached dump location")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    dump = download_dump(Path(args.dump))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w", encoding="utf-8") as f:
        for rec in filter_records(dump, set(args.departments), args.limit):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    print(f"Wrote {n} records ({'+'.join(args.departments)}) to {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
