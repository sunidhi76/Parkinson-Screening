"""Download the UCI Parkinson's voice dataset (parkinsons.data) into data/.

Order tried: (1) direct UCI URL, (2) `ucimlrepo` package, (3) clear manual instructions.
Optional `--mirror` fetches a third-party GitHub copy (identical bytes were observed from two independent
mirrors, but it is NOT verified against UCI by this script)."""
from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

DEST = Path(__file__).resolve().parent / "parkinsons.data"
UCI_URLS = ["https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data",
            "https://archive.ics.uci.edu/static/public/174/parkinsons.zip"]
MIRRORS = ["https://raw.githubusercontent.com/rshikharev/parkenson_project/main/parkinsons.data",
           "https://raw.githubusercontent.com/maskot1977/ipython_notebook/master/toydata/parkinsons.data"]
MANUAL = """Could not download automatically. Manual steps:
  1. Open https://archive.ics.uci.edu/dataset/174/parkinsons and download the dataset (CC BY 4.0).
  2. Extract `parkinsons.data` and save it as data/parkinsons.data
Cite: Little et al., 'Suitability of dysphonia measurements for telemonitoring of Parkinson's disease', IEEE TBME 2008."""


def _valid(text: str) -> bool:
    lines = text.strip().splitlines()
    return len(lines) == 196 and lines[0].startswith("name,") and "status" in lines[0]


def _fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=30) as r:
        return r.read()


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--mirror", action="store_true", help="use third-party GitHub mirrors")
    a = ap.parse_args()
    if DEST.exists() and _valid(DEST.read_text()):
        print(f"{DEST} already present."); return 0
    for url in UCI_URLS[:1] + (MIRRORS if a.mirror else []):
        try:
            data = _fetch(url).decode()
            if _valid(data):
                DEST.write_text(data); print(f"saved {DEST} from {url}\nmd5 {hashlib.md5(data.encode()).hexdigest()}"); return 0
        except Exception as e:  # noqa: BLE001
            print(f"failed {url}: {e}")
    try:
        from ucimlrepo import fetch_ucirepo
        d = fetch_ucirepo(id=174)
        df = d.data.original
        if "name" in df.columns and len(df) == 195:
            df.to_csv(DEST, index=False); print(f"saved {DEST} via ucimlrepo"); return 0
    except Exception as e:  # noqa: BLE001
        print(f"ucimlrepo failed: {e}")
    print(MANUAL); return 1


if __name__ == "__main__":
    sys.exit(main())
