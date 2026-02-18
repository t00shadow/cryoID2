import pathlib
import sys
from typing import Iterable, Optional

import pandas as pd
import requests
from requests.adapters import HTTPAdapter, Retry

# EDIT
CSV_PATH = pathlib.Path("identifier/all_hits.csv") # input CSV
OUT_DIR  = pathlib.Path("alphafold_models") # where structures will be saved

# retries
retries = Retry(total=4, backoff_factor=0.25, status_forcelist=[500, 502, 503, 504])
session = requests.Session()
session.mount("https://", HTTPAdapter(max_retries=retries))

def first_uniprot_accession(query: str, timeout: int = 15) -> Optional[str]:
    """Return the first UniProt accession for "query" or None if none found."""
    url = "https://rest.uniprot.org/uniprotkb/search"
    params = {
        "query": query,
        "fields": "accession",
        "format": "json",
        "size": 1, # take only the first hit
    }
    try:
        r = session.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        results = r.json().get("results", [])
        return results[0]["primaryAccession"] if results else None
    except requests.RequestException:
        return None

def download_alphafold(
    accession: str,
    out_dir: pathlib.Path,
    formats: Iterable[str] = ("pdb", "cif"),
    timeout: int = 15,
) -> Optional[pathlib.Path]:
    """
    Try the listed "formats" and save the first one that exists.
    Returns the saved path, or None if no model is available.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in formats:
        url = f"https://alphafold.ebi.ac.uk/files/AF-{accession}-F1-model_v4.{ext}"
        try:
            r = session.get(url, timeout=timeout)
            if r.ok and r.content and not r.text.startswith("<!DOCTYPE"):
                path = out_dir / f"{accession}.{ext}"
                path.write_bytes(r.content)
                if path.stat().st_size > 0:
                    return path
                path.unlink(missing_ok=True)
        except requests.RequestException:
            continue
    return None

def main():
    # read the CSV
    try:
        df = pd.read_csv(CSV_PATH)
    except FileNotFoundError:
        sys.exit(f"Input file not found: {CSV_PATH}")
    if "target_name" not in df.columns:
        sys.exit("CSV must contain a 'target_name' column.")

    queries = df["target_name"].dropna().unique()

    # process each query
    for q in queries:
        acc = first_uniprot_accession(q)
        if acc is None:
            print(f"{q}: no UniProt accession found")
            continue

        path = download_alphafold(acc, OUT_DIR)
        if path is None:
            print(f"{q} ({acc}): AlphaFold model not available")
        else:
            print(f"{q} ({acc}): saved {path}")

if __name__ == "__main__":
    main()
