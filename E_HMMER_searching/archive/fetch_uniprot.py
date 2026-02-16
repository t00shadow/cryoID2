import asyncio
import os

import pandas as pd
import requests
from requests.adapters import HTTPAdapter, Retry
from tqdm import tqdm

retries = Retry(total=5, backoff_factor=0.25, status_forcelist=[500, 502, 503, 504])
session = requests.Session()
session.mount("https://", HTTPAdapter(max_retries=retries))

async def uniprot_search(query: str, 
                         fields: str = "accession,protein_name", 
                         size: int = 500,
                         ) -> pd.DataFrame:
    """Search UniProt for a given query."""
    base_url = "https://rest.uniprot.org/uniprotkb/search"
    params = {
        "query": query,
        "fields": fields,
        "format": "json",
        "size": size
    }
    headers = {"accept": "application/json"}
    r = session.get(base_url, headers=headers, params=params)
    data = r.json()

    accessions = []
    protein_names = []
    for i in range(len(data["results"])):
        accession = data["results"][i]["primaryAccession"]
        protein_name = data["results"][i]["proteinDescription"]["submissionNames"][0]["fullName"]["value"]
        accessions.append(accession)
        protein_names.append(protein_name)
    df = pd.DataFrame({"accession": accessions, "protein_name": protein_names})
    return df

async def get_uniprot_accession(hit_path: str):
    hit_df = pd.read_csv(hit_path)
    querys = hit_df["target_name"].unique()
    uni_df = await asyncio.gather(uniprot_search(query for query in querys))
    uni_df = pd.concat(uni_df, ignore_index=True)
    uni_df.to_csv(os.path.join(os.path.dirname(hit_path), "uniprot_accession.csv"), index=False)

def main():
    hit_path = "E_HMMER_searching/test_files_pyhmmer/all_hits.csv"
    asyncio.run(get_uniprot_accession(hit_path))

if __name__ == "__main__":
    main()