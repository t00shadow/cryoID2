import asyncio
import glob
import os
import shutil

import aiohttp
import re
import pandas as pd
from aiohttp_retry import ExponentialRetry, RetryClient
from tqdm import tqdm

SAVE_PATH = 'E_HMMER_searching/test_files/fasta_files'
SLEEP_BETWEEN_REQUESTS = 0.1  # seconds to wait between requests; adjust as needed

async def main(name, pdb_ids, emdb_ids):
    """Download FASTA files sequentially, pausing between requests to avoid rate limits."""
    with tqdm(total=len(pdb_ids), desc=f"Downloading FASTA for {name} set") as pbar:
        retry_options = ExponentialRetry(attempts=5, start_timeout=0.1)
        async with aiohttp.ClientSession() as base_session:
            retry_client = RetryClient(base_session, retry_options=retry_options)
            # Run downloads sequentially so we can sleep between requests
            for pdb_id, emdb_id in zip(pdb_ids, emdb_ids):
                await download_fasta(pdb_id, emdb_id, retry_client, pbar)
                # Sleep between requests to be gentle with the server
                await asyncio.sleep(SLEEP_BETWEEN_REQUESTS)

async def download_fasta(pdb_id: str, emdb_id: str, session: aiohttp.ClientSession, pbar: tqdm) -> None:
    link = f'https://www.rcsb.org/fasta/entry/{pdb_id.split(",")[0]}'
    async with session.get(link) as response:
        if response.status == 200:
            fasta_data = await response.text()
            # Modify lines starting with '>'
            fasta_lines = []
            for line in fasta_data.splitlines():
                if line.startswith('>'):
                    line = f">{emdb_id}|{line[1:]}"
                fasta_lines.append(line)
            modified_fasta_data = '\n'.join(fasta_lines)
            # sanitize filenames to avoid invalid path characters
            def _sanitize(s: str) -> str:
                # replace any character that's not alphanumeric, dot, underscore or hyphen with underscore
                return re.sub(r'[^A-Za-z0-9._-]', '_', s)

            safe_emdb = _sanitize(str(emdb_id))
            safe_pdb = _sanitize(str(pdb_id))
            out_path = os.path.join(SAVE_PATH, f"{safe_emdb}_{safe_pdb}.fasta")
            # ensure parent dir exists (should already from main, but be defensive)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, 'w') as fasta_file:
                fasta_file.write(modified_fasta_data)
        else:
            tqdm.write(f"Failed to download {pdb_id}. Status code: {response.status}")
    pbar.update(1)

def combine_fasta_files():
    combined_fasta_path = f'{SAVE_PATH}/combined.fasta'
    combined_fasta_content = []

    for fastas in glob.glob(f'{SAVE_PATH}/*.fasta'):
        with open(fastas, 'r') as fasta_file:
            combined_fasta_content.append(fasta_file.read())

    with open(combined_fasta_path, 'w') as combined_file:
        combined_file.write('\n'.join(combined_fasta_content))

if __name__ == '__main__':
    if os.path.exists(SAVE_PATH):
        shutil.rmtree(SAVE_PATH)

    csv_path = 'E_HMMER_searching/test_files/test_pool.csv'
    name = 'test'

    pdb_ids = pd.read_csv(csv_path)['fitted_pdbs'].tolist()
    emdb_ids = pd.read_csv(csv_path)['emdb_id'].tolist()
    os.makedirs(SAVE_PATH, exist_ok=True)
    asyncio.run(main(name, pdb_ids, emdb_ids))
    combine_fasta_files()
    count = len(glob.glob(f'{SAVE_PATH}/*.fasta'))
    print(f"{count} entries in {name} set")
