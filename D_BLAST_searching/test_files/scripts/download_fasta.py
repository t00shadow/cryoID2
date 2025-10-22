import asyncio
import glob
import os

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import pandas as pd
from tqdm import tqdm

from Bio import SeqIO


class fasta_downloader:
    def __init__(self, 
                 csv_path:str,
                 save_path: str,
                 name: str = None,
                 sleep_between_requests: float = 0.1,
                 retry_attempts: int = 3,
                 download: bool = True,
                 sort: bool = True,
                 ):
        if name is not None:
            self.name = name
        else:
            self.name = os.path.basename(csv_path).split('.')[0]
        self.pdb_ids = pd.read_csv(csv_path)['fitted_pdbs'].tolist()
        self.emdb_ids = pd.read_csv(csv_path)['emdb_id'].tolist()
        self.save_path = save_path
        self.raw_fasta_path = os.path.join(save_path, self.name, 'raw')
        self.fasta_p_path = os.path.join(self.save_path, self.name, f'{self.name}_p.fasta')
        self.fasta_n_path = os.path.join(self.save_path, self.name, f'{self.name}_n.fasta')
        self.sleep_between_requests = sleep_between_requests
        self.retry_attempts = retry_attempts
        self.download = download
        self.sort = sort

        # configure a single requests Session with Retry and reuse it
        retry_strategy = Retry(
            total=self.retry_attempts,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"],
            backoff_factor=0.5,
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session = requests.Session()
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def __enter__(self):
        # Allow use as a synchronous context manager: with fasta_downloader(...) as dl:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Ensure the shared session is closed whether exiting normally or due to an exception
        try:
            self.session.close()
        except Exception:
            pass
        # Do not suppress exceptions; return False so exceptions propagate
        return False

    async def _download_fastas(self) -> None:
        """Download FASTA files sequentially, pausing between requests to avoid rate limits."""
        with tqdm(total=len(self.pdb_ids), desc=f"Downloading FASTA for {self.name} set") as pbar:
            # Run downloads sequentially so we can sleep between requests
            for pdb_id, emdb_id in zip(self.pdb_ids, self.emdb_ids):
                await self._download_fasta(pdb_id, emdb_id, pbar)
                # Sleep between requests to be gentle with the server
                await asyncio.sleep(self.sleep_between_requests)

    async def _download_fasta(self, pdb_id: str, emdb_id: str, pbar: tqdm) -> None:
        link = f'https://www.rcsb.org/fasta/entry/{pdb_id.split(",")[0]}'
        try:
            # run the blocking requests.get in a thread using the shared session
            response = await asyncio.to_thread(self.session.get, link, timeout=30)
            if response.status_code == 200:
                fasta_data = response.text
                with open(os.path.join(self.raw_fasta_path, f"{emdb_id}_{pdb_id}.fasta"), 'w') as fasta_file:
                    fasta_file.write(fasta_data)
            else:
                tqdm.write(f"Failed to download {pdb_id}. Status code: {response.status_code}")
        except Exception as e:
            tqdm.write(f"Error downloading {pdb_id}: {e}")
        pbar.update(1)

    async def _sort_fasta(self) -> None:
        # remove existing files if they exist
        os.remove(self.fasta_p_path) if os.path.exists(self.fasta_p_path) else None
        os.remove(self.fasta_n_path) if os.path.exists(self.fasta_n_path) else None

        link = 'https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb_id}/{chain_id}'
        failed = []
        for fastas in tqdm(glob.glob(os.path.join(self.raw_fasta_path, '*.fasta')), desc="Classifying chains"):
            for record in SeqIO.parse(fastas, 'fasta-pearson'):
                pdb_id = record.description.split('|')[0].split('_')[0]
                chain_id = record.description.split('|')[0].split('_')[1]
                try:
                    response = await asyncio.to_thread(self.session.get, link.format(pdb_id=pdb_id, chain_id=chain_id), timeout=30)
                    if response.status_code == 200:
                        data = response.json()
                        identity = data.get('entity_poly', {}).get('rcsb_entity_polymer_type', '')
                        if identity == 'Protein':
                            with open(self.fasta_p_path, 'a') as fasta_p:
                                SeqIO.write(record, fasta_p, 'fasta')
                        elif identity == 'DNA' or identity == 'RNA':
                            with open(self.fasta_n_path, 'a') as fasta_n:
                                SeqIO.write(record, fasta_n, 'fasta')
                    else:
                        tqdm.write(f"Failed to fetch data for {pdb_id} chain {chain_id}. Status code: {response.status_code}")
                except Exception as e:
                    # Log but don't crash the whole sorting process for a slow/failed request
                    tqdm.write(f"Error fetching data for {pdb_id} chain {chain_id}: {e}")
                    failed.append(record)
                await asyncio.sleep(self.sleep_between_requests)
        if failed:
            await self._retry_failed(failed)
    
    async def _retry_failed(self, failed: list,) -> None:
        link = 'https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb_id}/{chain_id}'
        for record in tqdm(failed, desc="Retrying failed requests"):
            pdb_id = record.description.split('|')[0].split('_')[0]
            chain_id = record.description.split('|')[0].split('_')[1]
            try:
                response = await asyncio.to_thread(self.session.get, link.format(pdb_id=pdb_id, chain_id=chain_id), timeout=30)
                if response.status_code == 200:
                    data = response.json()
                    identity = data.get('entity_poly', {}).get('rcsb_entity_polymer_type', '')
                    if identity == 'Protein':
                        with open(self.fasta_p_path, 'a') as fasta_p:
                            SeqIO.write(record, fasta_p, 'fasta')
                    elif identity == 'DNA' or identity == 'RNA':
                        with open(self.fasta_n_path, 'a') as fasta_n:
                            SeqIO.write(record, fasta_n, 'fasta')
                    failed.remove(record)
                else:
                    tqdm.write(f"Failed to fetch data for {pdb_id} chain {chain_id}. Status code: {response.status_code}")
            except Exception as e:
                # Log but don't crash the whole sorting process for a slow/failed request
                tqdm.write(f"Error fetching data for {pdb_id} chain {chain_id}: {e}")
            await asyncio.sleep(self.sleep_between_requests)
        if failed:
            await self._retry_failed(failed)

    def run(self) -> None:
        os.makedirs(self.save_path, exist_ok=True)
        os.makedirs(self.raw_fasta_path, exist_ok=True)

        if self.download:
            asyncio.run(self._download_fastas())
            count = len(glob.glob(os.path.join(self.raw_fasta_path, '*.fasta')))
            assert count == len(self.pdb_ids), f"Expected {len(self.pdb_ids)} FASTA files, but found {count}."
            print(f"{count} entries in {self.name} set")

        asyncio.run(self._sort_fasta()) if self.sort else None

        # close the shared requests session
        try:
            self.session.close()
        except Exception:
            pass

        print(f"FASTA files sorted into protein and nucleotide sequences for {self.name} set")


if __name__ == '__main__':
    csv_path = 'D_BLAST_searching/test_files/gprot_ribo.csv'
    save_path = 'D_BLAST_searching/test_files/fasta'
    with fasta_downloader(csv_path, save_path) as dl:
        dl.run()
