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
                 ):
        if name is not None:
            self.name = name
        else:
            self.name = os.path.basename(csv_path).split('.')[0]

        if os.path.basename(csv_path).split('.')[-1] == 'csv':
            raw_data = pd.read_csv(csv_path)
        elif os.path.basename(csv_path).split('.')[-1] == 'tsv':
            raw_data = pd.read_csv(csv_path, sep='\t')
        if 'fitted_pdbs' and 'emdb_id' and 'accession' not in raw_data.columns:
            raise ValueError(f"CSV file must contain 'fitted_pdbs', 'emdb_id' or 'accession' columns.")
        if 'fitted_pdbs' in raw_data.columns:
            self.pdb_ids = raw_data['fitted_pdbs'].tolist()
        else:
            self.pdb_ids = None
        if 'emdb_id' in raw_data.columns:
            self.emdb_ids = raw_data['emdb_id'].tolist()
        else:
            self.emdb_ids = None
        if 'accession' in raw_data.columns: # uniprot accession
            self.accession_ids = raw_data['accession'].tolist()
        else:
            self.accession_ids = None
        
        self.save_path = save_path
        self.raw_fasta_path = os.path.join(save_path, self.name, 'raw')
        self.fasta_p_path = os.path.join(self.save_path, self.name, f'{self.name}_p.fasta')
        self.fasta_n_path = os.path.join(self.save_path, self.name, f'{self.name}_n.fasta')
        self.sleep_between_requests = sleep_between_requests
        self.retry_attempts = retry_attempts

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
        task_len = len(self.pdb_ids or self.emdb_ids or self.accession_ids)
        
        with tqdm(total=task_len, desc=f"Downloading FASTA for {self.name} set") as pbar:
            # Run downloads sequentially so we can sleep between requests
            if self.pdb_ids is not None:
                for pdb_id, emdb_id in zip(self.pdb_ids, self.emdb_ids):
                    await self._download_fasta(pdb_id, emdb_id, None, pbar)
                    # Sleep between requests to be gentle with the server
                    await asyncio.sleep(self.sleep_between_requests)
            else:
                for assession_id in self.accession_ids:
                    await self._download_fasta(None, None, assession_id, pbar)
                    # Sleep between requests to be gentle with the server
                    await asyncio.sleep(self.sleep_between_requests)

    async def _download_fasta(self, pdb_id: str, emdb_id: str, assession_id: str, pbar: tqdm) -> None:
        if assession_id:
            link = f'https://rest.uniprot.org/uniprotkb/{assession_id}.fasta'
        else:
            link = f'https://www.rcsb.org/fasta/entry/{pdb_id.split(",")[0]}'
        try:
            # run the blocking requests.get in a thread using the shared session
            response = await asyncio.to_thread(self.session.get, link, timeout=30)
            if response.status_code == 200:
                fasta_data = response.text
                if assession_id:
                    with open(os.path.join(self.raw_fasta_path, f"{assession_id}.fasta"), 'w') as fasta_file:
                        fasta_file.write(fasta_data)
                else:
                    with open(os.path.join(self.raw_fasta_path, f"{emdb_id}_{pdb_id}.fasta"), 'w') as fasta_file:
                        fasta_file.write(fasta_data)
            else:
                tqdm.write(f"Failed to download {pdb_id or assession_id}. Status code: {response.status_code}")
        except Exception as e:
            tqdm.write(f"Error downloading {pdb_id}: {e}")
        pbar.update(1)

    async def _sort_fasta(self) -> None:
        """Sort downloaded FASTA files into protein and nucleotide sequences."""
        link = 'https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb_id}/{chain_id}'
        records_to_process = []
        for fastas in tqdm(glob.glob(os.path.join(self.raw_fasta_path, '*.fasta')), desc="Classifying chains"):
            for record in SeqIO.parse(fastas, 'fasta-pearson'):
                records_to_process.append(record)
        
        # Process with retries
        max_retries = self.retry_attempts
        for attempt in range(max_retries):
            failed = []
            for record in tqdm(records_to_process, desc=f"Classifying chains (attempt {attempt+1})"):
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
                        failed.append(record)
                except Exception as e:
                    # Log but don't crash the whole sorting process for a slow/failed request
                    tqdm.write(f"Error fetching data for {pdb_id} chain {chain_id}: {e}")
                    failed.append(record)
                await asyncio.sleep(self.sleep_between_requests)
            records_to_process = failed
            if not failed:
                break
        if failed:
            tqdm.write(f"Some records could not be classified after {max_retries} attempts.")
            tqdm.write(f"Unclassified records: {[record.id for record in failed]}")

    def combine(self) -> None:
        """Combine raw FASTA files into a single file without sorting."""
        combined_fasta_path = os.path.join(self.save_path, self.name, f'{self.name}_combined.fasta')
        records = [] # save records to avoid duplicates
        with open(combined_fasta_path, 'w') as combined_fasta:
            for fasta_file in glob.glob(os.path.join(self.raw_fasta_path, '*.fasta')):
                for record in SeqIO.parse(fasta_file, 'fasta-pearson'):
                    if record.id not in records:
                        SeqIO.write(record, combined_fasta, 'fasta')
                        records.append(record.id)
        print(f"Combined FASTA file created at {combined_fasta_path}")

    def download(self) -> None:
        os.makedirs(self.save_path, exist_ok=True)
        os.makedirs(self.raw_fasta_path, exist_ok=True)

        asyncio.run(self._download_fastas())
        count = len(glob.glob(os.path.join(self.raw_fasta_path, '*.fasta')))
        assert count == len(self.pdb_ids), f"Expected {len(self.pdb_ids)} FASTA files, but found {count}."
        print(f"{count} entries in {self.name} set")

    def sort(self) -> None:
        # remove existing files if they exist
        os.remove(self.fasta_p_path) if os.path.exists(self.fasta_p_path) else None
        os.remove(self.fasta_n_path) if os.path.exists(self.fasta_n_path) else None

        asyncio.run(self._sort_fasta())
        print(f"FASTA files sorted into protein and nucleotide sequences for {self.name} set")


if __name__ == '__main__':
    csv_path = 'pool_files/MASSIVE-COMPLETE-eee84186-group_by_protein-main.tsv'
    save_path = 'pool_files'
    with fasta_downloader(csv_path, save_path) as dl:
        # dl.download()
        dl.combine()
        # dl.sort()
