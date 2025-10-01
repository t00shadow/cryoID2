import asyncio
import glob
import os

import aiohttp
# import re
import pandas as pd
from aiohttp_retry import ExponentialRetry, RetryClient
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
        self.pdb_ids = pd.read_csv(csv_path)['fitted_pdbs'].tolist()
        self.emdb_ids = pd.read_csv(csv_path)['emdb_id'].tolist()
        self.save_path = save_path
        self.raw_fasta_path = os.path.join(save_path, self.name, 'raw')
        self.sleep_between_requests = sleep_between_requests
        self.retry_attempts = retry_attempts

    async def _download_fastas(self):
        """Download FASTA files sequentially, pausing between requests to avoid rate limits."""
        with tqdm(total=len(self.pdb_ids), desc=f"Downloading FASTA for {self.name} set") as pbar:
            retry_options = ExponentialRetry(attempts=self.retry_attempts)
            async with aiohttp.ClientSession() as base_session:
                retry_client = RetryClient(base_session, retry_options=retry_options)
                # Run downloads sequentially so we can sleep between requests
                for pdb_id, emdb_id in zip(self.pdb_ids, self.emdb_ids):
                    await self._download_fasta(pdb_id, emdb_id, retry_client, pbar)
                    # Sleep between requests to be gentle with the server
                    await asyncio.sleep(self.sleep_between_requests)

    async def _download_fasta(self, pdb_id: str, emdb_id: str, session: aiohttp.ClientSession, pbar: tqdm) -> None:
        link = f'https://www.rcsb.org/fasta/entry/{pdb_id.split(",")[0]}'
        async with session.get(link) as response:
            try:
                if response.status == 200:
                    fasta_data = await response.text()
                    # # Modify lines starting with '>'
                    # fasta_lines = []
                    # for line in fasta_data.splitlines():
                    #     if line.startswith('>'):
                    #         line = f">{emdb_id}|{line[1:]}"
                    #     fasta_lines.append(line)
                    # modified_fasta_data = '\n'.join(fasta_lines)
                    # # sanitize filenames to avoid invalid path characters
                    # def _sanitize(s: str) -> str:
                    #     # replace any character that's not alphanumeric, dot, underscore or hyphen with underscore
                    #     return re.sub(r'[^A-Za-z0-9._-]', '_', s)

                    # safe_emdb = _sanitize(str(emdb_id))
                    # safe_pdb = _sanitize(str(pdb_id))
                    # out_path = os.path.join(self.save_path, self.name, f"{safe_emdb}_{safe_pdb}.fasta")

                    # os.makedirs(os.path.dirname(out_path), exist_ok=True)

                    # with open(out_path, 'w') as fasta_file:
                    #     fasta_file.write(modified_fasta_data)
                    with open(os.path.join(self.raw_fasta_path, f"{emdb_id}_{pdb_id}.fasta"), 'w') as fasta_file:
                        fasta_file.write(fasta_data)
                else:
                    tqdm.write(f"Failed to download {pdb_id}. Status code: {response.status}")
            except Exception as e:
                tqdm.write(f"Error downloading {pdb_id}: {e}")
        pbar.update(1)

    async def _sort_fasta(self):
        fasta_p_path = os.path.join(self.save_path, self.name, f'{self.name}_p.fasta')
        fasta_n_path = os.path.join(self.save_path, self.name, f'{self.name}_n.fasta')
        # remove existing files if they exist
        os.remove(fasta_p_path) if os.path.exists(fasta_p_path) else None
        os.remove(fasta_n_path) if os.path.exists(fasta_n_path) else None

        link = 'https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb_id}/{chain_id}'
        failed = []
        for fastas in tqdm(glob.glob(os.path.join(self.raw_fasta_path, '*.fasta')), desc="Classifying chains"):
            for record in SeqIO.parse(fastas, 'fasta-pearson'):
                pdb_id = record.description.split('|')[0].split('_')[0]
                chain_id = record.description.split('|')[0].split('_')[1]
                # Increase timeouts to allow slow SSL handshakes / slow servers to respond.
                # Use a per-request timeout that sets generous connect/read timeouts.
                retry_options = ExponentialRetry(attempts=self.retry_attempts)
                # aiohttp.ClientTimeout accepts sock_connect and sock_read which help
                # cover SSL handshake and socket read timeouts (default sock_connect is 60s).
                # timeout = aiohttp.ClientTimeout(total=600, sock_connect=300, sock_read=300)
                async with aiohttp.ClientSession() as base_session:
                    retry_client = RetryClient(base_session, retry_options=retry_options)
                    try:
                        async with retry_client.get(link.format(pdb_id=pdb_id, chain_id=chain_id)) as response:
                            if response.status == 200:
                                data = await response.json()
                                identity = data.get('entity_poly', {}).get('rcsb_entity_polymer_type', '')
                                if identity == 'Protein':
                                    with open(fasta_p_path, 'a') as fasta_p:
                                        SeqIO.write(record, fasta_p, 'fasta')
                                elif identity == 'DNA' or identity == 'RNA':
                                    with open(fasta_n_path, 'a') as fasta_n:
                                        SeqIO.write(record, fasta_n, 'fasta')
                            else:
                                tqdm.write(f"Failed to fetch data for {pdb_id} chain {chain_id}. Status code: {response.status}")
                    except Exception as e:
                        # Log but don't crash the whole sorting process for a slow/failed request
                        tqdm.write(f"Error fetching data for {pdb_id} chain {chain_id}: {e}")
                        failed.append(record)
                await asyncio.sleep(self.sleep_between_requests)
        if failed:
            await self._retry_failed(failed)
    
    async def _retry_failed(self, failed: list):
        fasta_p_path = os.path.join(self.save_path, self.name, f'{self.name}_p.fasta')
        fasta_n_path = os.path.join(self.save_path, self.name, f'{self.name}_n.fasta')

        link = 'https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb_id}/{chain_id}'
        retry_options = ExponentialRetry(attempts=self.retry_attempts)
        #timeout = aiohttp.ClientTimeout(total=600, sock_connect=300, sock_read=300)
        for record in tqdm(failed, desc="Retrying failed requests"):
            async with aiohttp.ClientSession() as base_session:
                retry_client = RetryClient(base_session, retry_options=retry_options)
                pdb_id = record.description.split('|')[0].split('_')[0]
                chain_id = record.description.split('|')[0].split('_')[1]
                try:
                    async with retry_client.get(link.format(pdb_id=pdb_id, chain_id=chain_id)) as response:
                        if response.status == 200:
                            data = await response.json()
                            identity = data.get('entity_poly', {}).get('rcsb_entity_polymer_type', '')
                            if identity == 'Protein':
                                with open(fasta_p_path, 'a') as fasta_p:
                                    SeqIO.write(record, fasta_p, 'fasta')
                            elif identity == 'DNA' or identity == 'RNA':
                                with open(fasta_n_path, 'a') as fasta_n:
                                    SeqIO.write(record, fasta_n, 'fasta')
                            failed.remove(record)
                        else:
                            tqdm.write(f"Failed to fetch data for {pdb_id} chain {chain_id}. Status code: {response.status}")
                except Exception as e:
                    # Log but don't crash the whole sorting process for a slow/failed request
                    tqdm.write(f"Error fetching data for {pdb_id} chain {chain_id}: {e}")
            await asyncio.sleep(self.sleep_between_requests)
        if failed:
            await self._retry_failed(failed)

    def run(self):
        os.makedirs(self.save_path, exist_ok=True)
        os.makedirs(self.raw_fasta_path, exist_ok=True)

        # asyncio.run(self._download_fastas())
        # count = len(glob.glob(os.path.join(self.raw_fasta_path, '*.fasta')))
        # assert count == len(self.pdb_ids), f"Expected {len(self.pdb_ids)} FASTA files, but found {count}."
        # print(f"{count} entries in {self.name} set")

        asyncio.run(self._sort_fasta())
        print(f"FASTA files sorted into protein and nucleotide sequences for {self.name} set")


if __name__ == '__main__':
    csv_path = 'D_BLAST_searching/test_files/gprot_ribo.csv'
    save_path = 'D_BLAST_searching/test_files/fasta'
    downloader = fasta_downloader(csv_path, save_path)
    downloader.run()
