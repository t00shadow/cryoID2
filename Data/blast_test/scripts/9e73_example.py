from Bio import SeqIO
import os
import random
import subprocess
import sys

if __name__ == "__main__":
    pool = 'Data/blast_test/Ca_ncRNA.fasta'
    gt = 'Data/blast_test/9e73_gt.fasta'
    query = 'Data/blast_test/9e73_q.fasta'
    filtered_pool = 'Data/blast_test/Ca_ncRNA_filtered.fasta'
    mapped_pool = 'Data/blast_test/Ca_ncRNA_mapped.fasta'
    mapping = 'Data/blast_test/Ca_ncRNA_mapping.txt'

    mapped_records = []
    with open(mapping, 'w') as f:
        for idx, record in enumerate(SeqIO.parse(pool, "fasta")):
            f.write("Seq{:04d}\t{}\n".format(idx+1, record.id))
            record.id = "Seq{:04d}".format(idx+1)
            record.description = ""
            mapped_records.append(record)
    SeqIO.write(mapped_records, mapped_pool, "fasta")

    gt_len = len(list(SeqIO.parse(gt, "fasta"))[0])
    tolerance_factor = 0.5
    len_tolerance = int(tolerance_factor * gt_len)
    
    filtered_records = []
    for record in SeqIO.parse(mapped_pool, "fasta"):
        if abs(len(record) - gt_len) <= len_tolerance:
            filtered_records.append(record)       
    SeqIO.write(filtered_records, filtered_pool, "fasta")
    print(f"Filtered {len(filtered_records)} records from {pool} based on tolerance factor of {tolerance_factor} ({len_tolerance}).")

    out_dir = "Data/blast_test/9e73_example"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    n_candidates = 19
    n_pool = 0
    gt_id = SeqIO.read(gt, "fasta").id
    base_cmd = [sys.executable, "-m", "CryoID_v2", "b", "-q", query,
                "-v", "-b", "-r"]
    while True:
        # randomly select n_candidates from filtered_records
        selected_records = random.sample(filtered_records, n_candidates)
        selected_records.append(SeqIO.read(gt, "fasta"))  # Add the ground truth
        out_path = os.path.join(out_dir, f"ncRNA_p{n_pool}.fasta")
        SeqIO.write(selected_records, out_path, "fasta")
        result = subprocess.run(args=[*base_cmd, "-p", out_path],
                                capture_output=True,
                                text=True,
                                )
        
        # Look for "Most likely candidate:" in the output
        candidate = None
        found_match = False
        for line in result.stderr.split('\n'):
            if "Most likely candidate:" in line:
                candidate = line.split(":")[-1].strip()
                if candidate == gt_id:
                    found_match = True
                    break
        
        if found_match:
            print(f"Found match at pool {n_pool}")
            break
        else:
            print("Current pool number:", n_pool, ", candidate:", candidate)
        
        n_pool += 1
