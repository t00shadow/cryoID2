import os

rna_bases = {'A', 'U', 'G', 'C'}
dna_bases = {'A', 'T', 'G', 'C'}

def sort_pool(in_file, out_dir):
    file_name = os.path.basename(in_file).split('.')[0]
    pool = open(in_file, 'r')
    pool_n = open(os.path.join(out_dir, f"{file_name}_n.fasta"), 'w')
    pool_p = open(os.path.join(out_dir, f"{file_name}_p.fasta"), 'w')

    # read two lines at a time
    for line in pool:
        if line.startswith('>'):
            header = line.strip()
            sequence = next(pool).strip()
        
        if set(sequence).issubset(rna_bases):
            pool_n.write(f"{header}\n{sequence}\n")
        elif set(sequence).issubset(dna_bases):
            pool_n.write(f"{header}\n{sequence}\n")
        else:
            pool_p.write(f"{header}\n{sequence}\n")

    pool.close()
    pool_n.close()
    pool_p.close()


if __name__ == "__main__":
    sort_pool('D_BLAST_searching/test_files/fasta/gprot_ribo.fasta', 'D_BLAST_searching/test_files/fasta')