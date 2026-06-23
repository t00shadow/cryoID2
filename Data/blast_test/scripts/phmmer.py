from Bio import SeqIO
import glob

if __name__ == "__main__":
    with open("Data/blast_test/queries_mapped.fasta", "a") as output_handle:
        for file in glob.glob("Data/blast_test/query_files/*_mapped.fasta"):
            for record in SeqIO.parse(file, "fasta"):
                SeqIO.write(record, output_handle, "fasta")