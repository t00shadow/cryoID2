from Bio import SeqIO

record_iterator = SeqIO.parse("D_BLAST_searching/test_files/fasta/8RCM.fasta", "fasta")
first_record = next(record_iterator)
print(first_record.description.split('|')[0].split('_')[1])