from Bio import SeqIO
import argparse
import os

# record_iterator = SeqIO.parse("D_BLAST_searching/test_files/fasta/8RCM.fasta", "fasta")
# first_record = next(record_iterator)
# #print(first_record.description.split('|')[0].split('_')[1])
# print(first_record)

# def parser(cmdline):
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-q', '--queryfile', type=argparse.FileType('w'), required=True, 
#                         help='Queries (.fasta/.pdb file) to be searched')
    
#     args = parser.parse_args(cmdline.split())

#     return args


# args = parser('-q test.txt')
# print(args)
# args.queryfile.write('hello world')

path = os.path.splitext(os.path.basename('A/B/C/file.txt'))[0]
print(path)