#!/bin/bash

# Paths (relative to repository root)
query_dir="D_BLAST_searching/test_files/fasta"
query_file="Leon_test.fasta"
pool_file="malaria_pool.fasta"
output_dir="D_BLAST_searching/test_files/results"

# cryoID_v1
# python3 D_BLAST_searching/cryoID_v1/python3/search_pool_v1_1.py -od "$output_dir/cryoid_v1/" -q "$query_dir/$query_file" -p "D_BLAST_searching/test_files/fasta/cryoid_v1/${pool_file}" -r -n 500

# cryoID_v2
python3 D_BLAST_searching/cryoID_v2/search_pool.py -od "$output_dir/cryoid_v2/" -q "$query_dir/$query_file" -p "D_BLAST_searching/test_files/fasta/cryoid_v2/${pool_file}" -r -l 1 100000