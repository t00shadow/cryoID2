#!/bin/bash

# remove test files
rm test_files/test_*.hmm test_files/test_*_results.txt

# Create test HMM files
for i in {1..5}
do
    touch test_files/test_${i}.hmm
done

python3 HAMMER_test.py

# Run hmmsearch for each test HMM file
for i in {1..5}
do
    hmmsearch test_files/test_${i}.hmm search_pool.fasta > test_files/test_${i}_results.txt
done

echo "Test files created and hmmsearch results saved in test_files directory."