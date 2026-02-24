#!/bin/bash

# remove test files
rm test_files/test_*.hmm test_files/*_results.txt

# create test files
python3 HAMMER_test.py

# Run hmmsearch for each test HMM file
for i in `ls test_files/test_*.hmm`
do
    hmmsearch $i search_pool.fasta > ${i%.*}_results.txt
done

echo "Test files created and hmmsearch results saved."