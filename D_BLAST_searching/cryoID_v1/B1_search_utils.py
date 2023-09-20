#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
WHAT this for? - to be fullfilled

Original Author: Qibo Xu @ Prof. Hong Zhou lab
University of California, Los Angeles (UCLA) &
University of Science and Technology of China (USTC)

Version 2, released in ?/?/2023
"""

import logging
import os
import subprocess
import sys
from argparse import Namespace
from collections import deque


# 9/19/2023:
# generate (degenerated/not) local blastp/n database and get database size
def DBGenerate(params,):
    """
    params.seqfile = seqfile (sequence pool file, fasta format)
    params.logfile = logfile
    params.degeneration = True/False (whether degenerate amino acid sequence)
    """

    # logging setup
    logging.basicConfig(
        filename=params.logfile, level=logging.INFO, format="%(message)s"
    )

    # get database name
    file_dir = os.path.dirname(params.seqfile)
    file_name = os.path.basename(params.seqfile)
    if params.seqfile.endswith(".fasta"):
        dbname = os.path.join(file_dir, "database", f"{file_name[:-6]}_db")
    else:
        err_mes = f"Seqfile {params.seqfile} is not of .fasta format! Exiting."
        logging.exception(err_mes)
        raise TypeError(err_mes)

    # calculate the database size
    if os.path.exists(f"{dbname}.txt"):
        with open(f"{dbname}.txt", "r") as f1:
            line = f1.readline()
            try:
                dbsize = int(line.split("\t")[1])
            except TypeError:
                err_mes = f"Cannot get database size from {dbname}.txt. Check it!"
                logging.exception(err_mes)
                raise TypeError(err_mes)
    else:
        dbsize = 0
        with open(params.seqfile, "r") as f2:
            for line in f2:
                if not line.startswith(">"):
                    line = line.strip()
                    dbsize += len(line)

    # build local database with (degenerated/not) sequences

    if not os.path.exists(f"{dbname}.pin"):

        # # (degenerate/not) (reverse/not) sequences - delete it
        # map_params = Namespace(seqfile=params.seqfile,
        #                        reverse=False,
        #                        logfile=params.logfile,
        #                        degeneration=params.degeneration)
        # dbsize, mapped_seqfile = SeqMap(map_params)

        mapped_seqfile = params.seqfile
        # build the local database by running makeblastdb
        mkdb_arg = [
            "makeblastdb",
            f"-in {mapped_seqfile}",
            f"-out {dbname}",
            f"-dbtype {dbtype}",
            "-parse_seqids",
        ]

        try:
            proc = subprocess.Popen(
                mkdb_arg,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                encoding="utf-8",
            )
            out, error = proc.communicate()
            # os.remove(mapped_seqfile)
        except OSError:
            err_mes = "Cannot find makeblastdb. Make sure you include it in PATH!"
            logging.exception(err_mes)
            raise OSError(err_mes)
        if error:
            err_mes = f"Cannot call makeblastdb to generate local database.\n{error}"
            logging.exception(err_mes)
            raise OSError(err_mes)

        # save the database size in {dbname}.txt file
        with open(f"{dbname}.txt", "w") as f1:
            f1.write(f"database size:\t{dbsize}")

    return [dbsize, dbname]


# 9/19/2023: no longer use it but rewrite, keep possible for future degeneration of RNA
# Map the natural sequences to degenerate codes for blastp searching
def SeqMap(params,):
    """
    params.seqfile = queryfile (query file name),
    params.reverse = True/False (whether include the reverse sequence)
    params.logfile = logfile
    params.degeneration = True/False (whether degenerate sequences)
    """
    # logging and outfile setup
    logging.basicConfig(
        filename=params.logfile, level=logging.INFO, format="%(message)s"
    )

    # count number of amino acids in input sequence
    num_amino_acid = 0

    # store and pop reversed lines
    if params.reverse:
        reversed_line = deque([])

    outfile = f"{params.seqfile[:-6]}_mapped.fasta"
    o1 = open(outfile, "w")

    f1 = open(params.seqfile, "r")

    for line in f1:
        if line and not line.startswith(">") and not line.startswith("#"):
            line = line.strip()
            try:
                line = "".join(aaMapping[e] for e in line)
            except KeyError as error:
                logging.exception(
                    f"Error in amino acid {str(error)}: line {line} of {params.seqfile}"
                )
                raise KeyError(
                    f"Error in amino acid {str(error)}: line {line} of {params.seqfile}"
                )
            num_amino_acid += len(line)
            o1.write(f"{line}\n")
        else:
            o1.write(f"{line}")

        # inverse the sequence and write it to the file
        if params.reverse:
            line = line.strip()
            if not line.startswith(">") and not line.startswith("#"):
                line = line[::-1] + "\n"
            else:
                line += "_rev\n"

            reversed_line.append(line)

    while reversed_line:
        o1.write(f"{reversed_line.popleft()}")

    f1.close()
    o1.close()

    return num_amino_acid, outfile


# 9/19/2023: no longer use it but rewrite
# Group the amino acids and yield degenearete sequences for user inspection
def SeqGroup(params,):
    """
    params.seqfile = queryfile (query file name),
    """
    # logging and outfile setup
    logging.basicConfig(
        filename=params.logfile, level=logging.INFO, format="%(message)s"
    )

    # count the sequence number and length
    seq_num = 0
    seq_len = 0

    outfile = f"{params.seqfile[:-6]}_simplified.fasta"

    with open(params.seqfile, "r") as f1:
        for line in f1:
            if line and line.startswith(">"):
                seq_num += 1
            elif line and not line.startswith(">") and not line.startswith("#"):
                line = line.strip()
                try:
                    line = "".join(aaMapping[e] for e in line)
                    line = "".join(aaRevMapping[e] for e in line)
                    seq_len += len(line)
                except KeyError as error:
                    logging.exception(
                        f"Error in amino acid {str(error)}: line {line} of {params.seqfile}"
                    )
                    raise KeyError(
                        f"Error in amino acid {str(error)}: line {line} of {params.seqfile}"
                    )
                line += "\n"

            with open(outfile, "w") as o1:
                o1.write(f"{line}")


# 9/20/2023: rewrite this function and delete pdb->fasta with phenix
# generate (degenerated) query files for blastp/n searching. One file for one query.
def QueryMap(params,):
    """
    params.queryfile = queryfile (query file name)
    params.reverse = True/False (whether include the reverse sequence)
    params.verbose = True/False (whether show verbose results)
    """
    # logging setup
    logging.basicConfig(
        filename=params.logfile, level=logging.INFO, format="%(message)s"
    )

    # 9/19/2023: won't use phenix in the future. to be continued tomorrow.
    # providing pdb file, use phenix.print_sequence to print the sequence.
    # Note that mse represents X in coot.
    # if params.queryfile.endswith(".pdb"):
    #     print_seq = ["phenix.print_sequence", params.queryfile, "--letter_for_mse=X"]
    #     # call phenix.print_sequence as subprocess and get the output/error message
    #     try:
    #         proc = subprocess.Popen(
    #             print_seq,
    #             stdout=subprocess.PIPE,
    #             stderr=subprocess.PIPE,
    #             encoding="utf-8",
    #         )
    #         (out, error) = proc.communicate()
    #     except OSError:
    #         err_mes = "Error: find phenix.print_sequence. Include it in your PATH!"
    #         logging.exception(err_mes)
    #         raise OSError(err_mes)

    #     if error:
    #         err_mes = f"Error when calling phenix.print_sequence to convert pdb file to fasta file\n{error}"
    #         logging.exception(err_mes)
    #         raise OSError(err_mes)

    #     # create the .fasta file
    #     queryfile = params.queryfile.split(".pdb")[0] + ".fasta"
    #     with open(queryfile, "w") as f0:
    #         f0.write(out)

    # if fasta file provided, directly use it
    if params.queryfile.endswith(".fasta"):
        queryfile = params.queryfile

    # generate the degenerate query sequence file for the user
    if params.verbose:
        group_params = Namespace(seqfile=queryfile, logfile=params.logfile)
        SeqGroup(group_params)

    # initialize query number, total length of queries, residue X number
    query_nums, query_lens, x_num = 0, 0, 0
    # creat a list to save query length and query file names
    query_info = []

    # creat query files. One file for one query.
    with open(queryfile, "r") as f1:
        f1_lines = f1.readlines()
        for line in f1_lines:
            if line.startswith(">"):
                query_nums += 1
                query_name = f"{line[1:].strip()}_{str(query_nums)}"
                query_split = f"{queryfile[:-6]}_{str(query_nums)}.fasta"
                query_info.append([query_name, None, query_split, 0])
                with open(query_split, "a") as o1:
                    o1.write(f">{query_name}\n")
            else:
                with open(query_split, "a") as o1:
                    o1.write(f"{line.strip()}")

    # Creat 6-letter query_file and update query_len, query_file and x_num in query_info
    for query_index in range(len(query_info)):
        query_split = query_info[query_index][2]
        map_params = Namespace(
                seqfile=query_split, reverse=params.reverse, logfile=params.logfile
            )
        query_len, query_file = SeqMap(map_params)
        with open(query_file, "r") as f1:
            f1_lines = f1.readlines()
            x_num = f1_lines[-1].count('x') + f1_lines[-1].count('X')
        os.remove(query_split)
        query_info[query_index][1:3] = [query_len, query_file, x_num]

    query_lens, x_nums = sum(query_info[:][1]), sum(query_info[:][3])

    if not query_nums:
        err_mes = "No query found. Check your input! Exiting"
        logging.exception(err_mes)
        raise OSError(err_mes)
    elif query_nums > 10:
        war_mes = "The number of queries are more than 10! Are you sure?"
        logging.warning(war_mes)
        print(f"Warning: {war_mes}")
        sys.stdout.flush()
    elif query_nums > 50:
        war_mes = "The number of queries are more than 50! Are you sure?"
        logging.warning(war_mes)
        print(f"Warning: {war_mes}")
        sys.stdout.flush()
        # err_mes = "The number of queries are more than 50! Check it again! Exiting..."
        # logging.exception(err_mes)
        # raise OSError(err_mes)

    # calculate animo acid number except for X, the average query length and X number
    aa_num = query_lens - x_nums
    ave_aa_num = int(round(aa_num * 1.0 / query_nums))
    ave_x_num = x_num * 1.0 / query_nums

    # print the query set info and record in log file
    query_set_info = f"""QUERY SEQUENCE SUMMARY
    Number of query sequences used: {str(query_nums)}
    Number of identifiable residues per query: {str(ave_aa_num)}
    Number of unidentifiable residues per query:  {ave_x_num:.2g}"""
    print("********************************************************************")
    print(query_set_info)
    sys.stdout.flush()
    logging.info(query_set_info)

    # check if the query is too short based on the simulation results (10% tolerance)
    if query_nums > 6:
        cutoff = query_set[0]
    else:
        cutoff = query_set[query_nums]

    if ave_aa_num < cutoff:
        war_mes = """Warning: query length is short!
You may consider extending your queries, or the final results may be not reliable!"""
        print(war_mes)
        sys.stdout.flush()
        logging.warn(war_mes)

    print("")
    sys.stdout.flush()
    logging.info("")

    return query_info


# 9/20/2023: rewrite this function
# Search the database using blastp
def SeqBlast(params):
    """
    params includes:
        blastp parameters dbsize, qsize, query, outfile, database, evalue, gapscore
    """
    # logging setup
    logging.basicConfig(
        filename=params.logfile, level=logging.INFO, format="%(message)s"
    )

    # Calculate the search space
    searchsp = str(params.dbsize * params.qsize)

    # set maximum number of aligned sequences to keep in blastp search
    if params.dbsize < 10000:
        max_seq = params.dbsize
    else:
        max_seq = 10000

    # check output filename
    output = params.outfile or params.query[:-6] + "_blast"

    # run blast search according to the provided parameters
    blast_arg = [
        "blastp",
        f"-task {task}",
        f"-matrix {matrix}",
        f"-db {params.database}",
        f"-query {params.query}",
        f"-out {output}",
        f"-evalue {params.evalue}",
        f"-comp_based_stats {comp_based_stats}",
        f"-dbsize {params.dbsize}",
        f"-searchsp {searchsp}",
        f"-word_size {word_size}",
        f"-gapopen {params.gapscore[0]}",
        f"-gapextend {params.gapscore[1]}",
        f"-outfmt {outfmt}",
        f"-max_target_seqs {max_seq}",
    ]

    try:
        proc = subprocess.Popen(blast_arg)
        out, error = proc.communicate()

    except OSError:
        err_mes = "Could not find blastp. Make sure you include it in PATH!"
        logging.exception(err_mes)
        raise OSError(err_mes)
    if error:
        err_mes = f"Error when calling blastp to search local database.\n{error}"
        logging.exception(err_mes)
        raise OSError(err_mes)


# makeblastdb and blast settings
# default parameters for makeblastdb
dbtype = "prot"  # molecular type of target db, default protein

# default parameters for blastp. Usually it works well.
task = "blastp-short"  # Task to execute
# task = 'blastp'
matrix = "PAM30"  # Scoring matrix name
comp_based_stats = "F"  # No composition-based statistics
outfmt = "7 qaccver saccver nident length qlen mismatch gapopen gaps qstart qend \
sstart send qseq sseq evalue"  # alignment view options, 7 = Tabular with comment lines
word_size = "2"  # Word size for wordfinder algorithm
# gapopen = '32767'			# Cost to open a gap.
# gapextend = '32767'			# Cost to extend a gap.
# Allowable (gapopen, gapextend) pairs in blastp: (7,2), (6,2), (5,2), (10,1), (9,1), \
# (8,1), (15,3), (14,2), (14,1), (13,3), (32767,32767)


# Map sequences tables for blastp searching / user inspection
# Amino acid degeneration mapping table - based on PAM30 scorig matrix
aaMapping = {
    "G": "G",
    "A": "G",
    "S": "G",
    "C": "G",
    "V": "G",
    "T": "G",
    "I": "G",
    "g": "G",
    "a": "G",
    "s": "G",
    "c": "G",
    "v": "G",
    "t": "G",
    "i": "G",
    "P": "P",
    "p": "P",
    "L": "Z",
    "D": "Z",
    "N": "Z",
    "E": "Z",
    "Q": "Z",
    "M": "Z",
    "l": "Z",
    "d": "Z",
    "n": "Z",
    "e": "Z",
    "q": "Z",
    "m": "Z",
    "K": "M",
    "R": "M",
    "k": "M",
    "r": "M",
    # 'K':'Y', 'R':'Y',
    # 'k':'Y', 'r':'Y',
    "H": "Y",
    "F": "Y",
    "Y": "Y",
    "h": "Y",
    "f": "Y",
    "y": "Y",
    "W": "W",
    "w": "W",
    # All other amino acids -> X
    "X": "X",
    "x": "X",
    # Selenocysteine is a cysteine analogue with a selenium-containing selenol group in place of the sulfur-containing thiol group.
    "U": "G",
    "u": "G",
    # asparagine/aspartic acid - asx - B,
    "B": "Z",
    "b": "Z",
    # glutamine/glutamic acid - glx - Z
    "Z": "Z",
    "z": "Z",
    # Pyrrolysine pyrroline side-chain is similar to that of lysine
    # in being basic and positively charged at neutral pH.
    # 'O':'M', 'o':'M'
}

# Reverse Mapping table from the blastp codes to the 6 simplified groups
aaRevMapping = {
    "G": "G",
    "P": "P",
    "Z": "L",
    "M": "K",
    "Y": "Y",
    "W": "W",
    "X": "X"
}

# reference query table/dictionary based on the simulation. assume 10% tolerance error.
# if query length is shorter than specified, print warning infomation
# query_num: cutoff_query_length. O for query_num > 6
query_set = {1: 30, 2: 20, 3: 16, 4: 15, 5: 13, 6: 12, 0: 12}
