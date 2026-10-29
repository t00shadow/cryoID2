#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Canonical stage-D reduced-alphabet candidate-pool search.

Version 1.1, released in ?/?/2022
search_pool_v1_1.py translates both the segment sequences and all the candidate protein sequences of the pool into the simplified “6-letter” code and performs a modified blastp search of the entire pool
using the segment sequences as queries.
Revisor: Qibo Xu @ Prof. Hong Zhou lab
University of California, Los Angeles (UCLA)
Version 1.1, released in ?/?/2022

-----------------------------
search_pool.py translates both the segment sequences and all the candidate protein sequences of the pool into the simplified “6-letter” code and performs a modified blastp search of the entire pool
using the segment sequences as queries.

Original Author: Xiaorun Li (Lee) @ Prof. Hong Zhou lab
University of California, Los Angeles (UCLA) &
University of Science and Technology of China (USTC)

Version 1.0, released in 9/1/2019
"""

import datetime
import math
import operator
import os
import shutil
import sys
import time
from argparse import Namespace

from . import search_support as search_sup
from .evalues import alignment_log10_evalue, format_log10_evalue, linear_evalue
from Bio import SeqIO
from matplotlib import pyplot as plt

# default blast format
blastformat7 = '# Fields: query acc.ver, subject acc.ver, identical, alignment length, query length, mismatches, gap opens, gaps, q. start, q. end, s. start, s. end, query seq, subject seq, evalue'
blastformat7_with_bitscore = blastformat7 + ', bit score'

# save the results in class BlastSeq
class BlastSeq:

    def __init__(self, result, *, dbsize=None):
        # read data from blast results
        self.data = result
        results = result.split('\t')
        self.query, self.candidate = results[:2]        # 'subject' in blast results corresponds to candidate in our case
        self.identity, self.alignlength, self.querylength, self.mismatch, self.gapopen, self.gaps = map(int, results[2:8])
        self.qstart, self.qend, self.sstart, self.send = map(int, results[8:12])
        self.qseq, self.sseq = results[12:14]
        self.evalue = float(results[14])
        self.bit_score = float(results[15]) if len(results) == 16 else None
        self.log10_evalue = alignment_log10_evalue(
            self.evalue, bit_score=self.bit_score,
            search_space=dbsize * self.querylength if dbsize is not None else None,
        )
        if len(results) == 17:
            self.length = int(results[15])
            self.composite_evalue = float(results[16])

    # add composite E-value
    def set_composite_evalue(self, composite_evalue):
        self.composite_evalue = composite_evalue
        self.data += f'\t{str(composite_evalue)}'

    def set_composite_log10_evalue(self, composite_log10_evalue):
        self.composite_log10_evalue = float(composite_log10_evalue)

    # add gene names/length info
    def set_prot_len(self, length):
        self.length = int(length)
        self.data += f'\t{str(self.length)}'

    # add overall identity % info
    def set_ave_identity(self, ave_identity):
        self.ave_identity = float(ave_identity)
        self.data += f'\t{str(self.ave_identity)}'


def Blast(params):
    """
    Blast each query sequence against the specified database.

    Args:
    - params: a namespace object containing the following attributes:
        - query_info: a list of query sequences, where each query is represented by a list of four elements: [name, length, file name, X_num]
          query_info = [[query1], ... ], query1 = [name, length, file name, X_num]
        - dbname: the name of the blast database
        - dbsize: the size of the database
        - evalue: the E-value threshold for the blast search
        - gapscore: the gap opening and extension penalties for the blast search
        - logfile: the name of the log file to write blast messages to

    Returns:
    - blast_files: a list of paths to the blast result files
    """

    blast_files = []

    # loop over all query
    for query in params.query_info:
        query_name, query_len, query_file, X_num = query
        blast_file = query_file.split('_mapped.fasta')[0] + '_blast.txt'

        # Run blastp search
        blastp_params = Namespace(dbsize=params.dbsize, qsize=query_len, query=query_file,
                                  outfile=blast_file, database=params.dbname, evalue=params.evalue,
                                  gapscore=params.gapscore, logger=params.logger,
                                  max_target_seqs=params.max_target_seqs,
                                  cpu_threads=getattr(params, 'cpu_threads', None))
        search_sup.SeqBlast(blastp_params)

        # Keep the blast result files
        blast_files.append(blast_file)

        # Remove the mapped query file
        if not params.verbose:
            os.remove(query_file)

    return blast_files



#================================================================================================================================================================================================
# Filter(params) to place restrains on gap opens, align length etc.
#================================================================================================================================================================================================

def Filter(results, params):

    fields = []

    # save the first line for each candidate (best) and ignore the rest
    last_candidate = ''
    last_evalue = 1.0

    for field in results:

        # discard alignments if truncation at the ends (which are not treated as mismatches in blast) are more than 40%
        if field.querylength - (field.qend - field.qstart + 1) > 0.4 * field.querylength:
            continue

        # discard alignments if there are on average more than two gaps at each gap site
        #if field.gapopen > 0 and field.gaps > field.gapopen * 2:
            #continue

        # length filter applied here
        if field.length != -1:
            len_min, len_max = params.length

            if field.length < int(len_min) or field.length > int(len_max):
                continue

        # if verbosity asked, keep all the aligned segments
        if params.verbose:
            fields.append(field)
        # If verbosity not asked, only keep the alignment(s) with best/lowest E-value for each query each candidate
        else:
            if field.candidate != last_candidate:
                    fields.append(field)
                    last_candidate = field.candidate
                    last_evalue = field.log10_evalue
            elif field.candidate == last_candidate and field.log10_evalue == last_evalue:
                    fields.append(field)

    return fields

#================================================================================================================================================================================================
# read blast result files and store the results
#================================================================================================================================================================================================

def Readfile(params):

    # logging setup
    # logging.basicConfig(filename=params.logfile, level=logging.INFO, format='%(message)s')

    '''
    # first read in candidate info, including protein/gene names/length
    # dictionary to save protein, genes, length info for each candidate
    protein_info = {}

    # check if infofile file available
    if params.infofile and os.path.exists(params.infofile):
        f1 = open(params.infofile, 'r')

        # loop over lines to get genes length info
        for i, line in enumerate(f1):
            # skip the first header line
            if line:
                row = line[:-1].split('\t')

            if i == 0 and row:
                if row[0] == 'Entry' and row[1] == 'Gene names' and row[2] == 'Length':
                    continue
                else:
                    print 'Warning: Infofile file %s has wrong format! Skipping it...' % params.infofile
                    logging.warning('Infofile file %s has wrong format! Skipping it...', params.infofile)
                    break
            else:
                # from left to right: protein, genes, length
                # there may be more than one sequences corresponding to current protein
                for protein in row[1].split(' '):
                    protein_info[row[0]] = [protein, row[2]]

        f1.close()

    if not protein_info:
        print 'Warning: length info not available. Ignore the length factor of sequences and skip length filtering'
        logging.warning('Warning: length info not available. Ignore the length factor of sequences and skip length filtering')
    '''

    # split files first
    blastfiles = params.blastfiles

    # list for all the query results
    fields = []
    protlen_casefold = {key.casefold(): value for key, value in params.protlen.items()}

    while blastfiles:

        # temp list/set to read and record the query results in blast result files
        tem = []
        f = blastfiles.pop(0)
        f0 = open(f, 'r')

        for line in f0:
            stripped = line.rstrip('\r\n')
            if not stripped:
                continue

            # BLAST outfmt 7 omits the ``# Fields`` line when a query has zero
            # hits.  Treat that as a valid empty result instead of a malformed
            # file, while still validating a fields declaration when present.
            if stripped.startswith('# Fields:'):
                if stripped not in (blastformat7, blastformat7_with_bitscore):
                    params.logger.exception('Error: ' + f + ' wrong format! Exiting')
                    raise OSError('Error: ' + f + ' wrong format! Exiting')
                continue

            if stripped.startswith('#'):
                continue

            # save the results in class and list
            field = BlastSeq(stripped, dbsize=getattr(params, 'dbsize', None))
            candidate_id_map = getattr(params, 'candidate_id_map', None)
            if candidate_id_map is not None:
                if field.candidate not in candidate_id_map:
                    raise ValueError(f"BLAST returned an unknown internal candidate ID: {field.candidate!r}")
                field.candidate = candidate_id_map[field.candidate]

            # add length to class BlastSeq
            candidate_length = params.protlen.get(field.candidate)
            if candidate_length is None:
                candidate_length = protlen_casefold.get(field.candidate.casefold())
            if candidate_length is not None:
                field.set_prot_len(candidate_length)
            else:
                field.set_prot_len('-1')

            tem.append(field)

        # sort the list according to candidate and evalue and then save the best result for each candidate
        tem.sort(key=operator.attrgetter('candidate', 'log10_evalue'), reverse=False)

        # Filter(params) to place restrains on gap opens, align length etc.
        tem = Filter(tem, params)

        fields.extend(tem)

        f0.close()

        # if verbosity not asked, delete the blast result file
        if not params.verbose:
            os.remove(f)


    return fields

#================================================================================================================================================================================================
# calculate composite E-value for each candidate
#================================================================================================================================================================================================

def Calculate(results, params):

    # get the candidate set
    candidate = set()
    # candidate length dictionary to store the sequence length
    candidate_length = dict()

    # get the candidates that contains segments aligning well with queries
    for field in results:
        candidate.add(field.candidate)
        # get the length for each candidates
        candidate_length[field.candidate] = field.length

    # Accumulate in log space.  Large RNA query sets can contain hundreds of
    # components, making the direct E-value product underflow to zero and
    # creating false ties.  Log addition has the same mathematical ordering.
    candidate_log_evalue = dict.fromkeys(candidate, math.log10(params.evalue))

    # candidate average identity dictionary to store the average alignment identity info
    # initialized to 0
    candidate_ave_identity = dict.fromkeys(candidate, 0.0)
    # normlized by the total length of query sequences.
    norm_factor = 0.0

    # Index alignments once.  The original implementation rescanned every
    # alignment for every query, which becomes prohibitively slow for RNA maps
    # that yield hundreds of traced components.  Grouping by query preserves
    # the exact minimum-E-value calculation while making the work linear in
    # the number of alignments.
    results_by_query = {}
    for field in results:
        results_by_query.setdefault(field.query, []).append(field)

    # loop over query to get the composite E-value and average identity % for each candidate
    # query_info contains query name, length, file name and number of X for all the query sequences
    for query in params.query_info:

        # temp evalue dictionary to get the lowest E-value from all the alignments between each candidate and current query
        # initialize the composite E-value to the cutoff E-value
        tem_evalue = dict.fromkeys(candidate, math.log10(params.evalue))
        # temp identity dictionary to get the highest identity from all the alignments between each candidate and current query
        # initialize to 0
        tem_identity = dict.fromkeys(candidate, 0.0)

        query_fields = list(results_by_query.get(query[0], ()))
        if params.reverse:
            query_fields.extend(results_by_query.get(query[0] + '_rev', ()))

        for field in query_fields:
            if field.log10_evalue < tem_evalue[field.candidate]:
                tem_evalue[field.candidate] = field.log10_evalue
                tem_identity[field.candidate] = field.identity

        # loop over all candidate to update the composite E-value and ave_identity
        for icandidate in candidate_log_evalue:
            candidate_log_evalue[icandidate] += tem_evalue[icandidate]

            # apply the length factor here
            if candidate_length[icandidate] > 0:
                candidate_log_evalue[icandidate] += math.log10(
                    1.0 * candidate_length[icandidate] / 1000.0
                )
            # if length factor not available, ignore it.
            #else:
                #candidate_evalue[icandidate] *= 1.0

            # if there are segments aligned to query
            # print(tem_identity[icandidate])
            if tem_identity[icandidate] >= 1.0:
                candidate_ave_identity[icandidate] += tem_identity[icandidate]
            # if there is no aligned alignments, use half of the query length (exclude X) instead (usually blastp would fail if there is only 50% identity)
            else:
                # query contains query name, length, file name and number of X for the query
                candidate_ave_identity[icandidate] += 0.5 * (query[1] - query[3])


        norm_factor += query[1] - query[3]


    for icandidate in candidate_log_evalue:
        candidate_ave_identity[icandidate] /= norm_factor

    # save and sort the composite Evalue in results
    for field in results:
        log_evalue = candidate_log_evalue[field.candidate]
        # Keep the float attribute for result consumers. Very small
        # values may be zero here, but sorting and reporting use the log value.
        field.set_composite_evalue(linear_evalue(log_evalue))
        field.set_composite_log10_evalue(log_evalue)
        field.set_ave_identity(candidate_ave_identity[field.candidate])

    results.sort(key=operator.attrgetter('composite_log10_evalue', 'candidate'), reverse=False)

    return results


#================================================================================================================================================================================================
# write output files
#================================================================================================================================================================================================

def Writefile(fields, params):
    """
    Write the results of the search to files.

    Parameters:
        fields (List[Field]): a list of field objects representing the search results
        params (Namespace): a namespace object containing the search parameters

    Returns:
        None
    """

    # logging setup
    # logging.basicConfig(filename=params.logfile, level=logging.INFO, format='%(message)s')


    # create the output file
    # out_align = f"{params.outdir}{params.outfile}_alignment.txt"
    # out_result = f"{params.outdir}{params.outfile}_summary.txt"
    out_align = os.path.join(params.outdir, params.outfile + '_alignment.txt')
    out_result = os.path.join(params.outdir, params.outfile + '_summary.txt')

# ???????????STOP?????????????======================================================

    # write log/alignment results to file
    fout = open(out_align, 'w')
    fout1 = open(out_result, 'w')


    # save the brief summary in this list
    brief_field = []
    last_candidate = None
    #last_query = None
    ranking = 0

    for i, field in enumerate(fields):
        composite_text = format_log10_evalue(field.composite_log10_evalue)

        # By default only keep the first 100 most promising candidates
        # may add an argument to specify a number by the user
        if ranking >= params.num_candidate:
            break

        # the first candidate is most promising
        if i == 0:
            #print '#############################################'

            # check the best candidate to see if it satisfies the criteria
            if field.ave_identity < 0.7:
                # print('Warning: No high identity (> 70%) protein candidates identified! You may want to optimize your query sequences first, otherwise the results may not be reliable.\n', flush=True)
                params.logger.warning('Warning: No high identity (> 70%) protein candidates identified! You may want to optimize your query sequences first, otherwise the results may not be reliable.\n')

            out = 'Most likely candidate:\t' + field.candidate + '\nQuery set identity:\t{0:.0%}\nSequence length:\t'.format(field.ave_identity) +\
            str(field.length) + '\nComposite E-value:\t' + composite_text + '\n\n'
            out += 'The candidates have been ranked by composite E-value, and the top candidates are shown in the graph in the pop-up window. Presence of a statistically significant "gap" in composite E-values between the \
top candidate and all other candidates indicates a high confidence identification. The aligned segments can be used for further verification/model building.'

            # print out to screen and log file
            # print(out, flush=True)
            params.logger.info(out)

            #print('\nBrief summary of the top candidates:\n', file=fout2)

            # print header to log/alignments files
            # fout1.write('Ranking, candidate, length, ave_identity %, composite_E')
            print('Ranking, candidate, length, ave_identity %, composite_E, composite_log10_E', file=fout1)

            # check if gaps should be printed or not
            if params.gapscore == [32767, 32767]:
                print('Ranking, candidate, query, identity, query length, query_start, query_end, candidate_start, candidate_end, candidate_length, aligned_query, aligned_candidate, E-value, composite_E-value', file=fout)
            else:
                print('Ranking, candidate, query, identity, query length, num_gaps, query_start, query_end, candidate_start, candidate_end, candidate_length, aligned_query, aligned_candidate, E-value, composite_E-value', file=fout)


        # get the brief summary of all the promising candidates
        if field.candidate != last_candidate:
            ranking += 1
            brief_field.append(field)
            last_candidate = field.candidate

            print('', file=fout)

            # write brief summary to files.
            print(
                str(ranking) + '\t' + field.candidate + '\t' + str(field.length)
                + '\t' + '{0:.0%}\t{1}\t{2:.12f}'.format(
                    field.ave_identity,
                    composite_text,
                    field.composite_log10_evalue,
                ),
                file=fout1,
            )


        #if not (field.candidate == last_candidate and field.query == last.query):
        # revser mapping of aligned sequences from blast codes to degenerate groups. More readable for the users
        # for k, v in search_sup.aaRevMapping.items():
        #         field.sseq = field.sseq.replace(k, v)
        #         field.qseq = field.qseq.replace(k, v)

        # write detailed alignments info to files
        if params.gapscore == [32767, 32767]:
            print(str(ranking)+'\t'+field.candidate+'\t'+field.query+'\t'+str(field.identity)+'\t'+str(field.querylength)+'\t'+str(field.qstart)+'\t'+str(field.qend)+\
            '\t'+str(field.sstart)+'\t'+str(field.send)+'\t'+str(field.length)+'\t'+field.qseq+'\t'+field.sseq+'\t'+'{0:1.1e}'.format(field.evalue)+'\t'+composite_text,
            file=fout)
        else:
            print(str(ranking)+'\t'+field.candidate+'\t'+field.query+'\t'+str(field.identity)+'\t'+str(field.querylength)+'\t'+str(field.gaps)+'\t'+str(field.qstart)+'\t'+str(field.qend)+\
            '\t'+str(field.sstart)+'\t'+str(field.send)+'\t'+str(field.length)+'\t'+field.qseq+'\t'+field.sseq+'\t'+'{0:1.1e}'.format(field.evalue)+'\t'+composite_text,
            file=fout)

    fout.close()
    fout1.close()
    #fout2.close()

    # print('\nFor a brief summary of candidates\' information and their alignment details, please refer to %s and %s, respectively.' % (out_result, out_align) , flush=True)
    # print('********************************************************************', flush=True)

    # print to log file
    params.logger.info('')
    params.logger.info('For a brief summary of candidates\' information and their alignment details, please refer to %s and %s, respectively.', out_result, out_align)
    params.logger.info('')

    return brief_field


#================================================================================================================================================================================================
# plot the figure of final results
#================================================================================================================================================================================================

def Plot(fields, params):
    if len(fields) == 0:
        params.logger.warning('No promising candidates identified! No plot will be generated.')
        return
    elif len(fields) == 1:
        params.logger.warning('Only one promising candidate identified! No plot will be generated.')
        return

    # Save the data in lists
    identity_list = []
    logE_list = []

    # Show results of the first 20 candiates at most
    max_candidates = min(20, len(fields))

    for _, field in enumerate(fields[:max_candidates]):
        identity_list.append(field.ave_identity * 100.0)
        try:
            logE_list.append(field.composite_log10_evalue)
        except (AttributeError, ValueError):
            logE_list.append(-100)  # Append a default value if log is undefined

    # plot the figures to show average identity/composite E-value of promising candidates
    # set x axis based on the data lists
    m = len(identity_list)
    xaxis = range(1, m+1)

    plt.figure(1, figsize=(8, 6))

    ax1 = plt.subplot(211)
    ax1.scatter(xaxis, logE_list, s=25)

    # show the composite E-value gap between the first candidate and the rest ones
    #line1 = array([logE_list[0] for i in xaxis])
    line1 = [logE_list[0]] * m
    ax1.plot(xaxis, line1, color='pink',  linestyle='--')
    #line2 = array([logE_list[1] for i in xaxis])
    line2 = [logE_list[1]] * m
    ax1.plot(xaxis, line2, color='pink',  linestyle='--')

    ax1.set_xlabel('Ranking of candidates', fontsize=12)
    ax1.set_xlim(0.5, m + 0.5)
    ax1.set_xticks(range(1, m+1, 1))
    #ax1.set_ylim(-0, -50)
    #ax1.set_yticks(range(-0, -50, -10))
    #ax1.invert_yaxis()
    ax1.set_ylabel('Log of composite E-value', fontsize=12)
    ax1.axis('tight')
    plt.title('Results for the first {0} candidates'.format(str(m)), fontsize=12)


    # plot two sub figures
    ax2 = plt.subplot(212)
    ax2.scatter(xaxis, identity_list, s=25)

    # show the identity gap between the first candidate and the rest ones
    #line1 = array([identity_list[0] for i in xaxis])
    line1 = [identity_list[0]] * m
    ax2.plot(xaxis, line1, color='paleturquoise',  linestyle='--')

    # get the second highest identity statistics and draw a line
    max2_identity = max(identity_list[1:])
    line2 = [max2_identity] * m
    ax2.plot(xaxis, line2, color='paleturquoise',  linestyle='--')

    # set axis ticks
    ax2.set_xlim(0.5, m + 0.5)
    ax2.set_xticks(range(1, m+1, 1))
    ax2.set_xlabel('Ranking of candidates', fontsize=12)
    #ax2.set_ylim(40, 100)
    #ax2.set_yticks(range(40, 110, 10))
    ax2.set_ylabel('Query set identity (%)', fontsize=12)
    ax2.axis('tight')

    plt.subplots_adjust(hspace=0.3)


    # set figure name
    fig_name = params.outfile + '_results.png'

    plt.savefig(os.path.join(params.outdir, fig_name), dpi=300)
    params.logger.info('The figure of the top candidates has been saved as %s in your output folder.', fig_name)
    #plt.show()


    '''
    try:
        subprocess.call(['xdg-open', fig_name])
    except OSError:
        print '\nCryoID failed to open the png image. You may open it yourself in your folder.'
    '''

#================================================================================================================================================================================================
# process blast results and generate the final searching results
#================================================================================================================================================================================================
def Process(params):

    # Read the blast results
    results = Readfile(params)

    # Calculate compounted E-value
    results = Calculate(results, params)

    # Write output files
    results = Writefile(results, params)

    return results


#================================================================================================================================================================================================
# to search the queries aganist the candidate pool database
#================================================================================================================================================================================================
def Search(params):
    # Set up logging
    # logfile = f"{params.outdir}{params.outfile}.log"
    # logfile = os.path.join(params.outdir, f"{params.outfile}.log")
    # logging.basicConfig(filename=logfile, level=logging.INFO, format="%(message)s")

    logger = params.logger

    # Log the start time and command
    cmd = " ".join(sys.argv[1:])
    start_time = str(datetime.datetime.now())[:-7]
    logger.info('-'*75)
    logger.info(f"Starting job search_pool {cmd} ({start_time}):")
    logger.info('')
    # print(f"Starting job search_pool {cmd} ({start_time}):\n")

    # start timing
    job_start = time.time()

    # Get candidate protein sequences and their length
    # if params.candidate_pool.name.endswith((".txt", ".list")):
    #     retrieve_params = Namespace(candidate_pool=params.candidate_pool.name, logger=logger)
    #     seqfile_name = search_sup.RetrievSeq(retrieve_params)
    # elif params.candidate_pool.name.endswith(".fasta"):
    #     seqfile_name = params.candidate_pool.name

    params.outfile = str(params.outfile)
    params.outdir = str(params.outdir)
    seqfile_name = str(params.candidate_pool)

    # Get the length of each sequence
    protlen = {}
    candidate_ids = {}
    for index, record in enumerate(SeqIO.parse(seqfile_name, "fasta"), start=1):
        if record.id in protlen:
            raise ValueError(f"Duplicate candidate FASTA identifier: {record.id!r}")
        candidate_ids[record.id] = f"cryoid_seq_{index:08d}"
        protlen[record.id] = len(record.seq)              # eg. record.id = "tr|Q8IKL5|Q8IKL5_PLAF7", record.seq = "MNIRLIKKIGINY...ESKWYSFWEENA"

    # Generate a local blastp database from the (degenerate) candidate sequences
    DB_params = Namespace(seqfile=seqfile_name, logger=logger, verbose=params.verbose,
                          candidate_ids=candidate_ids)
    dbsize, dbname = search_sup.DBGenerate(DB_params)

    # Map the query sequences to files for blastp search
    query_params = Namespace(queryfile=params.queryfile, reverse=params.reverse, verbose=params.verbose, logger=logger)
    query_info = search_sup.QueryMap(query_params)

    # Blast query sequences against local database
    Blast_params = Namespace(query_info=query_info, dbsize=dbsize, dbname=dbname, evalue=params.evalue, gapscore=params.gapscore, logger=logger, verbose=params.verbose, max_target_seqs=len(protlen))
    Blast_params.cpu_threads = getattr(params, 'cpu_threads', None)
    blast_files = Blast(Blast_params)

    # Interpret the blast results
    if len(blast_files):
        # Process blast results and generate the final searching results
        process_params = Namespace(blastfiles=blast_files, candidate_pool=params.candidate_pool, protlen=protlen, reverse=params.reverse, length=params.length, queryfile=params.queryfile, outfile=params.outfile, outdir=params.outdir,
                                   evalue=params.evalue, verbose=params.verbose, gapscore=params.gapscore, num_candidate=params.num_candidate, query_info=query_info, logger=logger,
                                   dbsize=dbsize, candidate_id_map={internal: original for original, internal in candidate_ids.items()})
        results = Process(process_params)

        # Log the time used for the job
        job_end = time.time()
        logger.info(f"Job complete in {job_end - job_start:.1f} seconds!")
        params.logger.info('')
        # print(f"Job complete in {job_end - job_start:.1f} seconds!", flush=True)

        # Plot the figure
        # if params.verbose:
        try:
            plot_params = Namespace(outfile=params.outfile, outdir=params.outdir, logger=params.logger,)
            Plot(results, plot_params)
        except Exception:
            logger.exception("Failed to plot final figure.")
            # print('Failed to plot final figure.', flush=True)

    else:
        logger.error('No blast results found. Please check your input.')
        raise OSError('No blast results found. Exiting.')

    if not params.verbose:
        query_dir = os.path.join(os.path.dirname(params.queryfile), 'query_files')
        shutil.rmtree(query_dir)

    params.logger.info('-'*75)

    return results[0].candidate if results else None
