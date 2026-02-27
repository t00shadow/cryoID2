import argparse
import configparser
import logging
import pathlib
import textwrap

# from D_BLAST_searching.cryoID_v2.search_pool import Search as blast_search
# from Alphafold_fetch.struct_fetch_v4 import alphafold_fetch


class CryoID_v2:
    def __init__(self):
        self.prog = 'CryoID_v2'

        self.config = configparser.ConfigParser()
        self.config.read('config.ini')
        self.profile = 'DEFAULT'

        self.args = None

        self.data_dir = pathlib.Path() / 'Data'

        # configure logger
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)  
        std_out_hdlr = logging.StreamHandler()
        std_out_hdlr.setLevel(logging.INFO)
        file_hdlr = logging.FileHandler(self.prog+'.log')
        file_hdlr.setLevel(logging.INFO)
        file_hdlr.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p'))
        self.logger.addHandler(std_out_hdlr)
        self.logger.addHandler(file_hdlr)

    def parser(self):
        mainparser = argparse.ArgumentParser(
                    prog=self.prog,
                    description='%(prog)s: description here',
                    usage='%(prog)s <subcommand> [options]',
                    formatter_class=argparse.RawDescriptionHelpFormatter,
                    )
        subparsers = mainparser.add_subparsers(
            title='subcommands',
            description=textwrap.dedent('''
                profile: Load user config profile
                predict: Predict protein seqs from map
                blast: Perform BLAST search
                alphafold: Fetch Alphafold structures
            '''),
            help='Use %(prog)s <subcommand> -h for more information on a specific subcommand',
            dest='subcommand',
        )

        # -----Subparser for loading user profile-----
        profile_parser = subparsers.add_parser(
            'profile',
            prog=self.prog+' profile',
            description='Load user config profile',
            usage='%(prog)s <profile_name>',
            aliases=['p'],
        )
        profile_parser.add_argument(
            'name',
            type=str,
            help='Load user config profile',
            metavar='profile',
        )
        profile_parser.set_defaults(func=self.load_profile)

        # -----Subparser for blast-----
        blast_parser = subparsers.add_parser('blast',
                                            prog=self.prog+' blast',
                                            description='Perform BLAST search',
                                            usage='%(prog)s -q queries.pdb -p candidate_pool.fasta',
                                            aliases=['b'],
                                            )
        blast_parser.add_argument(
            '-q',
            '--queryfile',
            type=pathlib.Path,
            required=True,
            help='Queries (.fasta/.pdb file) to be searched',
        )
        blast_parser.add_argument(
            '-p',
            '--candidate_pool',
            type=pathlib.Path,
            required=True,
            help='Candidate protein/rna sequence pool file in fasta format',
        )
        blast_parser.add_argument(
            '-b', 
            '--base', 
            action='store_true', 
            default=False,
            help='Use Blastn instead of Blastp to search for nucleotide sequences',
            )
        blast_parser.add_argument(
            '-o',
            '--outfile',
            type=pathlib.Path,
            default=None,
            help='Output file basename. Default value is the same as query file basename',
        )
        blast_parser.add_argument(
            '-od',
            '--outdir',
            type=pathlib.Path,
            default=None,
            help='Output file directory. Default value is the same as query file directory',
        )
        blast_parser.add_argument(
            '-r',
            '--reverse',
            action='store_true',
            default=False,
            help='If N/C terminal polarity of query sesquences are unknown, add -r flag',
        )
        blast_parser.add_argument(
            '-l',
            '--length',
            type=int,
            nargs=2,
            default=self._get_list('blast_length'),
            help='Set target sequence length filter for the final results. Note that this \
    argument takes two parameters seperated by a space. Default value is %(default)s for a \
    sequence length range of 1~10000'
        )
        blast_parser.add_argument(
            '-e',
            '--evalue',
            type=float,
            default=self.config.getfloat(self.profile, 'blast_evalue'),
            help='Expectation value (E) threshold for saving hits in blast search for \
    each query. Default %(default)s'
        )
        blast_parser.add_argument(
            '-g',
            '--gapscore',
            type=int,
            nargs=2,
            default=self._get_list('blast_gapscore'),
            help='Penalty score pairs for gapopen and gapextend. Note that this argument \
    takes two parameters seperated by a space. Default value is %(default)s for ungapped \
    searching. You may change it to 15 3 for gapped searching'
        )
        blast_parser.add_argument(
            '-n',
            '--num_candidate',
            type=int,
            default=self.config.getint(self.profile, 'blast_candidate_num'),
            help='Maximum number of candidates in the final result lists. Default %(default)s',
        )
        blast_parser.add_argument(
            '-v',
            '--verbose',
            action='store_true',
            default=False,
            help='Keep simplified query sequence files and all significant blast alignments'
        )
        # blast_parser.set_defaults(func=self.blast_search)

        # -----Subparser for alphafold-----
        alphafold_fetch_parser = subparsers.add_parser(
            'alphafold',
            prog=self.prog+' alphafold',
            description='Fetch Alphafold structures',
            usage='%(prog)s -f <csv_file> -m <map_file>',
            aliases=['a'],
            )
        alphafold_fetch_parser.add_argument(
            '-f',
            '--file',
            required=True,
            type=pathlib.Path,
            help='CSV file containing the list of sequences',
            )
        alphafold_fetch_parser.add_argument(
            '-m',
            '--map',
            required=True,
            type=pathlib.Path,
            help='Map file',
            metavar='map',
            )
        alphafold_fetch_parser.add_argument(
            '-n',
            '--nrows',
            type=int,
            default=1,
            help='Number of CSV rows to process (with non-empty gene_id + sequence), default: %(default)s',
            metavar='nrows',
            )
        alphafold_fetch_parser.add_argument(
            '-o',
            '-output',
            type=pathlib.Path,
            default=self.data_dir,
            help='Output directory',
            metavar='outdir',
            )
        # alphafold_fetch_parser.set_defaults(func=alphafold_fetch)

        args = mainparser.parse_args()

        # Check args
        # -----BLAST args-----
        # check args : query file format, candidate_pool file format, evalue, length range, gapscore, output name
        if args.subcommand == 'blast' or args.subcommand == 'b':
            if not (args.queryfile.suffix == '.fasta' or args.queryfile.suffix == '.pdb'):
                raise ValueError('Query file must be in .fasta or .pdb format.')
                
            if not (args.candidate_pool.suffix == '.fasta'):
                raise ValueError('Sequence pool file format is not recognized.')
            
            if args.evalue <= 0.0 or args.evalue >= 1000.0:
                raise ValueError('E-value should be between 0.0 and 1000.0.')
            
            if not (args.length[0] > 0 and args.length[0] < args.length[1]):
                raise ValueError('Please input a valid length range.')
            
            if not (args.gapscore == [32767, 32767] or args.gapscore == [15, 3] or args.gapscore == [13, 3]):
                raise ValueError('Please input acceptable gapscores: [32767, 32767], [15, 3], or [13, 3].')
            
            if args.num_candidate <= 0:
                raise ValueError('The number of candidates must be a positive integer.')
            
            if args.outfile is None:
                args.outfile = args.queryfile.stem

            if args.outdir is None:
                args.outdir = args.queryfile.parent

        return args
    
    def run(self):
        if self.args is None:
            raise ValueError('No arguments provided.')
        self.args.func(self.args)   

    def load_profile(self, args):
        profile_name = args.name
        if not self.config.has_section(profile_name):
            raise ValueError(f'Profile {profile_name} not found in config file.')
        self.profile = profile_name
        self.logger.info(f'Profile {profile_name} loaded.')

    def _get_list(self, key:str, type_func=int):
        value = self.config.get(self.profile, key)
        return [type_func(x) for x in value.split(',')]


if __name__ == '__main__':
    prog = CryoID_v2()
    prog.arg = prog.parser()
    prog.run()