import argparse
import configparser
import logging
import pathlib
import sys
import textwrap

# from H_Backbone_tracing.A1_clustering import main as clustering
from .BLAST.search_pool import Search as blast_search
# from Alphafold_fetch.struct_fetch_v4 import main as alphafold_fetch

from . import __version__

CONFIG_FILE: pathlib.Path = pathlib.Path(__file__).resolve().parent / 'config.ini'


def get_list(configparser: configparser.ConfigParser, profile: str, key: str, type: type = int) -> list:
    '''Helper function to get list of values from config file given a key.
    The values in the config file should be separated by commas.'''
    value = configparser.get(profile, key)
    return [type(x) for x in value.split(',')]

class CryoID_v2:

    __instance = None

    # program name
    prog: str = 'CryoID'

    # version info
    version: str = __version__

    @staticmethod
    def get_instance():
        if CryoID_v2.__instance is None:
            CryoID_v2()
        return CryoID_v2.__instance
    
    def __init__(self):
        # singleton pattern to ensure only one instance of the program is created
        if CryoID_v2.__instance is not None:
            raise Exception("Only one instance of CryoID_v2 is allowed.")
        else:
            CryoID_v2.__instance = self

        # load config file and default profile
        self.configparser: configparser.ConfigParser = configparser.ConfigParser()
        self.configparser.read(CONFIG_FILE)
        self.profile: str = self.configparser.get('DEFAULT', 'profile', fallback='DEFAULT')

        # parsing command line arguments
        # subparsers
        self.subparsers_info: dict[str, str] = {
            'profile': 'Load user config profile',
            'predict': 'Predict protein seqs from map',
            'blast': 'Perform BLAST search',
            'alphafold': 'Fetch Alphafold structures'
        }
        self.parser: argparse.ArgumentParser = self.create_parser()
        self.args: argparse.Namespace = None

        # default data directory
        # self.data_dir: pathlib.Path = pathlib.Path() / 'Data'

        # configure logger
        self.logger: logging.Logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)  
        std_out_hdlr: logging.StreamHandler = logging.StreamHandler()
        std_out_hdlr.setLevel(logging.INFO)
        file_hdlr: logging.FileHandler = logging.FileHandler(self.prog+'.log')
        file_hdlr.setLevel(logging.INFO)
        file_hdlr.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p'))
        self.logger.addHandler(std_out_hdlr)
        self.logger.addHandler(file_hdlr)

    def create_parser(self) -> argparse.ArgumentParser:
        mainparser = argparse.ArgumentParser(
                    prog=self.prog,
                    description='%(prog)s: description here',
                    usage='%(prog)s <subcommand> [options]',
                    formatter_class=argparse.RawDescriptionHelpFormatter,
                    )
        mainparser.add_argument(
            '-v',
            '--version',
            action='version',
            version=f'%(prog)s {self.version}',
            help='Show program version and exit',
        )
        subparsers = mainparser.add_subparsers(
            title='subcommands',
            description=textwrap.dedent(
                '\n'.join([f'{k}: {v}' for k, v in self.subparsers_info.items()])
                ),
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
            'pname',
            type=str,
            help='User config profile name',
            metavar='profile',
        )
        profile_parser.set_defaults(func=self.load_profile)

        # -----Subparser for blast-----
        blast_parser = subparsers.add_parser('blast',
                                            prog=self.prog+' blast',
                                            description='Perform BLAST search',
                                            usage='%(prog)s -q queries.fasta -p candidate_pool.fasta',
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
            default=get_list(self.configparser, self.profile, 'blast_length'),
            help='Set target sequence length filter for the final results. Note that this \
    argument takes two parameters seperated by a space. Default value is %(default)s.'
        )
        blast_parser.add_argument(
            '-e',
            '--evalue',
            type=float,
            default=self.configparser.getfloat(self.profile, 'blast_evalue'),
            help='Expectation value (E) threshold for saving hits in blast search for \
    each query. Default %(default)s'
        )
        blast_parser.add_argument(
            '-g',
            '--gapscore',
            type=int,
            nargs=2,
            choices=((32767, 32767), (15, 3), (13, 3)),
            default=get_list(self.configparser, self.profile,'blast_gapscore'),
            dest='gapscore',
            metavar=('GAPOPEN', 'GAPEXTEND'),
            help='Penalty score pairs for gapopen and gapextend. Note that this argument \
    takes two parameters seperated by a space. Default value is %(default)s for ungapped \
    searching. You may change it to 15 3 or 13 3 for gapped searching'
        )
        blast_parser.add_argument(
            '-n',
            '--num_candidate',
            type=int,
            default=self.configparser.getint(self.profile, 'blast_candidate_num'),
            help='Maximum number of candidates in the final result lists. Default %(default)s',
        )
        blast_parser.add_argument(
            '-v',
            '--verbose',
            action='store_true',
            default=False,
            help='Keep simplified query sequence files and all significant blast alignments'
        )
        blast_parser.set_defaults(func=blast_search)

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
            # default=self.data_dir,
            help='Output directory',
            metavar='outdir',
            )
        # alphafold_fetch_parser.set_defaults(func=alphafold_fetch)

        return mainparser
    
    def parse(self, argv: list[str]) -> argparse.Namespace:
        args = self.parser.parse_args(argv)

        # Check args
        # -----BLAST args-----
        # check args : query file format, candidate_pool file format, evalue, length range, gapscore, output name
        if args.subcommand == 'blast' or args.subcommand == 'b':
            if not (args.queryfile.suffix == '.fasta' or args.queryfile.suffix == '.pdb'):
                raise ValueError('Query file must be in .fasta or .pdb format.')
            
            if not args.queryfile.exists(): 
                raise FileNotFoundError(f'Query file {args.queryfile} does not exist.')
                    
            if not (args.candidate_pool.suffix == '.fasta'):
                raise ValueError('Sequence pool file format is not recognized.')
            
            if not args.candidate_pool.exists():
                raise FileNotFoundError(f'Sequence pool file {args.candidate_pool} does not exist.')
            
            if args.evalue <= 0.0 or args.evalue >= 1000.0:
                raise ValueError('E-value should be between 0.0 and 1000.0.')
            
            if not (args.length[0] > 0 and args.length[0] < args.length[1]):
                raise ValueError('Please input a valid length range.')
            
            # if not (args.gapscore == [32767, 32767] or args.gapscore == [15, 3] or args.gapscore == [13, 3]):
            #     raise ValueError('Please input acceptable gapscores: [32767, 32767], [15, 3], or [13, 3].')
            
            if args.num_candidate <= 0:
                raise ValueError('The number of candidates must be a positive integer.')
            
            if args.outfile is None:
                args.outfile = args.queryfile.stem

            if args.outdir is None:
                args.outdir = args.queryfile.parent

        # -----Alphafold args-----
        if args.subcommand == 'alphafold' or args.subcommand == 'a':
            if not args.file.exists():
                raise FileNotFoundError(f'CSV file {args.file} does not exist.')
            
            if not args.map.exists():
                raise FileNotFoundError(f'Map file {args.map} does not exist.')

        # add logger to args 
        args.logger = self.logger

        self.args = args 

    def load_profile(self) -> None:
        '''Load user config profile.'''
        profile_name = self.args.pname

        # Load default profile 
        # This allows user to reset to default profile after loading other profiles
        default_names = ['DEFAULT', 'default', 'd', 'D', 'Default', 'def', 'Def']
        if profile_name in default_names:
            self.profile = 'DEFAULT'
            self.configparser.set('DEFAULT', 'profile', 'DEFAULT')
            self.logger.info('Default profile loaded.')
            return
        
        if not self.configparser.has_section(profile_name):
            raise configparser.NoSectionError(f'Profile [{profile_name}] not found in config file.')
        else:
            self.profile = profile_name
            self.configparser.set('DEFAULT', 'profile', profile_name)
            self.logger.info(f'Profile [{profile_name}] loaded.')

    def run(self) -> None:
        """Run the program with the provided arguments."""
        if self.args is None:
            raise ValueError('No arguments provided.')
        else:
            self.args.func(self.args)  

def main():
    prog = CryoID_v2()
    prog.parse(sys.argv[1:])
    prog.run()