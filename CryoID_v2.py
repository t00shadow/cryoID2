import argparse
import configparser
import pathlib
import textwrap
from collections import namedtuple

# from Alphafold_fetch.struct_fetch_v4 import alphafold_fetch


class CryoID_v2:
    def __init__(self):
        self.prog = 'CryoID_v2'
        self.config = configparser.ConfigParser()
        self.config.read('config.ini')

        self.user_input = self.argparse()
        print(self.user_input)
        # self.user_input.func(self.user_input)

    def argparse(self):
        parser = argparse.ArgumentParser(
                    prog=self.prog,
                    description='%(prog)s: description here',
                    usage='%(prog)s <subcommand> [options]',
                    formatter_class=argparse.RawDescriptionHelpFormatter,
                    )
        parser.add_argument(
            '-c',
            '--config',
            type=str,
            help='Load user config profile',
            metavar='profile',
        )
        subparsers = parser.add_subparsers(
            title='subcommands',
            description=textwrap.dedent('''
                predict: Predict protein seqs from map
                blast: Perform BLAST search
                alphafold: Fetch Alphafold structures
            '''),
            help='Use %(prog)s <subcommand> -h for more information on a specific subcommand',
        )

        # Subparser for alphafold
        alphafold_fetch_parser = subparsers.add_parser('alphafold',
                                                       prog=self.prog+' alphafold',
                                                       description='Fetch Alphafold structures',
                                                       usage='%(prog)s -f <csv_file> -m <map_file>',
                                                       )
        alphafold_fetch_parser.add_argument('-f',
                                            '--file',
                                            required=True,
                                            type=pathlib.Path, 
                                            help='CSV file containing the list of sequences',
                                            )
        alphafold_fetch_parser.add_argument('-m',
                                            '--map',
                                            required=True,
                                            type=pathlib.Path,
                                            help='Map file',
                                            metavar='map',
                                            )
        alphafold_fetch_parser.add_argument('-n',
                                            '--nrows',
                                            type=int,
                                            default=1,
                                            help='Number of CSV rows to process (with non-empty gene_id + sequence), default: %(default)s',
                                            metavar='nrows',
                                            )
        alphafold_fetch_parser.add_argument('-o',
                                            '-output',
                                            type=pathlib.Path,
                                            help='Output directory',
                                            metavar='outdir',
                                            )
        # alphafold_fetch_parser.set_defaults(func=alphafold_fetch)

        blast_parser = subparsers.add_parser('blast',
                                            prog=self.prog+' blast',
                                            description='Perform BLAST search',
                                            usage='%(prog)s blast -q queries.pdb -p candidate_pool.fasta',
                                            )
        

        args = parser.parse_args()

        return args
    

if __name__ == "__main__":
    CryoID_v2()