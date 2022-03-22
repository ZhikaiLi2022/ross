import argparse

import ross
from ross import pipeline
from ross import TODODIR, INFODIR, INPDIR, OUTDIR


def main():

#####################################################################
# Initiate parser
#

    parser = argparse.ArgumentParser(
                                     description="ROSS: Automated Extraction of period of EEBs", 
                                     prog='ross',
    )
    parser.add_argument('-version', '--version',
                        action='version',
                        version="%(prog)s {}".format(ross.__version__),
                        help="Print version number and exit.",
    )

