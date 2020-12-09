# -*- coding: utf-8 -*-
"""
This is a skeleton file that can serve as a starting point for a Python
console script. To run this script uncomment the following lines in the
[options.entry_points] section in setup.cfg:

    console_scripts =
         fibonacci = inferface.skeleton:run

Then run `python setup.py install` which will install the command `fibonacci`
inside your current environment.
Besides console scripts, the header (i.e. until _logger...) of this file can
also be used as template for Python modules.

Note: This skeleton file can be safely removed if not needed!
"""

import argparse
import sys
import logging

from inferface import __version__

__author__ = "Stefanie Stoppel"
__copyright__ = "Stefanie Stoppel"
__license__ = "mit"

from inferface.embeddings import create_face_embeddings
from inferface.run import run_stage
from inferface.utils import write_to_csv

_logger = logging.getLogger(__name__)


def parse_args(args):
    """Parse command line parameters

    Args:
      args ([str]): command line parameters as list of strings

    Returns:
      :obj:`argparse.Namespace`: command line parameters namespace
    """
    parser = argparse.ArgumentParser(
        description="Just a Fibonacci demonstration")
    parser.add_argument(
        "-s",
        "--stage",
        dest="stage",
        type=str,
        choices=['train', 'test', 'preprocess'],
        default='train'
        )
    parser.add_argument(
        "-c",
        "--checkpoint",
        dest="checkpoint",
        default=''
    )
    parser.add_argument(
        "--version",
        action="version",
        version="InferFace {ver}".format(ver=__version__))
    parser.add_argument(
        "-v",
        "--verbose",
        dest="loglevel",
        help="set loglevel to INFO",
        action="store_const",
        const=logging.INFO)
    parser.add_argument(
        "-vv",
        "--very-verbose",
        dest="loglevel",
        help="set loglevel to DEBUG",
        action="store_const",
        const=logging.DEBUG)
    return parser.parse_args(args)


def setup_logging(loglevel):
    """Setup basic logging

    Args:
      loglevel (int): minimum loglevel for emitting messages
    """
    logformat = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
    logging.basicConfig(level=loglevel, stream=sys.stdout,
                        format=logformat, datefmt="%Y-%m-%d %H:%M:%S")


def main(args):
    """Main entry point allowing external calls

    Args:
      args ([str]): command line parameter list
    """
    args = parse_args(args)
    setup_logging(args.loglevel)

    if args.stage == 'test' and (args.checkpoint is None or args.checkpoint == ''):
        _logger.error("Argument --stage requires argument --checkpoint to be set to a valid checkpoint path.")
        sys.exit(1)

    if args.stage == 'preprocess':
        embeddings, no_faces_found = create_face_embeddings('/home/steffi/dev/independent_study/fairface_margin_025/val')
        write_to_csv(embeddings,
                     '/home/steffi/dev/independent_study/fairface_margin_025/embeddings/val.csv')
        write_to_csv(no_faces_found,
                     '/home/steffi/dev/independent_study/fairface_margin_025/embeddings/val_no_faces.csv')

    if args.stage == 'train':
        _logger.info("Starting training stage...")
        run_stage(stage=args.stage)
    elif args.stage == 'test':
        _logger.info("Starting testing stage...")
        run_stage(stage=args.stage, checkpoint_path=args.checkpoint)

    _logger.info("Done.")


def run():
    """Entry point for console_scripts
    """
    main(sys.argv[1:])


if __name__ == "__main__":
    run()
