#!/usr/bin/env python2
"""Swap faces in image files
  Usage:
    face scan [options] [--] <FILES>...
    face swap [options] [--] heads <HEAD_DIR> faces <FACE_DIR>

  Options:
    -h --help     show this help message and exit
    --version     show version and exit
    -q --quiet    little output
    -v --verbose  more output

"""
__version__ = 0.2
import sys

import docopt

#from . import *
#from .cli import main
from cli import main

kwargs = docopt.docopt(__doc__, version=__version__)

sys.exit(main(**kwargs))
