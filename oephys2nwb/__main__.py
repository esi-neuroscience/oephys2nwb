# -*- coding: utf-8 -*-
#
# CLI setup
#

import sys

from .export2nwb import clarg_parser, export2nwb

# Invoke command-line argument parse helper
args_dict = clarg_parser(sys.argv[1:])

# If `args_dict` is not `None`, call actual function
if args_dict:
    export2nwb(**args_dict)
