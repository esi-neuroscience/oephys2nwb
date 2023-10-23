#
# CLI setup
#
# Copyright © 2023 Ernst Strüngmann Institute (ESI) for Neuroscience
# in Cooperation with Max Planck Society
#
# SPDX-License-Identifier: BSD-3-Clause
#

import sys

from .export2nwb import clarg_parser, export2nwb

# Invoke command-line argument parse helper
args_dict = clarg_parser(sys.argv[1:])

# If `args_dict` is not `None`, call actual function
if args_dict:
    export2nwb(**args_dict)
