#
# Simple script for testing exporter w/o pip-installing it
#
#
# Copyright © 2023 Ernst Strüngmann Institute (ESI) for Neuroscience
# in Cooperation with Max Planck Society
#
# SPDX-License-Identifier: BSD-3-Clause
#

# Builtin/3rd party package imports
import numpy as np
from pynwb import NWBHDF5IO

# Add acme to Python search path
import os
import sys
pkg_path = os.path.abspath("..")
if pkg_path not in sys.path:
    sys.path.insert(0, pkg_path)
import getpass

# Import package
from oephys2nwb import export2nwb

# Prepare code to be executed using, e.g., iPython's `%run` magic command
if __name__ == "__main__":

    # Test stuff within here...
    dataDir = f"/cs/home/{getpass.getuser()}/Data/oephys2nwb"
    testDir = os.path.join(dataDir, "modtestrecording_2021-11-09_17-06-14")
    # testDir = os.path.join(dataDir, "testrecording_2021-11-09_17-06-14")
    outFile = os.path.join(dataDir, "test_out.nwb")
    if os.path.isfile(outFile):
        os.unlink(outFile)

    export2nwb(testDir, outFile)
