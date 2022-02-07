# -*- coding: utf-8 -*-
#
# Simple script for testing exporter w/o pip-installing it
#

# Builtin/3rd party package imports
import numpy as np
from pynwb import NWBHDF5IO

# Add acme to Python search path
import os
import sys
pkg_path = os.path.abspath(".." + os.sep + "..")
if pkg_path not in sys.path:
    sys.path.insert(0, pkg_path)

# Import package
from oephys2nwb import export2nwb

# Prepare code to be executed using, e.g., iPython's `%run` magic command
if __name__ == "__main__":

    # Test stuff within here...
    dataDir = "../testrecording_2021-11-09_17-06-14/"
    outFile = "/mnt/hpx/home/fuertingers/tt2.nwb"

    export2nwb(dataDir, outFile, trial_markers=[3000, 3090])

    nwbio = NWBHDF5IO(outFile, "r", load_namespaces=True)
    nwbfile = nwbio.read()

    tstart = nwbfile.epochs['start_time'].data[()]




