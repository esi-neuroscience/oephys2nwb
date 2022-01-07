# -*- coding: utf-8 -*-
#
# Main package initializer
#

# Builtin/3rd party package imports
import subprocess
import warnings
import inspect
from pkg_resources import get_distribution, DistributionNotFound

# Get package version: either via meta-information from egg or via latest git commit
try:
    __version__ = get_distribution("esi-oephys2nwb").version
except DistributionNotFound:
    proc = subprocess.Popen("git describe --always",
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, shell=True)
    out, err = proc.communicate()
    if proc.returncode != 0:
        proc = subprocess.Popen("git rev-parse HEAD:oephys2nwb/__init__.py",
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, shell=True)
        out, err = proc.communicate()
        if proc.returncode != 0:
            msg = "WARNING: oephys2nwb is not installed in site-packages nor cloned via git. " +\
                "Please consider obtaining its sources from supported channels. "
            warnings.showwarning(msg, ImportWarning, __file__, inspect.currentframe().f_lineno)
            out = "-999"
    __version__ = out.rstrip("\n")

# Import local modules
from . import export2nwb as e2nwb
from .export2nwb import *

# Manage user-exposed namespace imports
__all__ = []
__all__.extend(e2nwb.__all__)
