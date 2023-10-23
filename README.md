<!--
Copyright (c) 2023 Ernst Strüngmann Institute (ESI) for Neuroscience
in Cooperation with Max Planck Society
SPDX-License-Identifier: CC-BY-NC-SA-1.0
-->

# oephys2nwb: Export Open Ephys binary data to NWB 2.x

## Summary

This package can be used to export data saved in
[Open Ephys binary format](https://open-ephys.github.io/gui-docs/User-Manual/Recording-data/Binary-format.html)
(the default for the Open Ephys GUI) to the [NWB 2.0 format](https://www.nwb.org/).
At the time of writing the
[NWBFormat plugin](https://open-ephys.github.io/gui-docs/User-Manual/Recording-data/NWB-format.html)
of Open Ephys only supports the deprecated NWB 1.0 format specification. The present
package is developed and maintained at the
[Ernst Strüngmann Institute (ESI) gGmbH for Neuroscience in Cooperation with Max Planck Society](https://www.esi-frankfurt.de/>)
and released free of charge under the
[BSD 3-Clause "New" or "Revised" License](https://en.wikipedia.org/wiki/BSD_licenses#3-clause_license_(%22BSD_License_2.0%22,_%22Revised_BSD_License%22,_%22New_BSD_License%22,_or_%22Modified_BSD_License%22)).

## Installation

The package can be installed with `pip`

```shell
pip install esi-oephys2nwb
```

To get the latest development version, simply clone our GitHub repository and
(optionally) create a dedicated conda development environment:

```shell
git clone https://github.com/esi-neuroscience/oephys2nwb.git
cd oephys2nwb/
python setup.py --version
conda env create -f oephys2nwb-dev.yml
conda activate oephys2nwb-dev
pip install -e .
```

## Usage

We recommend setting up a dedicated conda environment for `oephys2nwb`. For general
information about conda, please refer to the [official documentation](https://docs.conda.io/en/latest/).

```shell
conda create -n oephys2nwb-env python=3.8 pip
conda activate oephys2nwb-env
pip install esi-oephys2nwb
```

Alternatively, we provide a conda environment file to automatically install all
required dependencies

```shell
wget https://raw.githubusercontent.com/esi-neuroscience/oephys2nwb/main/oephys2nwb.yml
conda env create -f oephys2nwb.yml
conda activate oephys2nwb
```

Once the package is installed the exporter can be used either embedded in Python
code or as a stand-alone command line utility.

### Command Line

Activate the conda environment the package was installed in and invoke the exporter
as follows

```shell
conda activate oephys2nwb-env
python -m oephys2nwb -i /path/to/recordingDir -o /path/to/outputFile.nwb
```

Calling `oephys2nwb` without arguments prints the function help

```shell
python -m oephys2nwb
```

### Python Code

Alternatively, the package can be used like any other Python module

```python
from oephys2nwb import export2nwb

input = "/path/to/recordingDir"
output = "/path/to/outputFile.nwb"

export2nwb(input, output)
```

## Documentation and Contact

To report bugs or ask questions please use our
[GitHub issue tracker](https://github.com/esi-neuroscience/oephys2nwb/issues).
More usage details and background information is available in our
[online documentation](https://esi-oephys2nwb.readthedocs.io/en/latest/).
