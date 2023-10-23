#!/usr/bin/env python3
#
# Export binary OpenEphys data to NWB 2.x
#
# Copyright © 2023 Ernst Strüngmann Institute (ESI) for Neuroscience
# in Cooperation with Max Planck Society
#
# SPDX-License-Identifier: BSD-3-Clause
#

import os
import sys
import xml.etree.ElementTree as ET
import uuid
import json
import warnings
import subprocess
import numpy as np
from numpy.typing import ArrayLike, NDArray
from argparse import ArgumentParser, RawTextHelpFormatter
from dataclasses import dataclass, field
from typing import Optional, Union, List, Tuple, Dict, Any
from pathlib import Path
from datetime import datetime
from pydantic import validate_arguments
from pynwb import NWBFile, NWBHDF5IO
from hdmf.backends.hdf5.h5_utils import H5DataIO
from pynwb.ecephys import ElectricalSeries
from ndx_events import TTLs
from open_ephys.analysis import Session

__all__ = ["export2nwb"]


# Factory generating default unit-mapping dict for `EphysInfo` class
def _unitConversionMapping():
    return {"uV" : 1e-6, "mV" : 1e-3, "V" : 1.0}

@dataclass
class EphysInfo:
    """
    Local helper class for parsing OpenEphys XML/JSON meta-data
    """

    # Quantities the class is instantiated with
    data_dir : str
    session_description : Optional[str] = None
    identifier : Optional[str] = None
    session_id : Optional[str] = None
    session_start_time : Optional[datetime] = None
    experimenter : Optional[str] = None
    lab : Optional[str] = None
    institution : Optional[str] = None
    experiment_description : Optional[str] = None

    # All other attributes set during `__post_init__`
    settingsFile : str = field(default="settings.xml", init=False)
    jsonFile : str = field(default="structure.oebin")
    root : ET.Element = field(init=False)
    machine : str = field(init=False)
    device : str = field(init=False)
    experimentDir : str = field(init=False)
    recordingDirs : List = field(init=False)
    eventDirs : List = field(default_factory=lambda : [], init=False)
    eventDtypes : List = field(default_factory=lambda : [], init=False)
    xmlRecChannels : List = field(init=False)
    xmlRecGroups : List = field(init=False)
    xmlEvtChannels : List = field(init=False)
    xmlEvtGroups : List = field(init=False)
    sampleRate : float = field(default=None, init=False)
    recChannelUnitConversion : Dict = field(default_factory=_unitConversionMapping, init=False)

    def __post_init__(self) -> None:
        """
        Manager method invoking all required parsing helpers
        """

        self.data_dir = os.path.abspath(os.path.expanduser(os.path.normpath(self.data_dir)))
        if not os.path.isdir(self.data_dir):
            err = "Provided path {} does not point to an existing directory"
            raise IOError(err.format(self.data_dir))

        if self.session_description is None:
            self.session_description = os.path.basename(self.data_dir)

        self.process_xml()

        self.process_json()

        return

    def process_xml(self) -> None:
        """
        Read OpenEphys XML settings file and vet its contents
        """

        xmlFiles = []
        for cur_path, _, files in os.walk(self.data_dir):
            if self.settingsFile in files:
                xmlFiles.append(os.path.join(self.data_dir, cur_path, self.settingsFile))

        if len(xmlFiles) != 1:
            err = "Found {numf} {xmlfile} files in {folder}"
            raise ValueError(err.format(numf=len(xmlFiles), xmlfile=self.settingsFile, folder=self.data_dir))
        self.settingsFile = xmlFiles[0]

        basePath = Path(os.path.dirname(self.settingsFile))
        experimentDirs = [str(entry) for entry in basePath.iterdir() \
            if entry.is_dir() and entry.name.startswith("experiment")]
        if len(experimentDirs) != 1:
            err = "Found {numf} experiments in {folder}"
            raise ValueError(err.format(numf=len(experimentDirs), folder=self.data_dir))
        self.experimentDir = experimentDirs[0]

        self.recordingDirs = [str(entry) for entry in Path(self.experimentDir).iterdir() \
            if entry.is_dir() and entry.name.startswith("recording")]
        if len(self.recordingDirs) == 0:
            err = "No recording directory found"
            raise ValueError(err)
        if self.session_id is not None and len(self.recordingDirs) > 1:
            err = "Cannot use single session_id for {} recordings"
            raise ValueError(err.format(len(self.recordingDirs)))

        try:
            self.root = ET.parse(self.settingsFile).getroot()
        except Exception as exc:
            err = "Cannot parse {}, original error message: {}"
            raise ET.ParseError(err.format(self.settingsFile, str(exc)))

        self.date = self.xml_get("INFO/DATE").text
        self.machine = self.xml_get("INFO/MACHINE").text

        device = self.xml_get("SIGNALCHAIN/PROCESSOR").get("name")
        if device is None:
            err = "Invalid {} file: empty tag in element SIGNALCHAIN/PROCESSOR"
            raise ValueError(err.format(self.settingsFile))
        self.device = device.replace("Sources/", "")

        if self.session_start_time is None:
            try:
                self.session_start_time = datetime.strptime(self.date, "%d %b %Y %H:%M:%S")
            except ValueError:
                msg = "{xmlfile}: recording date in unexpected format: '{datestr}' " +\
                    "Please provide session start time manually (via keyword session_start_time)"
                raise ValueError(msg.format(xmlfile=self.settingsFile, datestr=self.date))

        if self.identifier is None:
            self.identifier = str(uuid.uuid4())

        if self.experimenter is None:
            self.experimenter = self.machine

        if self.machine.lower().startswith("esi-"):
            if self.institution is None:
                self.institution = "Ernst Strüngmann Institute (ESI) for Neuroscience " +\
                    "in Cooperation with Max Planck Society"
            if self.lab is None:
                self.lab = self.machine.lower().split("esi-")[1][2:5].upper() # get ESI lab code (HSV, LAU, FRI etc.)

        channels, groups = self.get_rec_channel_info()
        self.xmlRecChannels = channels
        self.xmlRecGroups = groups

        channels, groups = self.get_evt_channel_info()
        self.xmlEvtChannels = channels
        self.xmlEvtGroups = groups

        return

    def xml_get(self, elemPath : str) -> ET.Element:
        """
        Helper method that (tries to) fetch an element from settings.xml
        """
        elem = self.root.find(elemPath)
        if elem is None:
            xmlErr = "Invalid {} file: missing element {}"
            raise ValueError(xmlErr.format(self.settingsFile, elemPath))
        return elem

    def get_rec_channel_info(self) -> tuple[List, List]:
        """
        Helper method that fetches/parsers recording channels found in settings.xml
        """

        # Get XML element and fetch channels
        chanInfo = self.xml_get("SIGNALCHAIN/PROCESSOR/CHANNEL_INFO")
        chanList = []
        chanGroups = []
        for chan in chanInfo.iter("CHANNEL"):

            # Assign each channel to a group by creating a new XML tag: the
            # default group is "CH"
            chanName = chan.get("name")
            if chanName is None or chan.get("number") is None:
                err = "Invalid channel specification in {}"
                raise ValueError(err.format(self.settingsFile))
            if chanName.startswith("ADC"):
                chan.set("group", "ADC")
            else:
                chanParts = chanName.split("_")
                if len(chanParts) == 3:
                    chan.set("group", chanParts[1])
                else:
                    chan.set("group", "CH")
            chanList.append(chan)
            chanGroups.append(chan.get("group"))

        # Abort if no valid channels were found
        if len(chanList) == 0:
            err = "Found no valid channels in {} file"
            raise ValueError(err.format(self.settingsFile))

        return chanList, list(set(chanGroups))

    def get_evt_channel_info(self) -> tuple[List, List]:
        """
        Helper method that fetches/parsers event channels found in settings.xml
        """

        # Get XML element and fetch channels
        editor = self.xml_get("SIGNALCHAIN/PROCESSOR/EDITOR")
        chanList = []
        chanGroups = []

        for chan in editor.iter("EVENT_CHANNEL"):

            # Assign each channel to a group by creating a new XML tag: pay
            # special attention to TTL channels, default group is "EVT"
            chanName = chan.get("Name")
            if chanName is None or chan.get("Channel") is None:
                err = "Invalid event channel specification in {}"
                raise ValueError(err.format(self.settingsFile))
            if chanName.startswith("TTL"):
                chan.set("group", "TTL")
            else:
                chan.set("group", "EVT")
            chanList.append(chan)
            chanGroups.append(chan.get("group"))

        # Issue a warning in case no event channels were found
        if len(chanList) == 0:
            wrn = f"Found no valid event channels in {self.settingsFile} file"
            warnings.warn(wrn)

        return chanList, list(set(chanGroups))


    def process_json(self) -> None:
        """
        Helper method that fetches info from OpenEphys JSON file and ensures
        that contents match up with values obtained from settings.xml
        """

        for recDir in self.recordingDirs:

            # Each recording has its own structure.oebin file
            recJson = os.path.join(recDir, self.jsonFile)
            if not os.path.isfile(recJson):
                err = "Missing OpenEphys json metadata file {json} for recording {rec}"
                raise IOError(err.format(json=self.jsonFile, rec=recDir))
            with open(recJson, "r") as rj:
                recInfo = json.load(rj)

            # --- CONTINUOUS ---
            continuous = self.dict_get(recJson, recInfo, "continuous")
            if len(continuous) != 1:
                err = "Unexpected format of field continuous in JSON file {}"
                raise ValueError(err.format(recJson))
            continuous = continuous[0]

            nChannels = self.dict_get(recJson, continuous, "num_channels")
            if nChannels != len(self.xmlRecChannels):
                err = "Channel mismatch between {} and {}"
                raise ValueError(err.format(self.settingsFile, recJson))

            device = self.dict_get(recJson, continuous, "source_processor_name")
            if not (device in self.device or self.device in device):
                err = "Recording device mismatch between {} and {}"
                raise ValueError(err.format(self.settingsFile, recJson))

            srate = self.dict_get(recJson, continuous, "sample_rate")
            if self.sampleRate is not None:
                if srate != self.sampleRate:
                    err = "Unsupported: more than one sample-rate in JSON file {}"
                    raise ValueError(err.format(recJson))
            else:
                self.sampleRate = float(srate)

            channels = self.dict_get(recJson, continuous, "channels")
            remap_warn = []
            for ck, chan in enumerate(self.xmlRecChannels):
                jsonChan = channels[ck]
                name = self.dict_get(recJson, jsonChan, "channel_name")
                sourceIdx = self.dict_get(recJson, jsonChan, "source_processor_index")
                recIdx = self.dict_get(recJson, jsonChan, "recorded_processor_index")
                units = self.dict_get(recJson, jsonChan, "units")
                xmlIdx = int(chan.get("number"))
                if name != chan.get("name"):
                    if len(remap_warn) == 0:
                        remap_warn = "Assuming channel remapping for channels: "
                    remap_warn += name+' '
                    chan.set("name", name)
                elif sourceIdx != xmlIdx and recIdx != xmlIdx:
                    err = "Recording channel index mismatch in JSON file {}: expected {} found {} or {}"
                    raise ValueError(err.format(recJson, xmlIdx, sourceIdx, recIdx))
                chan.set("units", units)
            if len(remap_warn) > 0:
                print(remap_warn)


            # We allow each recording to have its own unit but no mV/uV mix ups within the same recording
            chanUnits = list(set(chan.get("units") for chan in self.xmlRecChannels))
            if len(chanUnits) > len(self.xmlRecGroups):
                err = "Found recording channel groups with inconsistent units in JSON file {}: found {}"
                raise ValueError(err.format(recJson, chanUnits))
            if any(chanUnit not in self.recChannelUnitConversion.keys() for chanUnit in chanUnits):
                err = "Invalid units {} in JSON file {}; supported voltage units are {}"
                raise ValueError(err.format(chanUnits, recJson, list(self.recChannelUnitConversion.keys())))

            # --- EVENTS ---
            events = self.dict_get(recJson, recInfo, "events")
            for event in events:
                desc = self.dict_get(recJson, event, "description")
                if "TTL Events" in desc:
                    if not all(chan.get("Name").startswith("TTL") for chan in self.xmlEvtChannels):
                        err = "Event channel mismatch in JSON file {}"
                        raise ValueError(err.format(recJson))
                    nChannels = self.dict_get(recJson, event, "num_channels")
                    if nChannels != len(self.xmlEvtChannels):
                        err = "Event channel mismatch between {} and {}"
                        raise ValueError(err.format(self.settingsFile, recJson))
                    eventDir = self.dict_get(recJson, event, "folder_name")
                    eventDir = os.path.join(recDir, "events", eventDir)
                    if len(os.listdir(eventDir)) == 0:
                        err = "No TTL events found in {} given by JSON file {}"
                        raise IOError(err.format(eventDir, recJson))
                    if not os.path.isfile(os.path.join(eventDir, "full_words.npy")):
                        err = "No TTL event markers found in {} given by JSON file {}"
                        raise IOError(err.format(eventDir, recJson))
                    self.eventDirs.append(eventDir)
                    evtDtype = self.dict_get(recJson, event, "type")
                    self.eventDtypes.append(evtDtype)
                    srate = self.dict_get(recJson, event, "sample_rate")
                    if srate != self.sampleRate:
                        err = "Unsupported: more than one sample-rate in JSON file {}"
                        raise ValueError(err.format(recJson))

            # --- SPIKES ---
            spikes = self.dict_get(recJson, recInfo, "spikes")
            for spike in spikes:
                raise NotImplementedError("Spike data is currently not supported")

        return

    def dict_get(
            self,
            recJson : str,
            dict : Dict,
            key : str) -> Union[int, str, List]:
        """
        Helper method that (tries to) fetch an element from OE JSON file
        """
        value = dict.get(key)
        if value is None:
            err = "Missing expected field {} in JSON file {}"
            raise ValueError(err.format(key, recJson))
        return value


def _array_parser(
        var : Any,
        varname : str = "varname",
        ntype : Optional[str] = None,
        hasinf: Optional[bool] = None,
        hasnan: Optional[bool] = None,
        dims: Optional[Union[Tuple, int]] = None) -> NDArray:
    """
    Helper to parse array-like objects and check for correct dype, shape, finiteness, etc.
    ntype = "numeric"/"str"

    Parameters
    ----------
    var : array_like
        Array-like object to verify
    varname : str
        Local variable name used in caller
    ntype : None or str
        Expected data type of `var`. Possible options are 'numeric' or 'str'
        If `ntype` is `None` the data type of `var` is not checked.
    hasinf : None or bool
        If `hasinf` is not `None` the input array `var` is considered invalid
        if it contains non-finite elements (`np.inf`).
    hasnan : None or bool
        If `hasnan` is not `None` the input array `var` is considered invalid
        if it contains undefined elements (`np.nan`).
    dims : None or int or tuple
        Expected number of dimensions (if `dims` is an integer) or shape
        (if `dims` is a tuple) of `var`. By default, singleton dimensions
        of `var` are ignored if `dims` is a tuple, i.e., for `dims = (10, )`
        an array `var` with `var.shape = (10, 1)` is considered valid. However,
        if singleton dimensions are explicitly queried by setting `dims = (10, 1)`
        any array `var` with `var.shape = (10, )` or `var.shape = (1, 10)` is
        considered invalid.
        Unknown dimensions can be represented as `None`, i.e., for
        `dims = (10, None)` arrays with shape `(10, 1)`, `(10, 100)` or
        `(10, 0)` are all considered valid, however, any 1d-array (e.g.,
        `var.shape = (10,)`) is invalid.
        If `dims` is an integer, `var.ndim` has to match `dims` exactly, i.e.,
        any array `var` with `var.shape = (10, )` is considered invalid if
        `dims = 2` and conversely, `dims = 1` and `var.shape = (10,  1)`
        triggers an exception.

    Returns
    -------
    arr : np.ndarray
        NumPy array conversion of input `var`
    """

    # If necessary, convert `var` to simplify parsing
    if not isinstance(var, np.ndarray):
        arr = np.array(var, dtype=type(var[0]))
    else:
        arr = var

    # If required, parse type (handle "int_like" and "numeric" separately)
    if ntype is not None:
        msg = "Expected {vname:s} with dtype = {dt:s}, found {act:s}"
        if ntype in ["numeric"]:
            if not np.issubdtype(arr.dtype, np.number):
                raise ValueError(msg.format(vname=varname, dt=ntype, act=str(arr.dtype)))
        else:
            if not np.issubdtype(arr.dtype, np.dtype("str").type):
                raise ValueError(msg.format(vname=varname, dt=ntype, act=str(arr.dtype)))

    # If required, parse finiteness of array-elements
    if hasinf is not None and np.isinf(arr).any():
        msg = "Expected {vname:s} to be finite, found array with {numinf:d} `inf` entries"
        raise ValueError(msg.format(vname=varname, numinf=np.isinf(arr).sum()))

    # If required, parse well-posedness of array-elements
    if hasnan is not None and np.isnan(arr).any():
        msg = "Expected {vname:s} to be a well-defined numerical array, " +\
            "found array with {numnan:d} `NaN` entries"
        raise ValueError(msg.format(vname=varname, numnan=np.isnan(arr).sum()))

    # If required parse dimensional layout of array
    if dims is not None:

        # Account for the special case of 1d character arrays (that
        # collapse to 0d-arrays when squeezed)
        ischar = int(np.issubdtype(arr.dtype, np.dtype("str").type))

        # Compare shape or dimension number
        if isinstance(dims, tuple):
            if len(dims) > 1:
                ashape = arr.shape
            else:
                if arr.size == 1:
                    ashape = arr.shape
                else:
                    ashape = max((ischar,), arr.squeeze().shape)
            msg = "Expected {vname:s} to be a {nd:d}-dimensional array, " +\
                "found array of shape {shp:s}"
            if len(dims) != len(ashape):
                raise ValueError(msg.format(vname=varname, nd=len(dims), shp=str(arr.shape)))
            msg = "Expected {vname:s} to have shape {dim:s}, found array of shape {shp:s}"
            for dk, dim in enumerate(dims):
                if dim is not None and ashape[dk] != dim:
                    raise ValueError(msg.format(vname=varname, dim=str(dims), shp=str(arr.shape)))
        else:
            ndim = max(ischar, arr.ndim)
            if ndim != dims:
                msg = "Expected {vname:s} to be a {nd:d}-dimensional array, " +\
                    "found array of shape {shp:s}"
                raise ValueError(msg.format(vname=varname, nd=dims, shp=str(arr.shape)))

    # Return (possibly ndarray-converted) input
    return arr


# Ensure provided trial start/stop times are actually found in data
def _is_close(timeArr : ArrayLike, trialTimes: ArrayLike) -> None:
    """
    WARNING: This only works for sorted `timeArr` arrays!
    """

    idx = np.searchsorted(timeArr, trialTimes, side="left")
    leftNbrs = np.abs(trialTimes - timeArr[np.maximum(idx - 1, np.zeros(idx.shape, dtype=np.intp))])
    rightNbrs = np.abs(trialTimes - timeArr[np.minimum(idx, np.full(idx.shape, timeArr.size - 1, dtype=np.intp))])
    shiftLeft = ((idx == timeArr.size) | (leftNbrs < rightNbrs))
    idx[shiftLeft] -= 1

    if not np.allclose(timeArr[idx], trialTimes):
        raise ValueError("Provided trial start/stop times cannot be found in data")

    return


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def export2nwb(
    data_dir : str,
    output : str,
    memuse : int = 3000,
    session_description : Optional[str] = None,
    identifier : Optional[str] = None,
    session_id : Optional[str] = None,
    session_start_time : Optional[datetime] = None,
    trial_start_times : Optional[Union[List[float], Tuple[float], np.ndarray]] = None,
    trial_stop_times : Optional[Union[List[float], Tuple[float], np.ndarray]] = None,
    trial_tags : Optional[Union[List[str], Tuple[str], np.ndarray]] = None,
    trial_markers : Optional[Union[List, Tuple, np.ndarray]] = None,
    experimenter : Optional[str] = None,
    lab : Optional[str] = None,
    institution : Optional[str] = None,
    experiment_description : Optional[str] = None) -> None:
    """
    Export binary OpenEphys data to NWB 2.x

    Parameters
    ----------
    data_dir : str
        Name of directory (may include full path) containing OpenEphys binary data
        (e.g., `"/path/to/recordingDir"`)
    output : str
        Name of NWB file (may include full path) to be generated (must not exist).
        The file-name extension can be chosen freely (e.g., `"/path/to/outputFile.myext"`),
        if no extension is provided the suffix `'.nwb'` is added (e.g., ``output = "myfile"``
        generates an NWB container `"myfile.nwb"` in the current directory).
    memuse : int
        Approximate in-memory cache size (in MB) for reading data from disk
    session_description : str or None
        Human readable caption of experimental session (e.g., `"Experiment_1"`).
        If not provided, the base name of `data_dir` is used.
    identifier : str or None
        Unique tag (does not need to be human readable) associated to experimental
        session (e.g., `"4310-d217-4558-be2f"`). If not provided, a randomly generated
        unique ID tag is used.
    session_id : str or None
        Annotation of recording within session (e.g., `"rec_1"`). If not provided,
        the base name of the recording folder(s) inside `data_dir` is used.
        **Note**: A custom value of `session_id` can only be provided if the session
        in `data_dir` contains exactly one recording.
    session_start_time : datetime or None
        Starting time of experimental session (e.g., `"datetime(2021, 11, 9, 17, 6, 14)"`).
        If not provided, `session_start_time` is read from the session's OpenEphys
        xml settings file.
    trial_start_times : array-like or None
        List, tuple or 1darray of trial onset times (in seconds). Length has to match
        `trial_stop_times`. Trials can be set via `trial_start_times`/`trial_stop_times`
        or `trial_markers`, **not*** both.
    trial_stop_times : array-like or None
        List, tuple or 1darray of trial stop times (in seconds). Length has to match
        `trial_start_times`. Trials can be set via `trial_start_times`/`trial_stop_times`
        or `trial_markers`, **not*** both.
    trial_tags : array-like or None
        List, tuple or 1darry comprising `numTrial` trial labels.
    trial_markers : 2 element list or tuple or None
        Event markers ``(start, stop)`` for delimiting trials. Trials can be set via
        `trial_start_times`/`trial_stop_times` or `trial_markers`, **not*** both.
    experimenter : str or None
        Name of person conducting the experimental session (e.g., `"Whodunnit"`).
        If not provided, the host name of the recording computer used for the
        experiment is used.
    lab : str or None
        Name of research lab or group that performed the experiment (e.g., `"HilbertSpace"`).
        If not provided and the experiment was performed at ESI, the associated
        lab is inferred from the session's OpenEphys xml settings file. Otherwise
        no default value is assigned.
    institution : str or None
        Name of the university/institution where the experiment was performed
        (e.g., "Hogwarts"). If not provided and the experiment was performed at ESI,
        then `institution` is set automatically. Otherwise no default value is
        assigned.
    experiment_description : str or None
        Human readable description of experiment (e.g., `"What a great idea that was"`).
        No default value is assigned if not provided.

    Returns
    -------
    Nothing : None

    Notes
    -----
    All optional parameters (i.e., keyword arguments like `session_description` etc.)
    are taken from the `NWBFile <https://pynwb.readthedocs.io/en/latest/pynwb.file.html#pynwb.file.NWBFile>`_
    constructor. Please refer to the
    `official NWB documentation <https://pynwb.readthedocs.io/en/stable/tutorials/general/file.html>`_
    for additional information on the role of these parameters.

    Examples
    --------
    Export a recording using default settings for all optional parameters:

    >>> from oephys2nwb import export2nwb
    >>> input = "/path/to/recordingDir"
    >>> output = "/path/to/outputFile.nwb"
    >>> export2nwb(input, output)

    """

    # First, ensure target NWB container does not exist yet
    outFile = os.path.abspath(os.path.expanduser(os.path.normpath(output)))
    if os.path.isfile(outFile):
        err = "Output file {} already exists"
        raise IOError(err.format(outFile))
    outBase, tmp = os.path.split(outFile)
    outName, outExt = os.path.splitext(tmp)
    if len(outExt) == 0:
        outExt = ".nwb"

    # If trial information was specified, process it here
    if trial_start_times is not None:
        trial_start_times = _array_parser(trial_start_times, varname="trial_start_times",
                                          ntype="numeric", hasinf=False, hasnan=False,
                                          dims=(None,)).flatten()
    if trial_stop_times is not None:
        trial_stop_times = _array_parser(trial_stop_times, varname="trial_stop_times",
                                         ntype="numeric", hasinf=False, hasnan=False,
                                         dims=(None,)).flatten()
    if trial_tags is not None:
        trial_tags = _array_parser(trial_tags, varname="trial_tags", ntype="str",
                                   dims=(None,)).flatten()
    if trial_markers is not None:
        trial_markers = _array_parser(trial_markers, varname="trial_markers",
                                      ntype="numeric", hasinf=False, hasnan=False,
                                      dims=(2,)).flatten()

    # Ensure trial specs are consistent
    nTrials = None
    if trial_start_times is not None:
        if trial_stop_times is None:
            raise ValueError("Cannot process `trial_start_times` without `trial_stop_times`.")
        nTrials = trial_start_times.size

    if trial_stop_times is not None:
        if nTrials is None:
            raise ValueError("Cannot process `trial_stop_times` without `trial_start_times`.")
        if trial_stop_times.size != nTrials:
            raise ValueError("Lengths of `trial_start_times` and `trial_stop_times` have to match!")

    if nTrials is not None:
        if any(trial_stop_times - trial_start_times <= 0):
            err = "Provided `trial_start_times` and `trial_stop_times` contain " +\
                "trials with length <= 0"
            raise ValueError(err)

    if trial_tags is not None:
        if nTrials is None:
            raise ValueError("Cannot process `trial_tags` without start/stop times!")
        if trial_tags.size != nTrials:
            err = "Lengths of `trial_tags` and `trial_start_times` and " +\
                "`trial_stop_times` have to match!"
            raise ValueError(err)

    if nTrials is not None and trial_markers is not None:
        err = "Cannot process `trial_markers` and `trial_start_times` and " +\
            "`trial_stop_times` simultaneously"
        raise ValueError(err)

    # All remaining error checks (except for basic type matching) is performed by `EphysInfo`
    eInfo = EphysInfo(data_dir,
                      session_description=session_description,
                      identifier=identifier,
                      session_id=session_id,
                      session_start_time=session_start_time,
                      experimenter=experimenter,
                      lab=lab,
                      institution=institution,
                      experiment_description=experiment_description)

    # If `data_dir` contains multiple recordings, prepare base-name of NWB containers
    if len(eInfo.recordingDirs) > 1:
        outName += "_recording{}"

    # Use collected info to create NWBFile instance
    nRecChannels = len(eInfo.xmlRecChannels)
    session = Session(data_dir)
    for rk, recDir in enumerate(eInfo.recordingDirs):

        # Load continuous OE data
        rec = session.recordnodes[0].recordings[rk]
        data = rec.continuous[0].samples
        timeStamps = rec.continuous[0].timestamps / eInfo.sampleRate

        # Get OE event data
        if len(eInfo.eventDirs) > 0:
            evtPd = session.recordnodes[0].recordings[rk].events
            evt = np.load(os.path.join(eInfo.eventDirs[rk], "full_words.npy")).astype(int)
            ts = evtPd.timestamp.to_numpy()  / eInfo.sampleRate

            # If 16bit event-markers are used, combine 2 full words
            if eInfo.eventDtypes[rk] == "int16":
                evt16 = np.zeros((evt.shape[0]), int)
                for irow in range(evt.shape[0]):
                    evt16[irow] = int(format(evt[irow,1], "08b") + format(evt[irow,0], "08b"), 2)
                evt = evt16

            # If trial markers were provided, ensure they are consistent w/event-data
            if trial_markers is not None:
                # find unique evt only
                evt_ts = np.stack((evt, ts),axis=-1)
                evt_unique, ind = np.unique(evt_ts, axis=0, return_index=True)
                evt_unique = evt_ts[np.sort(ind)]

                trial_start_idx = np.where(evt_unique[:,0] == trial_markers[0])[0]
                trial_stop_idx = np.where(evt_unique[:,0] == trial_markers[1])[0]

                # take care of recording stopping before trial end
                if trial_stop_idx.size == trial_start_idx.size-1:
                    trial_start_idx = trial_start_idx[:-1]

                if trial_start_idx.size != trial_stop_idx.size:
                    err = "Provided `trial_markers` yield unequal trial start/stop counts"
                    raise ValueError(err)

                trial_start_times = evt_unique[:,1][trial_start_idx]
                trial_stop_times = evt_unique[:,1][trial_stop_idx]

                if any(trial_stop_times - trial_start_times <= 0):
                    err = "Provided `trial_markers` contain trials with length <= 0"
                    raise ValueError(err)

            # If trial delimiters were provided, ensure those are actually found in the data
            if trial_start_times is not None:
                _is_close(timeStamps, trial_start_times)
                _is_close(timeStamps, trial_stop_times)

                # If no trial tags were provided, generate dummy ones
                if trial_tags is None:
                    trial_tags = [None] * len(trial_start_times)

        # Either use provided (single!) session ID or generate one based on `recDir`
        if eInfo.session_id is None:
            session_id = os.path.basename(recDir)
        else:
            session_id = eInfo.session_id

        nwbfile = NWBFile(eInfo.session_description,
                          eInfo.identifier,
                          eInfo.session_start_time,
                          experimenter=eInfo.experimenter,
                          lab=eInfo.lab,
                          institution=eInfo.institution,
                          experiment_description=eInfo.experiment_description,
                          session_id=session_id)

        device = nwbfile.create_device(eInfo.device)

        # This should never happen: expected no. of recording channels does not match data shape...
        if nRecChannels not in data.shape:
            err = "Binary data has shape {} which does not match expected number of channels {}"
            raise ValueError(err.format(data.shape, nRecChannels))
        if data.shape[1] != nRecChannels:
            data = data.T
        chanGains = np.array([float(chan.get("gain")) for chan in eInfo.xmlRecChannels])

        # Create separate `ElectricalSeries` objects for each recording channel group
        esCounter = 1
        elCounter = 0
        esList = []
        elecIdxs_efficient = []
        for groupName in eInfo.xmlRecGroups:

            # Every channel group is mapped onto an `electrode_group`
            chanDesc = "OpenEphys {} channels".format(groupName)
            xmlChans = [chan for chan in eInfo.xmlRecChannels if chan.get("group") == groupName]
            chanInfo = [(int(chan.get("number")), chan.get("name")) for chan in xmlChans]
            elecGroup = nwbfile.create_electrode_group(name=groupName,
                                                       description=chanDesc,
                                                       location="",
                                                       device=device)

            # Each channel is considered an "electrode" so that channel-names
            # are preserved in the NWB container
            for chanIdx, chanName in chanInfo:
                nwbfile.add_electrode(id=chanIdx,
                                      location=chanName,
                                      group=elecGroup,
                                      imp=1.0,
                                      filtering="None",
                                      x=0.0, y=0.0, z=0.0)

            # An NWB `ElectricalSeries` requires a dynamic electrode table to keep
            # track of signal sources. In our case, the table region is the entire
            # electrode_group; perform some index gymnastics to map the table correctly
            # onto the list of all electrodes
            elecIdxs = [eInfo.xmlRecChannels.index(chan) for chan in xmlChans]
            tableIdx = list(range(elCounter, elCounter + len(elecIdxs)))
            elecRegion = nwbfile.create_electrode_table_region(tableIdx, chanDesc)
            elCounter += len(elecIdxs)

            # FIXME: this hack is necessary due to inconsistent voltage scaling
            # (cf. https://github.com/open-ephys/plugin-GUI/issues/472)
            if groupName == "ADC":
                chanUnit = "V"
            else:
                chanUnit = xmlChans[0].get("units")

            # Use default name of NWB object to increase chances that 3rd party
            # tools operate seamlessly with it; also use `rate` instead of storing
            # timestamps to ensure tools relying on constant sampling rate work

            # Simple speed improvement by keeping data memory mapped when possible
            if np.all(np.diff(elecIdxs) == 1):
                elecIdxs_efficient.append(np.s_[elecIdxs[0]:elecIdxs[-1]+1])
            else:
                elecIdxs_efficient.append(elecIdxs)

            # calculate 1 mb chunksizes
            membytes = 1024 **2
            nSamp = int(membytes / (len(elecIdxs) * data.dtype.itemsize))
            chunk_sz = (nSamp, len(elecIdxs))

            wrapped_data = H5DataIO(
                data=np.empty(shape=(0, len(elecIdxs)), dtype=data.dtype),#data[:, elecIdxs_efficient],
                chunks=chunk_sz,          # <---- Enable chunking
                maxshape=(None, len(elecIdxs)),  # <---- Make the time dimension unlimited and hence resizeable
            )

            elecData = ElectricalSeries(name="ElectricalSeries_{}".format(esCounter),
                                        data=wrapped_data,
                                        electrodes=elecRegion,
                                        channel_conversion=chanGains[elecIdxs],
                                        conversion=eInfo.recChannelUnitConversion[chanUnit],
                                        starting_time=float(timeStamps[0]),
                                        rate=eInfo.sampleRate,
                                        description=chanDesc)
            nwbfile.add_acquisition(elecData)
            esList.append(elecData)
            esCounter += 1

        # Include trial delimiters (if provided)
        if trial_start_times is not None:
            for ti in range(len(trial_start_times)):
                nwbfile.add_epoch(trial_start_times[ti],
                                  trial_stop_times[ti],
                                  trial_tags[ti])
                                #   esList)

        # Same as above: each event channel group makes up its own NWB-data entity
        for groupName in eInfo.xmlEvtGroups:
            if groupName == "TTL":
                if evt.min() < 0 or evt.max() > np.iinfo("uint16").max:
                    raise ValueError("Only unsigned integer TTL pulse values are supported. ")
                ttlData = TTLs(name="TTL_PulseValues",
                               data=evt.astype("uint16"),
                               labels=["No labels defined"],
                               timestamps=ts,
                               resolution=1/eInfo.sampleRate,
                               description="TTL pulse values")
                nwbfile.add_acquisition(ttlData)
                ttlChan = TTLs(name="TTL_Channels",
                               data=evtPd.channel.to_numpy(),
                               labels=["No labels defined"],
                               timestamps=ts,
                               resolution=1/eInfo.sampleRate,
                               description="TTL channel number")
                nwbfile.add_acquisition(ttlChan)
                ttlState = TTLs(name="TTL_ChannelStates",
                               data=evtPd.state.to_numpy(),
                               labels=["No labels defined"],
                               timestamps=ts,
                               resolution=1/eInfo.sampleRate,
                               description="TTL channel states")
                nwbfile.add_acquisition(ttlState)
            else:
                raise NotImplementedError("Currently, only TTL pulse events are supported")

        # Spikes currently not supported; caught by `EphysInfo` class in `process_json`

        # Finally, write NWB file to disk
        print('Writing NWB file')
        outFileName = os.path.join(outBase, outName + outExt).format(rk)
        with NWBHDF5IO(outFileName, "w") as io:
            io.write(nwbfile)

        # delete memory maps
        data_loc = data.filename
        data_shape = data.shape
        data_dtype = data.dtype
        del data, rec, timeStamps, session

        # write data to NWB file in blocks
        with NWBHDF5IO(outFileName, mode='a') as io:
            nwbfile = io.read()
            for ii, elecData in enumerate(esList):
                print('Writing NWB data ',elecData.name)
                nwb_data = nwbfile.get_acquisition(elecData.name).data
                nwb_data.resize((data_shape[0],nwb_data.maxshape[1]))#data[:, elecIdxs_efficient[ii]].shape)

                # Given memory cap, compute how many data blocks can be grabbed per swipe:
                # `nSamp` is the no. of samples that can be loaded into memory without exceeding `memuse`
                # `rem` is the no. of remaining samples, s. t. ``nSamp + rem = angDset.shape[0]`
                # `blockList` is a list of samples to load per swipe, i.e., `[nSamp, nSamp, ..., rem]`
                membytes = (memuse * 1024**2)
                nSamp = int(membytes / (nwb_data.shape[1] * data_dtype.itemsize))
                rem = int(data_shape[0] % nSamp)
                blockList = [nSamp] * int(data_shape[0] // nSamp) + [rem] * int(rem > 0)

                real_memuse = (nSamp*nwb_data.shape[1]* data_dtype.itemsize)/1024**3
                print("Writing data in blocks of {} GB".format(round(real_memuse, 2)))


                for m, M in enumerate(blockList):
                    st, end = m * nSamp, m * nSamp + M
                    newfp = np.memmap(data_loc, dtype=data_dtype, mode='r', shape=(data_shape))
                    nwb_data[st:end, :] = newfp[st:end, elecIdxs_efficient[ii]]
                    del newfp


        # Perform validation of generated NWB file: https://pynwb.readthedocs.io/en/latest/validation.html
        this_python = os.path.join(os.path.dirname(sys.executable),'python')
        subprocess.run([this_python, "-m", "pynwb.validate", outFileName], check=True)

        # Happy breakdown
        print(f"Succesfully wrote {outFileName}")
        print("ALL DONE")

        return


# Parse CL args
def clarg_parser(args : List) -> Union[None, Dict]:
    """
    Helper function for parsing CL argument input
    """

    # Short description of the program
    desc = "Export binary OpenEphys data to NWB 2.0"
    docu = \
    """
    Detailed usage instructions

    This script can be either used from the command line or imported in
    Python.

    Command line use:

        export2nwb -i /path/to/recordingDir -o /path/to/outputFile.nwb

        Optional arguments (like experimenter or lab) are either inferred from
        OpenEphys meta data or can be provided via corresponding optional
        arguments (e.g., --experimenter "Whodunnit"). Please refer to the
        official NWB documentation for additional information on the role
        of the available optional arguments.

    Python module use:

        from oephys2nwb import export2nwb

        input = "/path/to/recordingDir"
        output = "/path/to/outputFile.nwb"

        export2nwb(input, output)

    More details can be found in the project README and the Python docstrings.
    """

    # Initialize parser with description and detailed help text
    parser = ArgumentParser(description=desc,
                            epilog=docu,
                            formatter_class=RawTextHelpFormatter)

    # Two "mandatory" args (make them optional so that call w/o args shows help)
    parser.add_argument("-i", "--input",
                        action="store", type=str, dest="data_dir", default=None,
                        help="(path to) directory containing OpenEphys binary data")
    parser.add_argument("-o", "--output",
                        action="store", type=str, dest="output", default=None,
                        help="(path to) NWB file to be generated (must not exist)")

    # Now the "real" optional stuff
    parser.add_argument("--session_description",
                        action="store", type=str, dest="session_description", default=None,
                        help="human readable caption of experimental session " +\
                            "(default: base name of input directory)")
    parser.add_argument("--identifier",
                        action="store", type=str, dest="identifier", default=None,
                        help="unique tag associated to experimental session " +\
                            "(default: randomly generated unique ID tag)")
    parser.add_argument("--session_id",
                        action="store", type=str, dest="session_id", default=None,
                        help="annotation associated to recording within session "+\
                            "(default: base name of recording directory)")
    parser.add_argument("--session_start_time",
                        action="store", type=str, dest="session_start_time", default=None,
                        help="starting time of session in format D MON YYYY HH:MM:SS, " +\
                            "e.g., 9 Nov 2021 17:06:14 (default: inferred from OpenEphys xml settings file)")
    parser.add_argument("--experimenter",
                        action="store", type=str, dest="experimenter", default=None,
                        help="name of experimenter (default: host name of recording computer)")
    parser.add_argument("--lab",
                        action="store", type=str, dest="lab", default=None,
                        help="lab/research group that performed experiment " +\
                            "(default: inferred if recorded at ESI otherwise None)")
    parser.add_argument("--institution",
                        action="store", type=str, dest="institution", default=None,
                        help="institution where experiment was performed" +\
                            "(default: inferred if recorded at ESI otherwise None)")
    parser.add_argument("--experiment_description",
                        action="store", type=str, dest="experiment_description", default=None,
                        help="human readable description of experiment (default: None)")

    # Parse CL-arguments: if first input wasn't provided, print help and exit
    args_dict = vars(parser.parse_args())
    if args_dict["data_dir"] is None:
        parser.print_help()
        return

    # Otherwise convert `session_start_time` to `datetime` object and return `args_dict``
    if args_dict["session_start_time"] is not None:
        timeStr = args_dict["session_start_time"]
        try:
            timeDt = datetime.strptime(timeStr, "%d %b %Y %H:%M:%S")
        except ValueError:
            msg = "Invalid format of recording time: '{datestr}' " +\
                "Please provide start time in format D MON YYYY HH:MM:SS " +\
                "(e.g., 9 Nov 2021 17:06:14)"
            raise ValueError(msg.format(datestr=timeStr))
        args_dict["session_start_time"] = timeDt
    return args_dict


# Run as script from command line
if __name__ == "__main__":

    # Invoke command-line argument parse helper
    args_dict = clarg_parser(sys.argv[1:])

    # If `args_dict` is not `None`, call actual function
    if args_dict:
        export2nwb(**args_dict)
