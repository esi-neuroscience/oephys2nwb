#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Export binary OpenEphys data to NWB 2.x
#

import os
import sys
import xml.etree.ElementTree as ET
import uuid
import json
import subprocess
import numpy as np
from argparse import ArgumentParser, RawTextHelpFormatter
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from pathlib import Path
from datetime import datetime
from pydantic import validate_arguments
from pynwb import NWBFile, NWBHDF5IO
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

    def __post_init__(self):
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

    def process_xml(self):
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
        experimentDirs = [str(entry) for entry in basePath.iterdir() if entry.is_dir()]
        if len(experimentDirs) != 1:
            err = "Found {numf} experiments in {folder}"
            raise ValueError(err.format(numf=len(experimentDirs), folder=self.data_dir))
        self.experimentDir = experimentDirs[0]

        self.recordingDirs = [str(entry) for entry in Path(self.experimentDir).iterdir() if entry.is_dir()]
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
        self.xmlRecChannels = list(channels)
        self.xmlRecGroups = list(groups)

        channels, groups = self.get_evt_channel_info()
        self.xmlEvtChannels = list(channels)
        self.xmlEvtGroups = list(groups)

    def xml_get(self, elemPath):
        """
        Helper method that (tries to) fetch an element from settings.xml
        """
        elem = self.root.find(elemPath)
        if elem is None:
            xmlErr = "Invalid {} file: missing element {}"
            raise ValueError(xmlErr.format(self.settingsFile, elemPath))
        return elem

    def get_rec_channel_info(self):
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

    def get_evt_channel_info(self):
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

        # Abort if no valid channels were found
        if len(chanList) == 0:
            err = "Found no valid event channels in {} file"
            raise ValueError(err.format(self.settingsFile))

        return chanList, list(set(chanGroups))


    def process_json(self):
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
            for ck, chan in enumerate(self.xmlRecChannels):
                jsonChan = channels[ck]
                name = self.dict_get(recJson, jsonChan, "channel_name")
                sourceIdx = self.dict_get(recJson, jsonChan, "source_processor_index")
                recIdx = self.dict_get(recJson, jsonChan, "recorded_processor_index")
                units = self.dict_get(recJson, jsonChan, "units")
                xmlIdx = int(chan.get("number"))
                if name != chan.get("name"):
                    err = "Recording channel name mismatch in JSON file {}: expected {} found {}"
                    raise ValueError(err.format(recJson, chan.get("name"), name))
                if sourceIdx != xmlIdx and recIdx != xmlIdx:
                    err = "Recording channel index mismatch in JSON file {}: expected {} found {} or {}"
                    raise ValueError(err.format(recJson, xmlIdx, sourceIdx, recIdx))
                chan.set("units", units)

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

    def dict_get(self, recJson, dict, key):
        """
        Helper method that (tries to) fetch an element from OE JSON file
        """
        value = dict.get(key)
        if value is None:
            err = "Missing expected field {} in JSON file {}"
            raise ValueError(err.format(key, recJson))
        return value


@validate_arguments
def export2nwb(data_dir : str,
               output : str,
               session_description : Optional[str] = None,
               identifier : Optional[str] = None,
               session_id : Optional[str] = None,
               session_start_time : Optional[datetime] = None,
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

        rec = session.recordnodes[0].recordings[rk]
        data = rec.continuous[0].samples
        timeStamps = rec.continuous[0].timestamps

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
            # FIXME: Memory efficient writing
            # https://pynwb.readthedocs.io/en/stable/tutorials/advanced_io/iterative_write.html#example-convert-large-binary-data-arrays
            elecData = ElectricalSeries(name="ElectricalSeries_{}".format(esCounter),
                                        data=data[:, elecIdxs],
                                        electrodes=elecRegion,
                                        channel_conversion=chanGains[elecIdxs],
                                        conversion=eInfo.recChannelUnitConversion[chanUnit],
                                        starting_time=float(timeStamps[0]),
                                        rate=eInfo.sampleRate,
                                        description=chanDesc)
            nwbfile.add_acquisition(elecData)
            esCounter += 1

        # Get OE event data;
        evtPd = session.recordnodes[0].recordings[rk].events
        evt = np.load(os.path.join(eInfo.eventDirs[rk], "full_words.npy")).astype(int)
        ts = evtPd.timestamp.to_numpy()

        # If 16bit event-markers are used, combine 2 full words
        if eInfo.eventDtypes[rk] == "int16":
            evt16 = np.zeros((evt.shape[0]), int)
            for irow in range(evt.shape[0]):
                evt16[irow] = int(format(evt[irow,1], "08b") + format(evt[irow,0], "08b"), 2)
            evt = evt16

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
                ttlChan = TTLs(name="TTL_ChannelValues",
                               data=evtPd.channel.to_numpy(),
                               labels=["No labels defined"],
                               timestamps=ts,
                               resolution=1/eInfo.sampleRate,
                               description="TTL pulse channels")
                nwbfile.add_acquisition(ttlChan)
            else:
                raise NotImplementedError("Currently, only TTL pulse events are supported")

        # Spikes currently not supported; caught by `EphysInfo` class in `process_json`

        # Finally, write NWB file to disk
        outFileName = os.path.join(outBase, outName + outExt).format(rk)
        with NWBHDF5IO(outFileName, "w") as io:
            io.write(nwbfile)

        # Perform validation of generated NWB file: https://pynwb.readthedocs.io/en/latest/validation.html
        subprocess.run(["python", "-m", "pynwb.validate", outFileName], check=True)

        return


# Parse CL args
def clarg_parser(args):
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
