# -*- coding: utf-8 -*-
#
# Script for exporting binary open ephys data to NWB 2.x
#

import os
import xml.etree.ElementTree as ET
import uuid
import json
from dataclasses import dataclass, field
from typing import Optional, List
from pathlib import Path
from datetime import datetime
from pydantic import validate_arguments
from pynwb import NWBFile


@dataclass
class EphysInfo:

    data_dir : str
    identifier : Optional[str] = None
    session_id : Optional[str] = None
    session_start_time : Optional[datetime] = None
    experimenter : Optional[str] = None
    lab : Optional[str] = None
    institution : Optional[str] = None
    experiment_description : Optional[str] = None

    settingsFile : str = field(default="settings.xml", init=False)
    jsonFile : str = field(default="structure.oebin")
    root : ET.Element = field(init=False)
    info : str = field(init=False)
    machine : str = field(init=False)
    device : str = field(init=False)
    experimentDir : str = field(init=False)
    recordingDirs : List = field(init=False)
    xmlRecChannels : List = field(init=False)
    xmlEvtChannels : List = field(init=False)
    sampleRate : float = field(default=None, init=False)

    # lon: float = 0.0
    # lat: float = 0.0

    def __post_init__(self):

        self.data_dir = os.path.abspath(os.path.expanduser(os.path.normpath(self.data_dir)))
        if not os.path.isdir(self.data_dir):
            err = "Provided path {} does not point to an existing directory"
            raise IOError(err.format(self.data_dir))

        self.process_xml()

        self.process_json()

    def process_xml(self):

        xmlFiles = []
        for cur_path, _, files in os.walk(self.data_dir):
            if self.settingsFile in files:
                xmlFiles.append(os.path.join(self.data_dir, cur_path, self.settingsFile))

        if len(xmlFiles) != 1:
            err = "Found {numf} {xmlfile} files in {folder}"
            raise ValueError(err.format(numf=len(xmlFiles), xmlfile=self.settingsFile, folder=self.data_dir))
        self.settingsFile = xmlFiles[0]

        basePath = Path(os.path.split(self.settingsFile)[0])
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

        self.device = self.xml_get("SIGNALCHAIN/PROCESSOR").get("name")
        if self.device is None:
            err = "Invalid {} file: empty tag in element SIGNALCHAIN/PROCESSOR"
            raise ValueError(err.format(self.settingsFile))

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

        self.xmlRecChannels = self.get_rec_channel_info()

        self.xmlEvtChannels = self.get_evt_channel_info()

    def xml_get(self, elemPath):
        elem = self.root.find(elemPath)
        if elem is None:
            xmlErr = "Invalid {} file: missing element {}"
            raise ValueError(xmlErr.format(self.settingsFile, elemPath))
        return elem

    def get_rec_channel_info(self):

        # Get XML element and fetch channels
        chanInfo = self.xml_get("SIGNALCHAIN/PROCESSOR/CHANNEL_INFO")
        chanList = []
        for chan in chanInfo.iter("CHANNEL"):

            # Assign each channel to a group by creating a new XML tag
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

        # Abort if no valid channels were found
        if len(chanList) == 0:
            err = "Found no valid channels in {} file"
            raise ValueError(err.format(self.settingsFile))

        return chanList

    def get_evt_channel_info(self):

        # Get XML element and fetch channels
        editor = self.xml_get("SIGNALCHAIN/PROCESSOR/EDITOR")
        chanList = []
        for chan in editor.iter("EVENT_CHANNEL"):

            # Assign each channel to a group by creating a new XML tag
            chanName = chan.get("Name")
            if chanName is None or chan.get("Channel") is None:
                err = "Invalid event channel specification in {}"
                raise ValueError(err.format(self.settingsFile))
            if chanName.startswith("TTL"):
                chan.set("group", "TTL")
            else:
                chan.set("group", "EVT")
            chanList.append(chan)

        # Abort if no valid channels were found
        if len(chanList) == 0:
            err = "Found no valid event channels in {} file"
            raise ValueError(err.format(self.settingsFile))

        return chanList


    def process_json(self):

        for recDir in self.recordingDirs:

            recJson = os.path.join(recDir, self.jsonFile)
            if not os.path.isfile(recJson):
                err = "Missing OpenEphys json metadata file {json} for recording {rec}"
                raise IOError(err.format(json=self.jsonFile, rec=recDir))
            with open(recJson, "r") as rj:
                recInfo = json.load(rj)

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
                self.sampleRate = srate

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
                srate = self.dict_get(recJson, event, "sample_rate")
                if srate != self.sampleRate:
                    err = "Unsupported: more than one sample-rate in JSON file {}"
                    raise ValueError(err.format(recJson))


    def dict_get(self, recJson, dict, key):
        value = dict.get(key)
        if value is None:
            err = "Missing expected field {} in JSON file {}"
            raise ValueError(err.format(key, recJson))
        return value



@validate_arguments
def export2nwb(data_dir : str,
               identifier : Optional[str] = None,
               session_id : Optional[str] = None,
               session_start_time : Optional[datetime] = None,
               experimenter : Optional[str] = None,
               lab : Optional[str] = None,
               institution : Optional[str] = None,
               experiment_description : Optional[str] = None) -> None:

    eInfo = EphysInfo(data_dir, session_start_time)

    import ipdb; ipdb.set_trace()


    # session_description, identifier, session_start_time

    # nwbfile = NWBFile('my first synthetic recording', 'EXAMPLE_ID', 'asdf',
    #                 experimenter='Dr. Bilbo Baggins',
    #                 lab='Bag End Laboratory',
    #                 institution='University of Middle Earth at the Shire',
    #                 experiment_description='I went on an adventure with thirteen dwarves to reclaim vast treasures.',
    #                 session_id='LONELYMTN' --> recording1)



if __name__ == "__main__":

    # Test stuff within here...
    dataDir = "testrecording_2021-11-09_17-06-14"

    root = export2nwb(dataDir)
