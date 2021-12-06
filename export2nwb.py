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
    experimentDir : str = field(init=False)
    recordingDirs : List = field(init=False)
    xmlChannels : List = field(init=False)

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
            err = "Cannot read {}, original error message: {}"
            raise ET.ParseError(err.format(self.settingsFile, str(exc)))

        self.date = self.xml_get("INFO/DATE")
        self.machine = self.xml_get("INFO/MACHINE")

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

        self.xmlChannels = self.get_channel_info()




    def xml_get(self, elemPath):
        elem = self.root.find(elemPath)
        if elem is None:
            xmlErr = "Invalid {} file: missing element {}"
            raise ValueError(xmlErr.format(self.settingsFile, elemPath))
        return elem.text

    def get_channel_info(self):

        # Abuse `xml_get` to see if element exists
        self.xml_get("SIGNALCHAIN/PROCESSOR/CHANNEL_INFO")

        # If we made it here, the xml file contains channel info, now get it
        chanInfo = self.root.find("SIGNALCHAIN/PROCESSOR/CHANNEL_INFO")
        chanList = []
        for chan in chanInfo.iter("CHANNEL"):
            chanList.append(chan)

        return chanList

    def process_json(self, recDir):

        recFile = os.path.join(recDir, self.jsonFile)
        if not os.path.isfile(recFile):
            err = "Missing OpenEphys json metadata file {json} for recording {rec}"
            raise IOError(err.format(json=self.jsonFile, rec=recDir))
        recJson = json.load(open(recFile, "r"))


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

    for recDir in eInfo.recordingDirs:
        recJson = eInfo.process_json(recDir)



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
