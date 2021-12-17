# -*- coding: utf-8 -*-
#
# Script for exporting binary OpenEphys data to NWB 2.x
#

import os
import xml.etree.ElementTree as ET
import uuid
import json
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from pathlib import Path
from datetime import datetime
from pydantic import validate_arguments
from pynwb import NWBFile
from pynwb.ecephys import ElectricalSeries
from ndx_events import TTLs
from open_ephys.analysis import Session

def _unitConversionMapping():
    return {"uV" : 1e-6, "mV" : 1e-3, "V" : 1.0}

@dataclass
class EphysInfo:

    data_dir : str
    session_description : Optional[str] = None
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

        self.data_dir = os.path.abspath(os.path.expanduser(os.path.normpath(self.data_dir)))
        if not os.path.isdir(self.data_dir):
            err = "Provided path {} does not point to an existing directory"
            raise IOError(err.format(self.data_dir))

        if self.session_description is None:
            self.session_description = os.path.basename(self.data_dir)

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
        elem = self.root.find(elemPath)
        if elem is None:
            xmlErr = "Invalid {} file: missing element {}"
            raise ValueError(xmlErr.format(self.settingsFile, elemPath))
        return elem

    def get_rec_channel_info(self):

        # Get XML element and fetch channels
        chanInfo = self.xml_get("SIGNALCHAIN/PROCESSOR/CHANNEL_INFO")
        chanList = []
        chanGroups = []
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
            chanGroups.append(chan.get("group"))

        # Abort if no valid channels were found
        if len(chanList) == 0:
            err = "Found no valid channels in {} file"
            raise ValueError(err.format(self.settingsFile))

        return chanList, list(set(chanGroups))

    def get_evt_channel_info(self):

        # Get XML element and fetch channels
        editor = self.xml_get("SIGNALCHAIN/PROCESSOR/EDITOR")
        chanList = []
        chanGroups = []
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
            chanGroups.append(chan.get("group"))

        # Abort if no valid channels were found
        if len(chanList) == 0:
            err = "Found no valid event channels in {} file"
            raise ValueError(err.format(self.settingsFile))

        return chanList, list(set(chanGroups))


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
            chanUnits = list(set(chan.get("units") for chan in self.xmlRecChannels))
            if len(chanUnits) > len(self.xmlRecGroups):
                err = "Found recording channel groups with inconsistent units in JSON file {}: found {}"
                raise ValueError(err.format(recJson, chanUnits))
            if any(chanUnit not in self.recChannelUnitConversion.keys() for chanUnit in chanUnits):
                err = "Invalid units {} in JSON file {}; supported voltage units are {}"
                raise ValueError(err.format(chanUnits, recJson, list(self.recChannelUnitConversion.keys())))

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

            spikes = self.dict_get(recJson, recInfo, "spikes")
            for spike in spikes:
                raise NotImplementedError("Spike data is currently not supported")


    def dict_get(self, recJson, dict, key):
        value = dict.get(key)
        if value is None:
            err = "Missing expected field {} in JSON file {}"
            raise ValueError(err.format(key, recJson))
        return value



@validate_arguments
def export2nwb(data_dir : str,
               session_description : Optional[str] = None,
               identifier : Optional[str] = None,
               session_id : Optional[str] = None,
               session_start_time : Optional[datetime] = None,
               experimenter : Optional[str] = None,
               lab : Optional[str] = None,
               institution : Optional[str] = None,
               experiment_description : Optional[str] = None) -> None:

    eInfo = EphysInfo(data_dir,
                      session_description=session_description,
                      identifier=identifier,
                      session_id=session_id,
                      session_start_time=session_start_time,
                      experimenter=experimenter,
                      lab=lab,
                      institution=institution,
                      experiment_description=experiment_description)

    nRecChannels = len(eInfo.xmlRecChannels)

    session = Session(data_dir)

    # Use collected info to create NWBFile instance
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

        if nRecChannels not in data.shape:
            err = "Binary data has shape {} which does not match expected number of channels {}"
            raise ValueError(err.format(data.shape, nRecChannels))
        if data.shape[1] != nRecChannels:
            data = data.T
        chanGains = np.array([float(chan.get("gain")) for chan in eInfo.xmlRecChannels])

        esCounter = 1
        elCounter = 0

        for groupName in eInfo.xmlRecGroups:

            chanDesc = "OpenEphys {} channels".format(groupName)
            xmlChans = [chan for chan in eInfo.xmlRecChannels if chan.get("group") == groupName]
            chanInfo = [(int(chan.get("number")), chan.get("name")) for chan in xmlChans]
            elecGroup = nwbfile.create_electrode_group(name=groupName,
                                                       description=chanDesc,
                                                       location="",
                                                       device=device)

            for chanIdx, chanName in chanInfo:
                nwbfile.add_electrode(id=chanIdx,
                                      location=chanName,
                                      group=elecGroup,
                                      imp=1.0,
                                      filtering="None",
                                      x=0.0, y=0.0, z=0.0)

            # Fixed
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

        # Events
        evtPd = session.recordnodes[0].recordings[rk].events
        evt = np.load(os.path.join(eInfo.eventDirs[rk], "full_words.npy")).astype(int)
        ts = evtPd.timestamp.to_numpy()

        if eInfo.eventDtypes[rk] == "int16":
            evt16 = np.zeros((evt.shape[0]), int)
            for irow in range(evt.shape[0]):
                evt16[irow] = int(format(evt[irow,1], "08b") + format(evt[irow,0], "08b"), 2)
            evt = evt16

        for groupName in eInfo.xmlEvtGroups:
            if groupName == "TTL":
                if evt.min() < 0:
                    raise ValueError("Only unsigned integer TTL pulse values are supported. ")
                evt = evt.astype("uint16")
                ttlData = TTLs(name="TTL_Pulses",
                               data=evt,
                               timestamps=ts,
                               resolution=1/eInfo.sampleRate,
                               description="TTL pulse values")
                nwbfile.add_acquisition(ttlData)
                ttlChan = TTLs(name="TTL_Channels",
                               data=evtPd.channel.to_numpy(),
                               timestamps=ts,
                               resolution=1/eInfo.sampleRate,
                               description="TTL pulse channels")
                nwbfile.add_acquisition(ttlChan)
            else:
                raise NotImplementedError("Currently, only TTL pulse events are supported")

        # Spikes currently not supported; caught by `EphysInfo` class in `process_json`

        import ipdb; ipdb.set_trace()

        # perform file validation: https://pynwb.readthedocs.io/en/latest/validation.html




    # session = Session(data_dir)
    chans = session.recordnodes[0].recordings[0].continuous[0].samples
    tpoins = chans = session.recordnodes[0].recordings[0].continuous[0].timestamps
    print(session.recordnodes[0].recordings[0].continuous[0].metadata)
    import ipdb; ipdb.set_trace()



        # if dtype == float: # Convert data to float array and convert bits to voltage.
        #     data = np.fromfile(f,np.dtype('>i2'),N) * float(header['bitVolts']) # big-endian 16-bit signed integer, multiplied by bitVolts
        # else:  # Keep data in signed 16 bit integer format.
        #     data = np.fromfile(f,np.dtype('>i2'),N)  # big-endian 16-bit signed integer
        # samples[indices[recordNumber]:indices[recordNumber+1]] = data


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
