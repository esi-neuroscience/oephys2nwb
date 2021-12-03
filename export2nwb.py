# -*- coding: utf-8 -*-
#
# Script for exporting binary open ephys data to NWB 2.x
#

import os
from typing import Type
import xml.etree.ElementTree as ET
from hashlib import blake2b
from datetime import datetime
from pynwb import NWBFile


def read_xml(xmlfile):
    pass


def export2nwb(recording_dir, session_id=None, session_start_time=None,
               experimenter=None, institution=None, experiment_description=None):

    if not isinstance(recording_dir, str):
        raise TypeError("blah")

    fullPath = os.path.abspath(os.path.expanduser(os.path.normpath(recording_dir)))
    if not os.path.isdir(fullPath):
        raise IOError("blah")

    xmlFiles = []
    settingsFile = "settings.xml"
    for cur_path, _, files in os.walk(fullPath):
        if settingsFile in files:
            xmlFiles.append(os.path.join(fullPath, cur_path, settingsFile))


    if len(xmlFiles) != 1:
        raise ValueError("blah")
    settingsFile = xmlFiles[0]


    expectedRootTags = ["INFO", "SIGNALCHAIN"]
    expectedInfoTags = ["DATE", "MACHINE"]
    expectedChanTags = ["SIGNALCHAIN/PROCESSOR/CHANNEL_INFO/CHANNEL"]

    # chanInfo = root.find("SIGNALCHAIN/PROCESSOR/CHANNEL_INFO")
    # for chan in chanInfo.iter("CHANNEL"):
    #     print(chan.get("name"))

    try:
        root = ET.parse(settingsFile).getroot()
    except ET.ParseError as pexc:
        raise ET.ParseError("blah, original error message: {}".format(str(pexc)))

    info = _xml_fetch_element(settingsFile, root, "INFO", "INFO")

    date = _xml_fetch_element(settingsFile, info, "DATE", "INFO/DATE")
    machine = _xml_fetch_element(settingsFile, info, "MACHINE", "INFO/MACHINE")

    if session_start_time is None:
        try:
            session_start_time = datetime.strptime(date.text, "%d %b %Y %H:%M:%S")
        except ValueError:
            msg = "settings.xml: recording date in unexpected format: '{}' " +\
                "Please provide session start time manually (via keyword session_start_time)"
            raise ValueError(msg.format(date.text))
    else:
        pass # parse session_start_time

    if session_id is None:
        session_id = blake2b(str(session_start_time).encode(), digest_size=16).hexdigest()
    else:
        pass # parse session_id

    if experimenter is None:
        experimenter = machine.text
    else:
        pass # parse experimenter

    if machine.text.startswith("ESI-"):
        if institution is None:
            institution = "Ernst Strüngmann Institute (ESI) for Neuroscience " +\
                "in Cooperation with Max Planck Society"
        if lab is None:
            lab = machine.text.lower().split("esi-")[1][2:5].upper() # get ESI lab code HSV, LAU, FRI etc.

    if institution is not None:
        pass # parse institution

    if lab is not None:
        pass # parse lab

    if not isinstance(experiment_description, str) and experiment_description is not None:
        raise TypeError("blah")

    # FIXME: put all str/None candidates in list and parse them together


    import ipdb; ipdb.set_trace()

    return root

    import ipdb; ipdb.set_trace()

    # session_description, identifier, session_start_time

    # nwbfile = NWBFile('my first synthetic recording', 'EXAMPLE_ID', 'asdf',
    #                 experimenter='Dr. Bilbo Baggins',
    #                 lab='Bag End Laboratory',
    #                 institution='University of Middle Earth at the Shire',
    #                 experiment_description='I went on an adventure with thirteen dwarves to reclaim vast treasures.',
    #                 session_id='LONELYMTN' --> recording1)

def _xml_fetch_element(settingsFile, rootElem, elemName, elemPath):

    xmlErr = "Invalid {} file: missing element {}"
    elem = rootElem.find(elemName)
    if elem is None:
        raise ValueError(xmlErr.format(settingsFile, elemPath))
    return elem


if __name__ == "__main__":

    # Test stuff within here...
    recDir = "testrecording_2021-11-09_17-06-14"

    root = export2nwb(recDir)
