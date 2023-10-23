 <!--
 Copyright (c) 2023 Ernst Strüngmann Institute (ESI) for Neuroscience
 in Cooperation with Max Planck Society
 SPDX-License-Identifier: CC-BY-NC-SA-1.0
 -->

# Changelog of oephys2nwb
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased: 0.6]
### NEW
### CHANGED
### REMOVED
### DEPRECATED
### FIXED

## [0.5] - 2023-10-23
### NEW
- added possibility to convert files without specific event channel
  information
- included SPDX licensing information in all project files
- updated build system to comply with PEP 517

### CHANGED
- use release branches instead of feature branches to support data created
  pre and post Open Ephys v 0.60
- switched to semantic versioning to differentiate main release branches
- write data in blocks by @KatharineShapcott #11

### FIXED
- use TTL sample_rate not message by @KatharineShapcott #14
- increase write speed x2 using channel index slice by @KatharineShapcott #10
- allow channel names to be remapped by @KatharineShapcott #9

## [2022.5rc0] - 2022-05-20
First release on PyPI

### NEW
- Initial oephys2nwb pre-release on PyPI

### CHANGED
- Made oephys2nwb GitHub repository public
