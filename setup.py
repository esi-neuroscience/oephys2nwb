# Builtins
import datetime
from setuptools import setup
from setuptools.config.setupcfg import read_configuration
import subprocess

# External packages
import yaml
from setuptools_scm import get_version

def _create_ymlDict(pkgList, envName, channels=["defaults", "conda-forge"], include_self=False):
    pipPkgs = []
    if include_self:
        pipPkgs.append(read_configuration("setup.cfg")["metadata"]["name"])
    condaPkgs = []
    for pkg in pkgList:
        pkg = "".join(pkg.split())
        if any(pkg in "".join(pipDep.split()) for pipDep in pipOnly):
            pipPkgs.append(pkg.replace("pip@", "git+"))
        else:
            condaPkgs.append(pkg)
    allDeps = condaPkgs + ["python " + str(setupOpts["python_requires"])]
    if len(pipPkgs) > 0:
        allDeps += ["pip", {"pip" : pipPkgs}]
    ymlDict = {"name" : envName,
               "channels" : channels,
               "dependencies" : allDeps}
    return ymlDict

# Set release version by hand
releaseVersion = "2022.5rc0"

# Read dependencies from setup.cfg and create conda environment files
setupOpts = read_configuration("setup.cfg")["options"]
pipOnly = ["pip@https://github.com/open-ephys/open-ephys-python-tools", "ndx-events", "sphinx_automodapi"]

# Create user-environment (no optional dependencies, include package itself)
# NOTE: PyPI does not allow git repos in dependencies, that's why we inject it here...
envUser = "oephys2nwb.yml"
ymlUser = _create_ymlDict(setupOpts["install_requires"], "oephys2nwb", include_self=True)
ymlUser["dependencies"][-1]["pip"].append('git+https://github.com/open-ephys/open-ephys-python-tools')

# Create developer-environment (include all optional dependencies but not actual package)
envDev = "oephys2nwb-dev.yml"
ymlDev = _create_ymlDict(setupOpts["install_requires"] + setupOpts["extras_require"]["dev"],
                         "oephys2nwb-dev", include_self=False)
ymlDev["dependencies"][-1]["pip"].append('git+https://github.com/open-ephys/open-ephys-python-tools')

# Write yml files
msg = "# This file was auto-generated by setup.py on {}. \n" +\
    "# Do not edit, all of your changes will be overwritten. \n"
msg = msg.format(datetime.datetime.now().strftime("%d/%m/%Y at %H:%M:%S"))
with open(envUser, 'w') as ymlFile:
    ymlFile.write(msg)
    yaml.dump(ymlUser, ymlFile, default_flow_style=False)
with open(envDev, 'w') as ymlFile:
    ymlFile.write(msg)
    yaml.dump(ymlDev, ymlFile, default_flow_style=False)

# If code has not been obtained via `git` or we're inside the main branch,
# use the hard-coded `releaseVersion` as version. Otherwise keep the local `tag.devx`
# scheme for TestPyPI uploads
proc = subprocess.run("git branch --show-current", shell=True, capture_output=True, text=True)
if proc.returncode !=0 or proc.stdout.strip() == "main":
    version = releaseVersion
    versionKws = {"use_scm_version" : False, "version" : version}
else:
    version = get_version(root='.', relative_to=__file__, local_scheme="no-local-version")
    versionKws = {"use_scm_version" : {"local_scheme": "no-local-version"}}

# Update citation file
citationFile = "CITATION.cff"
with open(citationFile) as ymlFile:
    ymlObj = yaml.safe_load(ymlFile)
ymlObj["version"] = version
ymlObj["date-released"] = datetime.datetime.now().strftime("%Y-%m-%d")
with open(citationFile, "w") as ymlFile:
    yaml.dump(ymlObj, ymlFile)

# Run setup (note: identical arguments supplied in setup.cfg will take precedence)
setup(
    setup_requires=['setuptools_scm'],
    **versionKws
)
