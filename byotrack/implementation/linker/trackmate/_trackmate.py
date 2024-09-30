# @ String detections
# @ String parameters
# @ String tracks

# pylint: skip-file
# type: ignore

import json
import sys

from java.io import File

from ij import IJ

from fiji.plugin.trackmate import Model
from fiji.plugin.trackmate import Settings
from fiji.plugin.trackmate import TrackMate
from fiji.plugin.trackmate import Logger
from fiji.plugin.trackmate.io import TmXmlWriter
from fiji.plugin.trackmate.action import ExportTracksToXML
from fiji.plugin.trackmate.detection import LabelImageDetectorFactory
from fiji.plugin.trackmate.tracking.jaqaman import SparseLAPTrackerFactory
from fiji.plugin.trackmate.tracking.kalman import AdvancedKalmanTrackerFactory

# We have to do the following to avoid errors with UTF8 chars generated in
# TrackMate that will mess with our Fiji Jython.
reload(sys)
sys.setdefaultencoding("utf-8")


print("Hello from ImageJ/Fiji")

print("Loading detections from", detections)

# Load detections as an image (As we use LabelImageDetector)
imp = IJ.openImage(detections)

# ----------------------------
# Create the model object now
# ----------------------------

# Some of the parameters we configure below need to have
# a reference to the model at creation. So we create an
# empty model now.

model = Model()

# Send all messages to ImageJ log window.
model.setLogger(Logger.DEFAULT_LOGGER)


# ------------------------
# Prepare settings object
# ------------------------

settings = Settings(imp)

# Configure detector - Let's use a LabelImageDetector (detections is already done)
settings.detectorFactory = LabelImageDetectorFactory()
settings.detectorSettings = {
    "TARGET_CHANNEL": 1,
    "SIMPLIFY_CONTOURS": False,
}

# Configure tracker from the parameters

## First: Load the parameters
with open(parameters, "r") as file:
    specs = json.load(file)

if specs["kalman_search_radius"] is None:  # Without kalman -> Use SparseLAPTracker
    settings.trackerFactory = SparseLAPTrackerFactory()
    specs.pop("kalman_search_radius")
else:
    settings.trackerFactory = AdvancedKalmanTrackerFactory()

## Generate default settings
settings.trackerSettings = settings.trackerFactory.getDefaultSettings()

# Override settings
for key, value in specs.items():
    settings.trackerSettings[key.upper()] = value

print("Settings:")
print(settings.trackerSettings)


# -------------------
# Instantiate plugin
# -------------------

trackmate = TrackMate(model, settings)

# --------
# Process
# --------

ok = trackmate.checkInput()
if not ok:
    sys.exit(str(trackmate.getErrorMessage()))

ok = trackmate.process()
if not ok:
    sys.exit(str(trackmate.getErrorMessage()))


# -------------
# Save results
# -------------

# Could use this function as it outputs a cleaner results... but not the one used in GUI so lets keep the other one
# ExportTracksToXML.export(model, settings, File(tracks))

writer = TmXmlWriter(File(tracks))
writer.appendSettings(settings)
writer.appendModel(model)
writer.writeToFile()
