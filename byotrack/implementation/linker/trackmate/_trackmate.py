# @String detections
# @String tracks
# @String max_link_cost
# @String max_gap
# @String max_gap_cost

# pylint: skip-file
# type: ignore

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

# We have to do the following to avoid errors with UTF8 chars generated in
# TrackMate that will mess with our Fiji Jython.
reload(sys)
sys.setdefaultencoding("utf-8")


print("Hello from ImageJ/Fiji")

print("Loading detections from temp file", detections)

# Load detections as an image (As we use LabelImageDetector)
imp = IJ.openImage(detections)
imp.setStack(imp.getStack(), 1, 1, imp.getStackSize())
imp.show()

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

# Configure detector - We use the Strings for the keys
settings.detectorFactory = LabelImageDetectorFactory()
settings.detectorSettings = {
    "TARGET_CHANNEL": 1,
    "SIMPLIFY_CONTOURS": False,
}

# Configure tracker - We want to allow merges and fusions
settings.trackerFactory = SparseLAPTrackerFactory()
settings.trackerSettings = settings.trackerFactory.getDefaultSettings()  # almost good enough
settings.trackerSettings["ALLOW_TRACK_SPLITTING"] = False
settings.trackerSettings["ALLOW_TRACK_MERGING"] = True

settings.trackerSettings["LINKING_MAX_DISTANCE"] = float(
    max_link_cost
)  # The max distance between two consecutive spots, in physical units, allowed for creating links.
settings.trackerSettings[
    "ALLOW_GAP_CLOSING"
] = True  # If True then the tracker will perform gap-closing, linking tracklets or segments separated by more than one frame.
settings.trackerSettings["MAX_FRAME_GAP"] = int(
    max_gap
)  # Gap-closing time-distance. The max difference in time-points between two spots to allow for linking. For instance a value of 2 means that the tracker will be able to make a link between a spot in frame t and a successor spots in frame t+2, effectively bridging over one missed detection in one frame.
settings.trackerSettings["GAP_CLOSING_MAX_DISTANCE"] = float(
    max_gap_cost
)  #  Gap-closing max spatial distance. The max distance between two spots, in physical units, allowed for creating links over missing detections.

# Do not allow merging or splitting
settings.trackerSettings[
    "ALLOW_TRACK_MERGING"
] = False  # If True then the tracker will perform tracklets or segments merging, that is: have two or more tracklet endings linking to one tracklet beginning. This leads to tracks possibly fusing together across time.
settings.trackerSettings[
    "ALLOW_TRACK_SPLITTING"
] = False  # If True then the tracker will perform tracklets or segments splitting, that is: have one tracklet ending linking to two or more tracklet beginnings . This leads to tracks possibly separating into several sub-tracks across time, like in cell division.

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
