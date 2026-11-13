import imagej
import scyjava as sj
import numpy as np
import zarr

# 1. Initialize Fiji using the baked-in installation (starts in ~2 seconds)
ij = imagej.init('/opt/Fiji.app', mode='headless')

# 2. Import TrackMate Java classes directly into Python
Model = sj.jimport('fiji.plugin.trackmate.Model')
Settings = sj.jimport('fiji.plugin.trackmate.Settings')
TrackMate = sj.jimport('fiji.plugin.trackmate.TrackMate')
LogDetectorFactory = sj.jimport('fiji.plugin.trackmate.detection.LogDetectorFactory')
SparseLAPTrackerFactory = sj.jimport('fiji.plugin.trackmate.tracking.jaqaman.SparseLAPTrackerFactory')

# 3. Load a 4D (T, Z, Y, X) Zarr volume and convert to an ImageJ ImagePlus
z_arr = np.asarray(zarr.open('/kaggle/input/biohub-cell-tracking-during-development/test/sample.zarr', mode='r')['0'])
imp = ij.py.to_imageplus(z_arr)

# Set physical voxel calibration (matching your SCALE_ZYX: z=1.625, y=0.40625, x=0.40625 um)
imp.getCalibration().pixelWidth = 0.40625
imp.getCalibration().pixelHeight = 0.40625
imp.getCalibration().pixelDepth = 1.625

# 4. Configure & run TrackMate headlessly
model = Model()
settings = Settings(imp)
settings.detectorFactory = LogDetectorFactory()
settings.detectorSettings = {
    'DO_SUBPIXEL_LOCALIZATION': True,
    'RADIUS': 2.5,
    'TARGET_CHANNEL': 1,
    'THRESHOLD': 5.0,
    'DO_MEDIAN_FILTERING': False,
}
settings.trackerFactory = SparseLAPTrackerFactory()
settings.trackerSettings = settings.trackerFactory.getDefaultSettings()

tm = TrackMate(model, settings)
assert tm.checkInput() and tm.process(), f"TrackMate error: {tm.getErrorMessage()}"

# 5. Pull detected spots & tracks back into Python
spots = model.getSpots()
print(f"Detected {spots.getNSpots(True)} spots across {model.getTrackModel().nTracks(True)} tracks")