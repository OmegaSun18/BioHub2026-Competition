import os
import glob
import jpype
import scyjava as sj
import imagej
import numpy as np
import zarr

# 1. Configure Java 21 BEFORE initializing Fiji
sj.config.set_java_constraints(fetch=True, vendor='zulu', version='21')
sj.config.add_option('-Djava.awt.headless=true')

# 2. Initialize Fiji headlessly
ij = imagej.init('/opt/Fiji.app', mode='headless')
print("Fiji initialized! Version:", ij.getVersion())

# 3. Import TrackMate Java classes
Model = sj.jimport('fiji.plugin.trackmate.Model')
Settings = sj.jimport('fiji.plugin.trackmate.Settings')
TrackMate = sj.jimport('fiji.plugin.trackmate.TrackMate')
LogDetectorFactory = sj.jimport('fiji.plugin.trackmate.detection.LogDetectorFactory')
SparseLAPTrackerFactory = sj.jimport('fiji.plugin.trackmate.tracking.jaqaman.SparseLAPTrackerFactory')
File = sj.jimport('java.io.File')
TmXmlWriter = sj.jimport('fiji.plugin.trackmate.io.TmXmlWriter')

# 4. Load the first 5 frames of the first .zarr dataset
test_zarrs = sorted(glob.glob('/kaggle/input/biohub-cell-tracking-during-development/test/*.zarr'))
assert len(test_zarrs) > 0, "No .zarr files found!"
sample_path = test_zarrs[0]

z_store = zarr.open(sample_path, mode='r')['0']
print("Full Zarr shape (T, Z, Y, X):", z_store.shape)

N_FRAMES = min(5, z_store.shape[0])
z_arr = np.asarray(z_store[:N_FRAMES])  # Shape: (T, Z, Y, X)
T, Z, Y, X = z_arr.shape

# Convert to ImagePlus and explicitly enforce (Channels=1, Slices=Z, Frames=T)
imp = ij.py.to_imageplus(z_arr)
imp.setDimensions(1, Z, T)
imp.getCalibration().pixelWidth = 0.40625
imp.getCalibration().pixelHeight = 0.40625
imp.getCalibration().pixelDepth = 1.625
print(f"ImagePlus configured -> Channels: {imp.getNChannels()}, Z-slices: {imp.getNSlices()}, Frames (T): {imp.getNFrames()}")

# 5. Configure TrackMate
model = Model()
settings = Settings(imp)

settings.detectorFactory = LogDetectorFactory()
det_settings = settings.detectorFactory.getDefaultSettings()
det_settings.put('DO_SUBPIXEL_LOCALIZATION', True)
det_settings.put('RADIUS', 2.5)
det_settings.put('TARGET_CHANNEL', jpype.JInt(1))
det_settings.put('THRESHOLD', 5.0)
det_settings.put('DO_MEDIAN_FILTERING', False)
settings.detectorSettings = det_settings

settings.trackerFactory = SparseLAPTrackerFactory()
trk_settings = settings.trackerFactory.getDefaultSettings()
trk_settings.put('LINKING_MAX_DISTANCE', 15.0)
trk_settings.put('GAP_CLOSING_MAX_DISTANCE', 15.0)
trk_settings.put('MAX_FRAME_GAP', jpype.JInt(2))
settings.trackerSettings = trk_settings

# 6. Run TrackMate
tm = TrackMate(model, settings)
assert tm.checkInput() and tm.process(), f"TrackMate error: {tm.getErrorMessage()}"

spots = model.getSpots()
n_tracks = model.getTrackModel().nTracks(True)
print(f"Success! Detected {spots.getNSpots(True)} spots across {n_tracks} tracks over {T} frames")

# 7. Save XML output to /kaggle/working (/mnt/data/biohub_working)
out_file = File('/kaggle/working/sanity_test_tracks.xml')
writer = TmXmlWriter(out_file)
writer.appendModel(model)
writer.appendSettings(settings)
writer.writeToFile()
print("Saved TrackMate XML to /kaggle/working/sanity_test_tracks.xml")