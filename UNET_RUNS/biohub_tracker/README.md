# biohub_tracker (v0)
Separate from biohub_unet: reads `detections/<video>.parquet` (columns t,z,y,x,z_um,y_um,x_um,score[,dup_frame]), writes `.geff`, scores with the organisers' metric
(`tracker/official/` = tracking_cellmot metrics.py + division_metrics.py, BSD-3, unchanged).
v0 = per-track constant-velocity Kalman filter + one assignment per frame pair that decides continue / divide / birth / death jointly. No appearance terms yet (v1-v4 add cost terms in tracker/link.py).
Repeated frames (dup_frame): zero time step, no filter update. Only nodes that have an edge are written (isolated detections only cost node-count penalty).
Defaults in LinkConfig are guesses; tune on dev with `tracker.cli sweep`, never on test.
