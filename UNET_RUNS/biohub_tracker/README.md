# biohub_tracker (v0)
Separate from biohub_unet: reads `detections/<video>.parquet` (columns t,z,y,x,z_um,y_um,x_um,score[,dup_frame]), writes `.geff`, scores with the organisers' metric
(`tracker/official/` = tracking_cellmot metrics.py + division_metrics.py, BSD-3, unchanged).
v0 = per-track constant-velocity Kalman filter + one assignment per frame pair that decides continue / divide / birth / death jointly. No appearance terms yet (v1-v4 add cost terms in tracker/link.py).
Repeated frames (dup_frame): zero time step, no filter update. Only nodes that have an edge are written (isolated detections only cost node-count penalty).
Defaults in LinkConfig are guesses; tune on dev with `tracker.cli sweep`, never on test.

## v1-v4 (appearance)
`python -m tracker.features` samples raw intensity, P, DoG and erosion at every detection with the model's own filters (needs the biohub package) -> `features/<video>.parquet`, row-aligned with the detections.
`tracker.cli run|sweep --variant v1..v4 --feat-dir ...` adds  w_app * sum_f min((dlog f)^2 / app_sigma^2, cap)  to every link cost. `scripts/run_variants_dev.sh` holds the v0 best fixed and sweeps only w_app.
