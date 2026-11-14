# ==================================================================================
# ---------------------- Checks the number of annotions at each time frame ---------
# ----------------------- for the orignal dataset ----------------------------------
# ==================================================================================

import glob
import os
from collections import Counter
import numpy as np
import zarr

train_dir = '/kaggle/input/biohub-cell-tracking-during-development/train'
geff_files = sorted(glob.glob(os.path.join(train_dir, '*.geff')))
assert len(geff_files) > 0, f"No .geff files found in {train_dir}!"

total_annotations_per_t = Counter()
files_present_per_t = Counter()
t50_file_breakdown = []

for geff_path in geff_files:
    fname = os.path.basename(geff_path)
    g = zarr.open_group(geff_path, mode='r')
    
    # Read the 1D int64 array of time steps for all annotated nodes
    t_vals = np.asarray(g['nodes/props/t/values'])
    
    unique_t, counts = np.unique(t_vals, return_counts=True)
    for t, count in zip(unique_t, counts):
        t_int = int(t)
        total_annotations_per_t[t_int] += int(count)
        files_present_per_t[t_int] += 1
        if t_int == 50:
            t50_file_breakdown.append((fname, int(count)))

# 1. Summary specifically for time step 50
total_t50_nodes = total_annotations_per_t.get(50, 0)
num_t50_files = files_present_per_t.get(50, 0)

print("=" * 65)
print(f"TIME STEP 50 SUMMARY (across {len(geff_files)} .geff files)")
print("=" * 65)
print(f"Total annotated nodes at t = 50 : {total_t50_nodes}")
print(f"Files with annotations at t = 50: {num_t50_files} / {len(geff_files)}")

if t50_file_breakdown:
    print("\nBreakdown of files containing t = 50:")
    for fname, cnt in sorted(t50_file_breakdown, key=lambda x: -x[1]):
        print(f"  {fname:<25} -> {cnt} annotations")

# 2. Overall distribution of annotated time steps across the dataset
print("\n" + "=" * 65)
print("ANNOTATED TIME STEPS ACROSS ALL FILES")
print("=" * 65)
print(f"{'Time Step (t)':<15} | {'Files Present':<15} | {'Total Annotations':<18}")
print("-" * 55)
for t in sorted(total_annotations_per_t.keys()):
    marker = "  <-- t=50" if t == 50 else ""
    print(f"{t:<15} | {files_present_per_t[t]:<15} | {total_annotations_per_t[t]:<18}{marker}")