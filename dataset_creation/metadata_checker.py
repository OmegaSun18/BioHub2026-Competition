# ==================================================================================
# ------------------------------- prints the meta data for the original dataset ----
# ==================================================================================

import glob
import json
import os
import zarr

train_dir = '/kaggle/input/biohub-cell-tracking-during-development/train'
geff_files = sorted(glob.glob(os.path.join(train_dir, '*.geff')))
assert len(geff_files) > 0, f"No .geff files found in {train_dir}!"

print(f"Found {len(geff_files)} .geff files.\n")

for i, geff_path in enumerate(geff_files):
    print("=" * 70)
    print(f"[{i+1}/{len(geff_files)}] {os.path.basename(geff_path)}")
    print("=" * 70)

    # 1. Inspect raw JSON metadata files on disk inside the .geff directory
    for root_dir, _, files in os.walk(geff_path):
        rel_dir = os.path.relpath(root_dir, geff_path)
        for fname in sorted(files):
            if fname in ('zarr.json', '.zattrs', '.zgroup', '.zarray'):
                fpath = os.path.join(root_dir, fname)
                print(f"\n--- Metadata file: {os.path.join(rel_dir, fname)} ---")
                with open(fpath, 'r') as f:
                    print(json.dumps(json.load(f), indent=2))

    # 2. Open with Zarr to inspect the group hierarchy, shapes, and dtypes
    g = zarr.open(geff_path, mode='r')
    print("\n--- Zarr Group Hierarchy ---")
    print(g.tree())

    print("\n--- Group Attributes (g.attrs) ---")
    print(json.dumps(dict(g.attrs), indent=2))

    # Stop after the first file if you just want a quick peek at the schema
    # Remove `break` to dump metadata for all files
    break