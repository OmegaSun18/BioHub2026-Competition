# ===========================================================================
# ------------------- Old code to fix the estimated number of nodes manually-
# ===========================================================================

import copy
import glob
import os
import zarr

ORIG_TRAIN_DIR = '/kaggle/input/biohub-cell-tracking-during-development/train'
DEV_DIR = '/kaggle/working/split_dataset/dev'
TEST_DIR = '/kaggle/working/split_dataset/test'

orig_geffs = sorted(glob.glob(os.path.join(ORIG_TRAIN_DIR, '*.geff')))
assert len(orig_geffs) > 0, f"No .geff files found in {ORIG_TRAIN_DIR}!"


def update_estimated_nodes(geff_path: str, new_est: int) -> None:
    g = zarr.open_group(geff_path, mode='r+')
    geff_meta = copy.deepcopy(dict(g.attrs).get('geff', {}))
    geff_meta.setdefault('extra', {})['estimated_number_of_nodes'] = int(new_est)
    # Reassign the top-level 'geff' key so Zarr v3 flushes the updated zarr.json
    g.attrs['geff'] = geff_meta


tot_orig_est = 0
tot_dev_est = 0
tot_test_est = 0

for orig_path in orig_geffs:
    stem = os.path.splitext(os.path.basename(orig_path))[0]
    dev_path = os.path.join(DEV_DIR, f"{stem}.geff")
    test_path = os.path.join(TEST_DIR, f"{stem}.geff")

    orig_g = zarr.open_group(orig_path, mode='r')
    orig_est = int(dict(orig_g.attrs)['geff']['extra']['estimated_number_of_nodes'])

    # Half for dev and half for test (preserving exact sum when orig_est is odd)
    dev_est = orig_est // 2
    test_est = orig_est - dev_est

    update_estimated_nodes(dev_path, dev_est)
    update_estimated_nodes(test_path, test_est)

    # Re-open in read-only mode to verify zarr.json was updated on disk
    ver_dev = int(dict(zarr.open_group(dev_path, mode='r').attrs)['geff']['extra']['estimated_number_of_nodes'])
    ver_test = int(dict(zarr.open_group(test_path, mode='r').attrs)['geff']['extra']['estimated_number_of_nodes'])
    assert ver_dev == dev_est and ver_test == test_est, f"Failed to persist metadata for {stem}!"
    assert ver_dev + ver_test == orig_est, f"Sum mismatch on {stem}: {ver_dev} + {ver_test} != {orig_est}"

    tot_orig_est += orig_est
    tot_dev_est += ver_dev
    tot_test_est += ver_test

print(f"Updated and verified {len(orig_geffs)} dev/test .geff pairs:")
print(f"  Original Train estimated_number_of_nodes : {tot_orig_est:,}")
print(f"  New Dev Split  estimated_number_of_nodes : {tot_dev_est:,}")
print(f"  New Test Split estimated_number_of_nodes : {tot_test_est:,}")
print(f"  Dev + Test Sum                           : {tot_dev_est + tot_test_est:,}")