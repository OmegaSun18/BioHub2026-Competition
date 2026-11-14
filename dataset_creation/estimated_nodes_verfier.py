import glob
import os
import zarr

ORIG_TRAIN_DIR = '/kaggle/input/biohub-cell-tracking-during-development/train'
DEV_DIR = '/kaggle/working/split_dataset/dev'
TEST_DIR = '/kaggle/working/split_dataset/test'

orig_geffs = sorted(glob.glob(os.path.join(ORIG_TRAIN_DIR, '*.geff')))
assert len(orig_geffs) > 0, f"No .geff files found in {ORIG_TRAIN_DIR}!"

def get_node_counts(geff_path: str) -> tuple[int | None, int]:
    g = zarr.open_group(geff_path, mode='r')
    geff_meta = dict(g.attrs).get('geff', {})
    est_nodes = geff_meta.get('extra', {}).get('estimated_number_of_nodes')
    actual_nodes = int(g['nodes/ids'].shape[0])
    return est_nodes, actual_nodes

tot_orig_est, tot_orig_act = 0, 0
tot_dev_est, tot_dev_act = 0, 0
tot_test_est, tot_test_act = 0, 0

print(f"{'Dataset Stem':<16} | {'Orig (Est / Act)':<20} | {'Dev (Est / Act)':<18} | {'Test (Est / Act)':<18}")
print("-" * 80)

for idx, orig_path in enumerate(orig_geffs):
    stem = os.path.splitext(os.path.basename(orig_path))[0]
    dev_path = os.path.join(DEV_DIR, f"{stem}.geff")
    test_path = os.path.join(TEST_DIR, f"{stem}.geff")

    o_est, o_act = get_node_counts(orig_path)
    d_est, d_act = get_node_counts(dev_path)
    t_est, t_act = get_node_counts(test_path)

    tot_orig_est += o_est or 0
    tot_orig_act += o_act
    tot_dev_est += d_est or 0
    tot_dev_act += d_act
    tot_test_est += t_est or 0
    tot_test_act += t_act

    # Print the first 15 files as a sample breakdown
    if idx < 15:
        print(
            f"{stem:<16} | {str(o_est):>8} / {o_act:<9} | "
            f"{str(d_est):>7} / {d_act:<8} | {str(t_est):>7} / {t_act:<8}"
        )

print("=" * 80)
print(f"TOTALS ACROSS ALL {len(orig_geffs)} FILES:")
print(f"  Original Train : estimated = {tot_orig_est:,} | actual = {tot_orig_act:,}")
print(f"  New Dev Split  : estimated = {tot_dev_est:,} | actual = {tot_dev_act:,}")
print(f"  New Test Split : estimated = {tot_test_est:,} | actual = {tot_test_act:,}")
print(f"  Dev + Test Sum : estimated = {tot_dev_est + tot_test_est:,} | actual = {tot_dev_act + tot_test_act:,}")