import glob
import os
import numpy as np
import zarr

ORIG_TRAIN_DIR = '/kaggle/input/biohub-cell-tracking-during-development/train'
DEV_DIR = '/kaggle/working/split_dataset/dev'
TEST_DIR = '/kaggle/working/split_dataset/test'

dev_geffs = sorted(glob.glob(os.path.join(DEV_DIR, '*.geff')))
assert len(dev_geffs) > 0, "No split .geff files found!"

empty_test_files = []
short_test_files = []

tot_orig_est = 0
tot_dev_est = 0
tot_test_est = 0


def validate_geff_with_zarr(
    geff_path: str,
    expected_estimated_nodes: int,
    expected_max_t: int = 49,
) -> tuple[int, int, int]:
    g = zarr.open_group(geff_path, mode='r')
    geff_meta = dict(g.attrs).get('geff', {})

    # 1. Read node IDs, coordinate arrays, and edge pairs directly via Zarr
    node_ids = np.asarray(g['nodes/ids'])
    t_vals = np.asarray(g['nodes/props/t/values'])
    z_vals = np.asarray(g['nodes/props/z/values'])
    y_vals = np.asarray(g['nodes/props/y/values'])
    x_vals = np.asarray(g['nodes/props/x/values'])
    edges = np.asarray(g['edges/ids'])

    n_nodes = int(node_ids.shape[0])
    n_edges = int(edges.shape[0])

    # 2. Verify all node property arrays match nodes/ids length
    for prop_name, arr in [('t', t_vals), ('z', z_vals), ('y', y_vals), ('x', x_vals)]:
        assert arr.shape == (n_nodes,), f"{geff_path}: '{prop_name}' shape {arr.shape} != ({n_nodes},)"

    # 3. Verify metadata estimated_number_of_nodes & axes min/max bounds
    actual_est = int(geff_meta['extra']['estimated_number_of_nodes'])
    assert actual_est == expected_estimated_nodes, (
        f"{geff_path}: estimated_number_of_nodes {actual_est} != expected {expected_estimated_nodes}!"
    )

    prop_map = {'t': t_vals, 'z': z_vals, 'y': y_vals, 'x': x_vals}
    for axis in geff_meta.get('axes', []):
        ax_name = axis['name']
        if ax_name in prop_map:
            arr = prop_map[ax_name]
            if n_nodes > 0:
                assert axis['min'] == float(arr.min()), f"{geff_path}: {ax_name} min mismatch!"
                assert axis['max'] == float(arr.max()), f"{geff_path}: {ax_name} max mismatch!"
            else:
                assert axis['min'] is None and axis['max'] is None

    # 4. Verify time step bounds (0..49)
    if n_nodes > 0:
        assert t_vals.min() >= 0 and t_vals.max() <= expected_max_t, (
            f"{geff_path}: out-of-range time steps [{t_vals.min()}, {t_vals.max()}]"
        )

    # 5. Verify no dangling edges exist
    if n_edges > 0:
        assert edges.ndim == 2 and edges.shape[1] == 2
        valid_ids = set(node_ids.tolist())
        assert all((int(u) in valid_ids) and (int(v) in valid_ids) for u, v in edges), (
            f"{geff_path}: dangling edge endpoint found!"
        )

    return n_nodes, n_edges, actual_est


for dev_geff_path in dev_geffs:
    stem = os.path.splitext(os.path.basename(dev_geff_path))[0]
    orig_geff_path = os.path.join(ORIG_TRAIN_DIR, f"{stem}.geff")
    test_geff_path = os.path.join(TEST_DIR, f"{stem}.geff")
    dev_zarr_path = os.path.join(DEV_DIR, f"{stem}.zarr")
    test_zarr_path = os.path.join(TEST_DIR, f"{stem}.zarr")

    # Read original estimated_number_of_nodes and compute expected dev/test halves
    orig_g = zarr.open_group(orig_geff_path, mode='r')
    orig_est = int(dict(orig_g.attrs)['geff']['extra']['estimated_number_of_nodes'])
    expected_dev_est = orig_est // 2
    expected_test_est = orig_est - expected_dev_est

    # Validate both GEFF stores purely via Zarr
    dev_nodes, dev_edges, dev_est = validate_geff_with_zarr(dev_geff_path, expected_dev_est)
    test_nodes, test_edges, test_est = validate_geff_with_zarr(test_geff_path, expected_test_est)

    assert dev_est + test_est == orig_est, (
        f"{stem}: dev_est ({dev_est}) + test_est ({test_est}) != orig_est ({orig_est})"
    )

    tot_orig_est += orig_est
    tot_dev_est += dev_est
    tot_test_est += test_est

    # Check raw Zarr frame counts
    dev_t = zarr.open_group(dev_zarr_path, mode='r')['0'].shape[0]
    test_t = zarr.open_group(test_zarr_path, mode='r')['0'].shape[0]

    if test_t == 0 or test_nodes == 0:
        empty_test_files.append((stem, dev_t, test_t, test_nodes))
    elif test_t < 50:
        short_test_files.append((stem, dev_t, test_t, test_nodes))

print(f"Validated {len(dev_geffs)} paired dev/test files purely with zarr!")
print(f"Estimated nodes total -> Original: {tot_orig_est:,} | Dev: {tot_dev_est:,} | Test: {tot_test_est:,} | Sum: {tot_dev_est + tot_test_est:,}")
print(f"Files with EMPTY test split (T_test=0 or 0 nodes): {len(empty_test_files)}")
print(f"Files with PARTIAL test split (1 <= T_test < 50) : {len(short_test_files)}")

if empty_test_files[:5]:
    print("Sample empty test files (stem, dev_T, test_T, test_nodes):", empty_test_files[:5])
if short_test_files[:5]:
    print("Sample partial test files (stem, dev_T, test_T, test_nodes):", short_test_files[:5])