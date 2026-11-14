import copy
import glob
import os
import shutil
import numpy as np
import zarr

# ==============================================================================
# Configuration
# ==============================================================================
SRC_TRAIN_DIR = '/kaggle/input/biohub-cell-tracking-during-development/train'
OUT_BASE_DIR = '/kaggle/working/split_dataset'
DEV_DIR = os.path.join(OUT_BASE_DIR, 'dev')
TEST_DIR = os.path.join(OUT_BASE_DIR, 'test')

os.makedirs(DEV_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)


# ==============================================================================
# Helper 1: Write a GEFF slice (Zarr v3 + updated GEFF v1.1 metadata)
# ==============================================================================
def write_geff_slice(src_group, out_geff_path, node_mask, t_offset=0):
    if os.path.exists(out_geff_path):
        shutil.rmtree(out_geff_path)

    # Read original node IDs and edges
    orig_node_ids = np.asarray(src_group['nodes/ids'])
    orig_edges = np.asarray(src_group['edges/ids'])  # shape: (E, 2), uint64

    kept_node_ids = orig_node_ids[node_mask]
    kept_node_set = set(kept_node_ids.tolist())

    # Filter edges so both endpoints exist in kept_node_ids
    if orig_edges.shape[0] > 0 and len(kept_node_set) > 0:
        edge_mask = np.array(
            [(int(u) in kept_node_set) and (int(v) in kept_node_set) for u, v in orig_edges],
            dtype=bool,
        )
        kept_edges = orig_edges[edge_mask]
    else:
        edge_mask = np.zeros((orig_edges.shape[0],), dtype=bool)
        kept_edges = np.empty((0, 2), dtype=orig_edges.dtype)

    # Create Zarr v3 root group for the new .geff
    dst_group = zarr.open_group(out_geff_path, mode='w', zarr_format=3)

    # --- Write nodes ---
    nodes_grp = dst_group.create_group('nodes')
    n_nodes = int(kept_node_ids.shape[0])
    nodes_grp.create_array(
        name='ids',
        data=kept_node_ids,
        chunks=(max(1, n_nodes),),
    )

    node_props_src = src_group['nodes/props']
    node_props_dst = nodes_grp.create_group('props')

    prop_values_map = {}
    for prop_name in node_props_src.keys():
        prop_sub_src = node_props_src[prop_name]
        prop_sub_dst = node_props_dst.create_group(prop_name)

        vals = np.asarray(prop_sub_src['values'])[node_mask]
        if prop_name == 't' and t_offset != 0:
            vals = vals - np.int64(t_offset)
        prop_values_map[prop_name] = vals

        prop_sub_dst.create_array(
            name='values',
            data=vals,
            chunks=(max(1, n_nodes),),
        )
        if 'missing' in prop_sub_src:
            miss = np.asarray(prop_sub_src['missing'])[node_mask]
            prop_sub_dst.create_array(
                name='missing',
                data=miss,
                chunks=(max(1, n_nodes),),
            )

    # --- Write edges ---
    edges_grp = dst_group.create_group('edges')
    n_edges = int(kept_edges.shape[0])
    edges_grp.create_array(
        name='ids',
        data=kept_edges,
        chunks=(max(1, n_edges), 2),
    )

    edge_props_dst = edges_grp.create_group('props')
    if 'props' in src_group['edges']:
        edge_props_src = src_group['edges/props']
        for prop_name in edge_props_src.keys():
            eprop_sub_src = edge_props_src[prop_name]
            eprop_sub_dst = edge_props_dst.create_group(prop_name)
            evals = np.asarray(eprop_sub_src['values'])[edge_mask]
            eprop_sub_dst.create_array(
                name='values',
                data=evals,
                chunks=(max(1, n_edges),),
            )
            if 'missing' in eprop_sub_src:
                emiss = np.asarray(eprop_sub_src['missing'])[edge_mask]
                eprop_sub_dst.create_array(
                    name='missing',
                    data=emiss,
                    chunks=(max(1, n_edges),),
                )

    # --- Update GEFF metadata attributes in root zarr.json ---
    geff_meta = copy.deepcopy(dict(src_group.attrs).get('geff', {}))
    if 'axes' in geff_meta and isinstance(geff_meta['axes'], list):
        for axis in geff_meta['axes']:
            ax_name = axis.get('name')
            if ax_name in prop_values_map:
                arr = prop_values_map[ax_name]
                if arr.size > 0:
                    axis['min'] = float(arr.min())
                    axis['max'] = float(arr.max())
                else:
                    axis['min'] = None
                    axis['max'] = None

    if isinstance(geff_meta.get('extra'), dict) and 'estimated_number_of_nodes' in geff_meta['extra']:
        geff_meta['extra']['estimated_number_of_nodes'] = n_nodes

    dst_group.attrs['geff'] = geff_meta
    return n_nodes, n_edges


# ==============================================================================
# Helper 2: Split a .geff file into dev (0..49) and test (50..99 -> 0..49)
# ==============================================================================
def split_geff_file(src_geff_path, dev_geff_path, test_geff_path):
    g = zarr.open_group(src_geff_path, mode='r')
    node_ids = np.asarray(g['nodes/ids'])
    t_vals = np.asarray(g['nodes/props/t/values'])
    edges = np.asarray(g['edges/ids'])  # (E, 2): [src_id, dst_id]

    id_to_t = dict(zip(node_ids.tolist(), t_vals.tolist()))

    # Identify boundary-crossing annotations: source at t=49, target at t=50
    bad_t49_node_ids = set()
    bad_t50_node_ids = set()
    for u, v in edges:
        u_int, v_int = int(u), int(v)
        tu = id_to_t.get(u_int)
        tv = id_to_t.get(v_int)
        if tu == 49 and tv == 50:
            bad_t49_node_ids.add(u_int)
            bad_t50_node_ids.add(v_int)

    # Development mask: 0 <= t <= 49, excluding boundary t=49 source nodes
    dev_mask = (t_vals >= 0) & (t_vals <= 49)
    if bad_t49_node_ids:
        dev_mask &= ~np.isin(node_ids, list(bad_t49_node_ids))

    # Test mask: 50 <= t <= 99, excluding boundary t=50 target nodes
    test_mask = (t_vals >= 50) & (t_vals <= 99)
    if bad_t50_node_ids:
        test_mask &= ~np.isin(node_ids, list(bad_t50_node_ids))

    dev_stats = write_geff_slice(g, dev_geff_path, dev_mask, t_offset=0)
    test_stats = write_geff_slice(g, test_geff_path, test_mask, t_offset=50)
    return len(bad_t49_node_ids), dev_stats, test_stats


# ==============================================================================
# Helper 3: Split a raw .zarr file into dev (0:50) and test (50:100 -> 0:50)
# ==============================================================================
def write_zarr_time_slice(src_root, out_zarr_path, t_start, t_end):
    if os.path.exists(out_zarr_path):
        shutil.rmtree(out_zarr_path)

    z_fmt = src_root.metadata.zarr_format
    dst_root = zarr.open_group(out_zarr_path, mode='w', zarr_format=z_fmt)
    dst_root.attrs.update(copy.deepcopy(dict(src_root.attrs)))

    def copy_node(src_node, dst_grp, name):
        if isinstance(src_node, zarr.Group):
            sub_grp = dst_grp.create_group(name)
            sub_grp.attrs.update(copy.deepcopy(dict(src_node.attrs)))
            for child_key in src_node.keys():
                copy_node(src_node[child_key], sub_grp, child_key)
        elif isinstance(src_node, zarr.Array):
            total_t = src_node.shape[0]
            actual_start = min(t_start, total_t)
            actual_end = min(t_end, total_t)
            new_t = max(0, actual_end - actual_start)
            new_shape = (new_t,) + src_node.shape[1:]

            orig_chunks = src_node.chunks
            new_chunks = (max(1, min(orig_chunks[0], new_t)),) + orig_chunks[1:]

            dst_arr = dst_grp.create_array(
                name=name,
                shape=new_shape,
                chunks=new_chunks,
                dtype=src_node.dtype,
            )
            dst_arr.attrs.update(copy.deepcopy(dict(src_node.attrs)))

            # Stream time-chunks to keep memory footprint low
            step = new_chunks[0]
            for offset in range(0, new_t, step):
                chunk_len = min(step, new_t - offset)
                dst_arr[offset : offset + chunk_len] = src_node[
                    actual_start + offset : actual_start + offset + chunk_len
                ]

    for key in src_root.keys():
        copy_node(src_root[key], dst_root, key)


def split_zarr_file(src_zarr_path, dev_zarr_path, test_zarr_path):
    src_root = zarr.open_group(src_zarr_path, mode='r')
    total_t = src_root['0'].shape[0]
    write_zarr_time_slice(src_root, dev_zarr_path, 0, 50)
    write_zarr_time_slice(src_root, test_zarr_path, 50, 100)
    return total_t


# ==============================================================================
# Main Execution Loop Across All Samples
# ==============================================================================
geff_files = sorted(glob.glob(os.path.join(SRC_TRAIN_DIR, '*.geff')))
assert len(geff_files) > 0, f"No .geff files found in {SRC_TRAIN_DIR}!"

print(f"Splitting {len(geff_files)} samples into:\n  DEV : {DEV_DIR}\n  TEST: {TEST_DIR}\n")

total_boundary_removed = 0
for idx, src_geff_path in enumerate(geff_files, start=1):
    stem = os.path.splitext(os.path.basename(src_geff_path))[0]
    src_zarr_path = os.path.join(SRC_TRAIN_DIR, f"{stem}.zarr")
    assert os.path.exists(src_zarr_path), f"Missing matching .zarr for {stem}"

    dev_geff_path = os.path.join(DEV_DIR, f"{stem}.geff")
    test_geff_path = os.path.join(TEST_DIR, f"{stem}.geff")
    dev_zarr_path = os.path.join(DEV_DIR, f"{stem}.zarr")
    test_zarr_path = os.path.join(TEST_DIR, f"{stem}.zarr")

    n_removed, (dev_n, dev_e), (test_n, test_e) = split_geff_file(
        src_geff_path, dev_geff_path, test_geff_path
    )
    total_t = split_zarr_file(src_zarr_path, dev_zarr_path, test_zarr_path)
    total_boundary_removed += n_removed

    if idx <= 5 or idx % 25 == 0 or idx == len(geff_files):
        print(
            f"[{idx:>3}/{len(geff_files)}] {stem} (T={total_t}) | "
            f"Boundary links cut: {n_removed} | "
            f"DEV: {dev_n} nodes, {dev_e} edges | "
            f"TEST: {test_n} nodes, {test_e} edges"
        )

print("\nDone! Total boundary-crossing track nodes removed per side:", total_boundary_removed)