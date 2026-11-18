# Cell Tracking Evaluation & Assertion Benchmark Report

- **Generated (UTC)**: `2026-11-18 17:11:58 UTC`
- **Ground Truth Directory**: `/kaggle/working/split_dataset/test`
- **Total Variants Evaluated**: `118` (`20` Fiji tracker runs + `98` submission tracker runs)
- **Node Count Tolerance Threshold (`t`)**: `0.15` (fixed; threshold fine-tuning disabled)
- **Companion Plot-Ready CSVs**:
  - Aggregate Summary Table: `/mnt/data/evaluation_summary_table.csv`
  - Per-Video Assertion Metrics Table: `/mnt/data/evaluation_per_video_metrics.csv`

## Assertion Definitions

- **A1 (Node Count Tolerance)**: `|N_pred - N_est| / N_est <= 0.15`, using `estimated_number_of_nodes` from each ground-truth `.geff`.
- **A2 (No Cell Merges)**: Every node has in-degree `<= 1` (two distinct source node IDs cannot point to the same target node).
- **A3 (No Duplicate Node IDs)**: No duplicate `node_id` entries unless consecutive in time (`t, t+1, ...`) within the same unbranched track.
- **A4 (Division Branching & Cooldown)**: Every splitting parent (`out_degree > 1`) has `out_degree == 2` (two children), and neither child immediately splits (`out_degree <= 1`).
- **A5 (Temporal Ordering)**: Every parent node is strictly earlier in time than its children (`t_parent < t_child`) and never reappears at or after a child's frame.
- **A6 (Spatial Proximity <= 14 um)**: Every parent-child edge distance in micrometers (`voxel_delta * scale_zyx` from `.zarr`) is `<= 14.0 um`.

## 1. Combined Leaderboard (All Detector + Tracker Variants)

| source_group | detector | tracker | n_videos | aggregate_score | edge_jaccard | division_jaccard | threshold_t | overall_assertion_pass_rate | a1_pass_rate | a2_pass_rate | a3_pass_rate | a4_pass_rate | a5_pass_rate | a6_pass_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| submissions | temporal_diff+erode@small | v4 | 199 | 0.8529 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| submissions | temporal_diff+erode@small | v3 | 199 | 0.8499 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9849 |
| submissions | temporal_diff+dog@small | v2 | 199 | 0.8495 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9950 |
| submissions | temporal_diff+erode@small | v2 | 199 | 0.8488 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9849 |
| submissions | temporal_diff+erode@small | v0 | 199 | 0.8485 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9698 |
| submissions | temporal_diff+dog@small | v4 | 199 | 0.8464 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| submissions | temporal_diff+erode@small | v1 | 199 | 0.8460 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9749 |
| submissions | temporal_diff+dog@small | v0 | 199 | 0.8459 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.9950 | 1.0000 | 0.9849 |
| submissions | temporal_diff+dog@small | v3 | 199 | 0.8458 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9950 |
| submissions | temporal_diff+dog@small | v1 | 199 | 0.8430 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.9950 | 1.0000 | 0.9899 |
| submissions | raw_only+dog+edge+erode@small | v4 | 199 | 0.8284 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9899 |
| submissions | raw_only+dog+edge+erode@small | v2 | 199 | 0.8267 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9799 |
| submissions | raw_only+dog+edge+erode@small | v3 | 199 | 0.8266 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9849 |
| submissions | raw_only+dog+edge+erode@small | v1 | 199 | 0.8238 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.9950 | 1.0000 | 0.9749 |
| submissions | raw_only+dog+edge+erode@small | v0 | 199 | 0.8237 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.9849 | 1.0000 | 0.9749 |
| submissions | temporal_stack+erode@small | v4 | 199 | 0.8229 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| submissions | temporal_stack+erode@small | v3 | 199 | 0.8225 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9849 |
| submissions | temporal_stack+erode@small | v2 | 199 | 0.8212 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.9950 | 1.0000 | 0.9849 |
| submissions | temporal_stack+erode@small | v1 | 199 | 0.8182 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.9950 | 1.0000 | 0.9849 |
| submissions | temporal_stack+erode@small | v0 | 199 | 0.8175 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.9950 | 1.0000 | 0.9849 |
| submissions | temporal_diff+erode@medium-4000 | v3 | 199 | 0.7941 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.9950 | 1.0000 | 0.9899 |
| submissions | temporal_diff+erode@medium-4000 | v4 | 199 | 0.7932 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.9950 | 1.0000 | 0.9849 |
| submissions | temporal_diff+erode@medium-4000 | v2 | 199 | 0.7929 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.9950 | 1.0000 | 0.9799 |
| submissions | temporal_stack+erode@large-6000 | v4 | 199 | 0.7905 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9950 |
| submissions | temporal_diff+erode@medium-4000 | v0 | 199 | 0.7902 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.9950 | 1.0000 | 0.9648 |
| submissions | temporal_diff+erode@medium-4000 | v1 | 199 | 0.7890 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.9950 | 1.0000 | 0.9749 |
| submissions | temporal_stack+erode@large-6000 | v3 | 199 | 0.7862 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.9950 | 1.0000 | 0.9950 |
| submissions | temporal_stack+erode@large-6000 | v2 | 199 | 0.7818 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.9950 | 1.0000 | 0.9950 |
| submissions | temporal_stack+erode@large-6000 | v1 | 199 | 0.7783 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.9950 | 1.0000 | 0.9648 |
| submissions | temporal_stack+erode@large-6000 | v0 | 199 | 0.7716 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.9749 | 1.0000 | 0.9598 |
| submissions | raw_only+dog+edge+erode@medium-4000 | v4 | 199 | 0.7478 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| submissions | raw_only+dog+edge+erode@medium-4000 | v3 | 199 | 0.7427 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9950 |
| submissions | raw_only+dog+edge+erode@medium-4000 | v2 | 199 | 0.7412 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.9950 | 1.0000 | 0.9950 |
| submissions | raw_only+dog+edge+erode@medium-4000 | v1 | 199 | 0.7384 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9899 |
| submissions | raw_only+dog+edge+erode@large-6000 | v3 | 199 | 0.7327 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9950 |
| submissions | raw_only+dog+edge+erode@large-6000 | v2 | 199 | 0.7324 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9950 |
| submissions | raw_only+dog+edge+erode@large-6000 | v4 | 199 | 0.7321 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9899 |
| submissions | raw_only+dog+edge+erode@large-6000 | v1 | 199 | 0.7300 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9950 |
| submissions | raw_only+dog+edge+erode@large-6000 | v0 | 199 | 0.7297 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9849 |
| submissions | raw_only+dog+edge+erode@medium | v4 | 199 | 0.7241 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9899 |
| submissions | raw_only+dog+edge+erode@medium | v3 | 199 | 0.7231 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9899 |
| submissions | raw_only+dog+edge+erode@medium | v2 | 199 | 0.7225 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9899 |
| submissions | raw_only+dog+edge+erode@large | v4 | 199 | 0.7224 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| submissions | raw_only+dog+edge+erode@medium | v1 | 199 | 0.7202 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9849 |
| submissions | temporal_stack+erode@medium-4000 | v4 | 199 | 0.7197 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.9598 | 1.0000 | 0.9950 |
| submissions | raw_only+dog+edge+erode@large | v3 | 199 | 0.7194 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9950 |
| submissions | raw_only+dog+edge+erode@medium-4000 | v0 | 199 | 0.7188 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.9899 | 1.0000 | 0.9749 |
| submissions | temporal_stack+erode@medium-4000 | v3 | 199 | 0.7181 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.9648 | 1.0000 | 0.9899 |
| submissions | raw_only+dog+edge+erode@large | v2 | 199 | 0.7177 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9950 |
| submissions | temporal_stack+erode@medium-4000 | v2 | 199 | 0.7176 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.9598 | 1.0000 | 0.9899 |
| submissions | raw_only+dog+edge+erode@medium | v0 | 199 | 0.7173 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.9950 | 1.0000 | 0.9899 |
| submissions | temporal_stack+erode@medium-4000 | v1 | 199 | 0.7163 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.9749 | 1.0000 | 0.9899 |
| submissions | raw_only+dog+edge+erode@large | v1 | 199 | 0.7150 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9899 |
| submissions | temporal_stack+erode@medium-4000 | v0 | 199 | 0.7144 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.9648 | 1.0000 | 0.9698 |
| submissions | temporal_diff+erode@medium | v4 | 199 | 0.7114 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.9698 | 1.0000 | 0.9899 |
| submissions | temporal_diff+erode@medium | v3 | 199 | 0.7085 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.9698 | 1.0000 | 0.9899 |
| submissions | temporal_diff+erode@medium | v2 | 199 | 0.7075 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.9749 | 1.0000 | 0.9899 |
| submissions | temporal_stack+erode@large | v4 | 199 | 0.7053 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9899 |
| submissions | temporal_stack+erode@large | v3 | 199 | 0.7034 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.9950 | 1.0000 | 0.9849 |
| submissions | temporal_stack+erode@large | v2 | 199 | 0.7030 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.9899 | 1.0000 | 0.9849 |
| submissions | raw_only+dog+edge+erode@large | v0 | 199 | 0.7024 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.9899 | 1.0000 | 0.9899 |
| submissions | temporal_stack+erode@large | v1 | 199 | 0.7010 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.9849 | 1.0000 | 0.9598 |
| submissions | temporal_diff+erode@medium | v1 | 199 | 0.7004 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.9548 | 1.0000 | 0.9598 |
| submissions | temporal_diff+erode@medium | v0 | 199 | 0.6980 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.9598 | 1.0000 | 0.9548 |
| submissions | temporal_stack+erode@large | v0 | 199 | 0.6966 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.9648 | 1.0000 | 0.9598 |
| submissions | temporal_diff+dog@large | v3 | 199 | 0.6632 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.9899 | 1.0000 | 0.9950 |
| submissions | temporal_diff+dog@large | v4 | 199 | 0.6630 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.9899 | 1.0000 | 0.9950 |
| submissions | temporal_diff+dog@large | v2 | 199 | 0.6616 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.9950 | 1.0000 | 0.9899 |
| submissions | temporal_diff+dog@large | v1 | 199 | 0.6572 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.9849 | 1.0000 | 0.9648 |
| submissions | temporal_diff+dog@large | v0 | 199 | 0.6533 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.9799 | 1.0000 | 0.9598 |
| submissions | temporal_stack+erode@medium | v4 | 199 | 0.6353 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| submissions | temporal_stack+erode@medium | v3 | 199 | 0.6343 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| submissions | temporal_stack+erode@medium | v2 | 199 | 0.6327 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.9899 | 1.0000 | 1.0000 |
| submissions | temporal_stack+erode@medium | v1 | 199 | 0.6302 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.9899 | 1.0000 | 0.9899 |
| submissions | temporal_stack+erode@medium | v0 | 199 | 0.6267 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.9648 | 1.0000 | 0.9950 |
| submissions | temporal_diff+dog@medium | v2 | 199 | 0.5961 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.9849 | 1.0000 | 0.9849 |
| submissions | temporal_diff+dog@medium | v4 | 199 | 0.5960 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.9899 | 1.0000 | 1.0000 |
| submissions | temporal_diff+dog@medium | v3 | 199 | 0.5959 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.9899 | 1.0000 | 0.9849 |
| submissions | temporal_diff+dog@medium | v0 | 199 | 0.5934 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.9950 | 1.0000 | 0.9598 |
| submissions | temporal_diff+dog@medium | v1 | 199 | 0.5932 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.9950 | 1.0000 | 0.9698 |
| submissions | temporal_diff+dog@medium-4000 | v4 | 199 | 0.5807 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.9849 | 1.0000 | 0.9950 |
| submissions | temporal_diff+dog@medium-4000 | v3 | 199 | 0.5759 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.9849 | 1.0000 | 0.9899 |
| submissions | temporal_diff+dog@medium-4000 | v1 | 199 | 0.5742 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.9698 | 1.0000 | 0.9899 |
| submissions | temporal_diff+dog@medium-4000 | v2 | 199 | 0.5742 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.9799 | 1.0000 | 0.9899 |
| submissions | temporal_diff+dog@medium-4000 | v0 | 199 | 0.5716 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.9598 | 1.0000 | 0.9899 |
| submissions | temporal_diff+dog@large-6000 | v4 | 199 | 0.5102 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9899 |
| submissions | temporal_diff+dog@large-6000 | v3 | 199 | 0.5078 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.9899 | 1.0000 | 0.9799 |
| submissions | temporal_diff+dog@large-6000 | v2 | 199 | 0.5060 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.9849 | 1.0000 | 0.9849 |
| submissions | temporal_diff+dog@large-6000 | v1 | 199 | 0.5025 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.9849 | 1.0000 | 0.9849 |
| submissions | temporal_diff+dog@large-6000 | v0 | 199 | 0.4998 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.9749 | 1.0000 | 0.9899 |
| submissions | dog_detector | v4 | 199 | 0.3834 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.9548 | 1.0000 | 0.9598 |
| submissions | dog_detector | v3 | 199 | 0.3719 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.8442 | 1.0000 | 0.9598 |
| submissions | dog_detector | v2 | 199 | 0.3680 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.8392 | 1.0000 | 0.9548 |
| submissions | dog_detector | v1 | 199 | 0.3611 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.8392 | 1.0000 | 0.9397 |
| submissions | dog_detector | v0 | 199 | 0.3577 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.8241 | 1.0000 | 0.9296 |
| fiji_tracks | temporal_diff+dog@medium-4000 | fiji_lap | 199 | 0.2968 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.7990 | 1.0000 | 0.7990 |
| fiji_tracks | temporal_stack+erode@large-6000 | fiji_lap | 199 | 0.2868 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.7990 | 1.0000 | 0.8040 |
| fiji_tracks | temporal_diff+dog@medium | fiji_lap | 199 | 0.2823 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.7990 | 1.0000 | 0.8090 |
| fiji_tracks | temporal_diff+dog@large-6000 | fiji_lap | 199 | 0.2783 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.7990 | 1.0000 | 0.7990 |
| fiji_tracks | dog_detector | fiji_lap | 199 | 0.2772 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.7990 | 1.0000 | 0.7990 |
| fiji_tracks | temporal_diff+dog@large | fiji_lap | 199 | 0.2755 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.8090 | 1.0000 | 0.8090 |
| fiji_tracks | temporal_diff+erode@medium | fiji_lap | 199 | 0.2398 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.7990 | 1.0000 | 0.8040 |
| fiji_tracks | temporal_diff+erode@medium-4000 | fiji_lap | 199 | 0.2319 | NaN | NaN | 0.1500 | 0.0101 | 0.0151 | 1.0000 | 1.0000 | 0.7990 | 1.0000 | 0.8141 |
| fiji_tracks | temporal_stack+erode@medium | fiji_lap | 199 | 0.2304 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.7990 | 1.0000 | 0.8090 |
| fiji_tracks | temporal_stack+erode@large | fiji_lap | 199 | 0.2297 | NaN | NaN | 0.1500 | 0.0151 | 0.0201 | 1.0000 | 1.0000 | 0.8141 | 1.0000 | 0.8090 |
| submissions | log_detector | v2 | 199 | 0.2274 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.9849 | 1.0000 | 0.9849 |
| fiji_tracks | temporal_stack+erode@small | fiji_lap | 199 | 0.2260 | NaN | NaN | 0.1500 | 0.0151 | 0.0201 | 1.0000 | 1.0000 | 0.8141 | 1.0000 | 0.8191 |
| fiji_tracks | temporal_diff+dog@small | fiji_lap | 199 | 0.2164 | NaN | NaN | 0.1500 | 0.0151 | 0.0151 | 1.0000 | 1.0000 | 0.8090 | 1.0000 | 0.8191 |
| fiji_tracks | raw_only+dog+edge+erode@medium-4000 | fiji_lap | 199 | 0.2119 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.8040 | 1.0000 | 0.8090 |
| fiji_tracks | raw_only+dog+edge+erode@large | fiji_lap | 199 | 0.2088 | NaN | NaN | 0.1500 | 0.0151 | 0.0151 | 1.0000 | 1.0000 | 0.8040 | 1.0000 | 0.8191 |
| fiji_tracks | raw_only+dog+edge+erode@medium | fiji_lap | 199 | 0.2009 | NaN | NaN | 0.1500 | 0.0151 | 0.0251 | 1.0000 | 1.0000 | 0.8191 | 1.0000 | 0.8241 |
| fiji_tracks | raw_only+dog+edge+erode@small | fiji_lap | 199 | 0.1941 | NaN | NaN | 0.1500 | 0.0151 | 0.0251 | 1.0000 | 1.0000 | 0.8141 | 1.0000 | 0.8241 |
| fiji_tracks | temporal_diff+erode@small | fiji_lap | 199 | 0.1939 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.8040 | 1.0000 | 0.8191 |
| fiji_tracks | temporal_stack+erode@medium-4000 | fiji_lap | 199 | 0.1597 | NaN | NaN | 0.1500 | 0.0201 | 0.0302 | 1.0000 | 1.0000 | 0.8291 | 1.0000 | 0.8241 |
| fiji_tracks | raw_only+dog+edge+erode@large-6000 | fiji_lap | 199 | 0.1421 | NaN | NaN | 0.1500 | 0.0101 | 0.0251 | 1.0000 | 1.0000 | 0.8040 | 1.0000 | 0.8241 |
| fiji_tracks | log_detector | fiji_lap | 199 | 0.1392 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.7990 | 1.0000 | 0.7990 |
| submissions | log_detector | v1 | 199 | 0.1040 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.8241 | 1.0000 | 0.8090 |
| submissions | log_detector | v0 | 199 | 0.0989 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.8090 | 1.0000 | 0.8040 |

## 2. Fiji LAP Division Tracker Results (`/mnt/data/fiji_tracks/`)

| source_group | detector | tracker | n_videos | aggregate_score | edge_jaccard | division_jaccard | threshold_t | overall_assertion_pass_rate | a1_pass_rate | a2_pass_rate | a3_pass_rate | a4_pass_rate | a5_pass_rate | a6_pass_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fiji_tracks | temporal_diff+dog@medium-4000 | fiji_lap | 199 | 0.2968 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.7990 | 1.0000 | 0.7990 |
| fiji_tracks | temporal_stack+erode@large-6000 | fiji_lap | 199 | 0.2868 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.7990 | 1.0000 | 0.8040 |
| fiji_tracks | temporal_diff+dog@medium | fiji_lap | 199 | 0.2823 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.7990 | 1.0000 | 0.8090 |
| fiji_tracks | temporal_diff+dog@large-6000 | fiji_lap | 199 | 0.2783 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.7990 | 1.0000 | 0.7990 |
| fiji_tracks | dog_detector | fiji_lap | 199 | 0.2772 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.7990 | 1.0000 | 0.7990 |
| fiji_tracks | temporal_diff+dog@large | fiji_lap | 199 | 0.2755 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.8090 | 1.0000 | 0.8090 |
| fiji_tracks | temporal_diff+erode@medium | fiji_lap | 199 | 0.2398 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.7990 | 1.0000 | 0.8040 |
| fiji_tracks | temporal_diff+erode@medium-4000 | fiji_lap | 199 | 0.2319 | NaN | NaN | 0.1500 | 0.0101 | 0.0151 | 1.0000 | 1.0000 | 0.7990 | 1.0000 | 0.8141 |
| fiji_tracks | temporal_stack+erode@medium | fiji_lap | 199 | 0.2304 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.7990 | 1.0000 | 0.8090 |
| fiji_tracks | temporal_stack+erode@large | fiji_lap | 199 | 0.2297 | NaN | NaN | 0.1500 | 0.0151 | 0.0201 | 1.0000 | 1.0000 | 0.8141 | 1.0000 | 0.8090 |
| fiji_tracks | temporal_stack+erode@small | fiji_lap | 199 | 0.2260 | NaN | NaN | 0.1500 | 0.0151 | 0.0201 | 1.0000 | 1.0000 | 0.8141 | 1.0000 | 0.8191 |
| fiji_tracks | temporal_diff+dog@small | fiji_lap | 199 | 0.2164 | NaN | NaN | 0.1500 | 0.0151 | 0.0151 | 1.0000 | 1.0000 | 0.8090 | 1.0000 | 0.8191 |
| fiji_tracks | raw_only+dog+edge+erode@medium-4000 | fiji_lap | 199 | 0.2119 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.8040 | 1.0000 | 0.8090 |
| fiji_tracks | raw_only+dog+edge+erode@large | fiji_lap | 199 | 0.2088 | NaN | NaN | 0.1500 | 0.0151 | 0.0151 | 1.0000 | 1.0000 | 0.8040 | 1.0000 | 0.8191 |
| fiji_tracks | raw_only+dog+edge+erode@medium | fiji_lap | 199 | 0.2009 | NaN | NaN | 0.1500 | 0.0151 | 0.0251 | 1.0000 | 1.0000 | 0.8191 | 1.0000 | 0.8241 |
| fiji_tracks | raw_only+dog+edge+erode@small | fiji_lap | 199 | 0.1941 | NaN | NaN | 0.1500 | 0.0151 | 0.0251 | 1.0000 | 1.0000 | 0.8141 | 1.0000 | 0.8241 |
| fiji_tracks | temporal_diff+erode@small | fiji_lap | 199 | 0.1939 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.8040 | 1.0000 | 0.8191 |
| fiji_tracks | temporal_stack+erode@medium-4000 | fiji_lap | 199 | 0.1597 | NaN | NaN | 0.1500 | 0.0201 | 0.0302 | 1.0000 | 1.0000 | 0.8291 | 1.0000 | 0.8241 |
| fiji_tracks | raw_only+dog+edge+erode@large-6000 | fiji_lap | 199 | 0.1421 | NaN | NaN | 0.1500 | 0.0101 | 0.0251 | 1.0000 | 1.0000 | 0.8040 | 1.0000 | 0.8241 |
| fiji_tracks | log_detector | fiji_lap | 199 | 0.1392 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.7990 | 1.0000 | 0.7990 |

## 3. Submissions Results (`/mnt/data/submissions/{detector}/{tracker}/submission.csv`)

| source_group | detector | tracker | n_videos | aggregate_score | edge_jaccard | division_jaccard | threshold_t | overall_assertion_pass_rate | a1_pass_rate | a2_pass_rate | a3_pass_rate | a4_pass_rate | a5_pass_rate | a6_pass_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| submissions | temporal_diff+erode@small | v4 | 199 | 0.8529 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| submissions | temporal_diff+erode@small | v3 | 199 | 0.8499 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9849 |
| submissions | temporal_diff+dog@small | v2 | 199 | 0.8495 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9950 |
| submissions | temporal_diff+erode@small | v2 | 199 | 0.8488 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9849 |
| submissions | temporal_diff+erode@small | v0 | 199 | 0.8485 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9698 |
| submissions | temporal_diff+dog@small | v4 | 199 | 0.8464 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| submissions | temporal_diff+erode@small | v1 | 199 | 0.8460 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9749 |
| submissions | temporal_diff+dog@small | v0 | 199 | 0.8459 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.9950 | 1.0000 | 0.9849 |
| submissions | temporal_diff+dog@small | v3 | 199 | 0.8458 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9950 |
| submissions | temporal_diff+dog@small | v1 | 199 | 0.8430 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.9950 | 1.0000 | 0.9899 |
| submissions | raw_only+dog+edge+erode@small | v4 | 199 | 0.8284 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9899 |
| submissions | raw_only+dog+edge+erode@small | v2 | 199 | 0.8267 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9799 |
| submissions | raw_only+dog+edge+erode@small | v3 | 199 | 0.8266 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9849 |
| submissions | raw_only+dog+edge+erode@small | v1 | 199 | 0.8238 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.9950 | 1.0000 | 0.9749 |
| submissions | raw_only+dog+edge+erode@small | v0 | 199 | 0.8237 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.9849 | 1.0000 | 0.9749 |
| submissions | temporal_stack+erode@small | v4 | 199 | 0.8229 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| submissions | temporal_stack+erode@small | v3 | 199 | 0.8225 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9849 |
| submissions | temporal_stack+erode@small | v2 | 199 | 0.8212 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.9950 | 1.0000 | 0.9849 |
| submissions | temporal_stack+erode@small | v1 | 199 | 0.8182 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.9950 | 1.0000 | 0.9849 |
| submissions | temporal_stack+erode@small | v0 | 199 | 0.8175 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.9950 | 1.0000 | 0.9849 |
| submissions | temporal_diff+erode@medium-4000 | v3 | 199 | 0.7941 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.9950 | 1.0000 | 0.9899 |
| submissions | temporal_diff+erode@medium-4000 | v4 | 199 | 0.7932 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.9950 | 1.0000 | 0.9849 |
| submissions | temporal_diff+erode@medium-4000 | v2 | 199 | 0.7929 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.9950 | 1.0000 | 0.9799 |
| submissions | temporal_stack+erode@large-6000 | v4 | 199 | 0.7905 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9950 |
| submissions | temporal_diff+erode@medium-4000 | v0 | 199 | 0.7902 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.9950 | 1.0000 | 0.9648 |
| submissions | temporal_diff+erode@medium-4000 | v1 | 199 | 0.7890 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.9950 | 1.0000 | 0.9749 |
| submissions | temporal_stack+erode@large-6000 | v3 | 199 | 0.7862 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.9950 | 1.0000 | 0.9950 |
| submissions | temporal_stack+erode@large-6000 | v2 | 199 | 0.7818 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.9950 | 1.0000 | 0.9950 |
| submissions | temporal_stack+erode@large-6000 | v1 | 199 | 0.7783 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.9950 | 1.0000 | 0.9648 |
| submissions | temporal_stack+erode@large-6000 | v0 | 199 | 0.7716 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.9749 | 1.0000 | 0.9598 |
| submissions | raw_only+dog+edge+erode@medium-4000 | v4 | 199 | 0.7478 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| submissions | raw_only+dog+edge+erode@medium-4000 | v3 | 199 | 0.7427 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9950 |
| submissions | raw_only+dog+edge+erode@medium-4000 | v2 | 199 | 0.7412 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.9950 | 1.0000 | 0.9950 |
| submissions | raw_only+dog+edge+erode@medium-4000 | v1 | 199 | 0.7384 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9899 |
| submissions | raw_only+dog+edge+erode@large-6000 | v3 | 199 | 0.7327 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9950 |
| submissions | raw_only+dog+edge+erode@large-6000 | v2 | 199 | 0.7324 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9950 |
| submissions | raw_only+dog+edge+erode@large-6000 | v4 | 199 | 0.7321 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9899 |
| submissions | raw_only+dog+edge+erode@large-6000 | v1 | 199 | 0.7300 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9950 |
| submissions | raw_only+dog+edge+erode@large-6000 | v0 | 199 | 0.7297 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9849 |
| submissions | raw_only+dog+edge+erode@medium | v4 | 199 | 0.7241 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9899 |
| submissions | raw_only+dog+edge+erode@medium | v3 | 199 | 0.7231 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9899 |
| submissions | raw_only+dog+edge+erode@medium | v2 | 199 | 0.7225 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9899 |
| submissions | raw_only+dog+edge+erode@large | v4 | 199 | 0.7224 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| submissions | raw_only+dog+edge+erode@medium | v1 | 199 | 0.7202 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9849 |
| submissions | temporal_stack+erode@medium-4000 | v4 | 199 | 0.7197 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.9598 | 1.0000 | 0.9950 |
| submissions | raw_only+dog+edge+erode@large | v3 | 199 | 0.7194 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9950 |
| submissions | raw_only+dog+edge+erode@medium-4000 | v0 | 199 | 0.7188 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.9899 | 1.0000 | 0.9749 |
| submissions | temporal_stack+erode@medium-4000 | v3 | 199 | 0.7181 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.9648 | 1.0000 | 0.9899 |
| submissions | raw_only+dog+edge+erode@large | v2 | 199 | 0.7177 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9950 |
| submissions | temporal_stack+erode@medium-4000 | v2 | 199 | 0.7176 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.9598 | 1.0000 | 0.9899 |
| submissions | raw_only+dog+edge+erode@medium | v0 | 199 | 0.7173 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.9950 | 1.0000 | 0.9899 |
| submissions | temporal_stack+erode@medium-4000 | v1 | 199 | 0.7163 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.9749 | 1.0000 | 0.9899 |
| submissions | raw_only+dog+edge+erode@large | v1 | 199 | 0.7150 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9899 |
| submissions | temporal_stack+erode@medium-4000 | v0 | 199 | 0.7144 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.9648 | 1.0000 | 0.9698 |
| submissions | temporal_diff+erode@medium | v4 | 199 | 0.7114 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.9698 | 1.0000 | 0.9899 |
| submissions | temporal_diff+erode@medium | v3 | 199 | 0.7085 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.9698 | 1.0000 | 0.9899 |
| submissions | temporal_diff+erode@medium | v2 | 199 | 0.7075 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.9749 | 1.0000 | 0.9899 |
| submissions | temporal_stack+erode@large | v4 | 199 | 0.7053 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9899 |
| submissions | temporal_stack+erode@large | v3 | 199 | 0.7034 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.9950 | 1.0000 | 0.9849 |
| submissions | temporal_stack+erode@large | v2 | 199 | 0.7030 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.9899 | 1.0000 | 0.9849 |
| submissions | raw_only+dog+edge+erode@large | v0 | 199 | 0.7024 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.9899 | 1.0000 | 0.9899 |
| submissions | temporal_stack+erode@large | v1 | 199 | 0.7010 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.9849 | 1.0000 | 0.9598 |
| submissions | temporal_diff+erode@medium | v1 | 199 | 0.7004 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.9548 | 1.0000 | 0.9598 |
| submissions | temporal_diff+erode@medium | v0 | 199 | 0.6980 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.9598 | 1.0000 | 0.9548 |
| submissions | temporal_stack+erode@large | v0 | 199 | 0.6966 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.9648 | 1.0000 | 0.9598 |
| submissions | temporal_diff+dog@large | v3 | 199 | 0.6632 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.9899 | 1.0000 | 0.9950 |
| submissions | temporal_diff+dog@large | v4 | 199 | 0.6630 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.9899 | 1.0000 | 0.9950 |
| submissions | temporal_diff+dog@large | v2 | 199 | 0.6616 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.9950 | 1.0000 | 0.9899 |
| submissions | temporal_diff+dog@large | v1 | 199 | 0.6572 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.9849 | 1.0000 | 0.9648 |
| submissions | temporal_diff+dog@large | v0 | 199 | 0.6533 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.9799 | 1.0000 | 0.9598 |
| submissions | temporal_stack+erode@medium | v4 | 199 | 0.6353 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| submissions | temporal_stack+erode@medium | v3 | 199 | 0.6343 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| submissions | temporal_stack+erode@medium | v2 | 199 | 0.6327 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.9899 | 1.0000 | 1.0000 |
| submissions | temporal_stack+erode@medium | v1 | 199 | 0.6302 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.9899 | 1.0000 | 0.9899 |
| submissions | temporal_stack+erode@medium | v0 | 199 | 0.6267 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.9648 | 1.0000 | 0.9950 |
| submissions | temporal_diff+dog@medium | v2 | 199 | 0.5961 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.9849 | 1.0000 | 0.9849 |
| submissions | temporal_diff+dog@medium | v4 | 199 | 0.5960 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.9899 | 1.0000 | 1.0000 |
| submissions | temporal_diff+dog@medium | v3 | 199 | 0.5959 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.9899 | 1.0000 | 0.9849 |
| submissions | temporal_diff+dog@medium | v0 | 199 | 0.5934 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.9950 | 1.0000 | 0.9598 |
| submissions | temporal_diff+dog@medium | v1 | 199 | 0.5932 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.9950 | 1.0000 | 0.9698 |
| submissions | temporal_diff+dog@medium-4000 | v4 | 199 | 0.5807 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.9849 | 1.0000 | 0.9950 |
| submissions | temporal_diff+dog@medium-4000 | v3 | 199 | 0.5759 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.9849 | 1.0000 | 0.9899 |
| submissions | temporal_diff+dog@medium-4000 | v1 | 199 | 0.5742 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.9698 | 1.0000 | 0.9899 |
| submissions | temporal_diff+dog@medium-4000 | v2 | 199 | 0.5742 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.9799 | 1.0000 | 0.9899 |
| submissions | temporal_diff+dog@medium-4000 | v0 | 199 | 0.5716 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.9598 | 1.0000 | 0.9899 |
| submissions | temporal_diff+dog@large-6000 | v4 | 199 | 0.5102 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9899 |
| submissions | temporal_diff+dog@large-6000 | v3 | 199 | 0.5078 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.9899 | 1.0000 | 0.9799 |
| submissions | temporal_diff+dog@large-6000 | v2 | 199 | 0.5060 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.9849 | 1.0000 | 0.9849 |
| submissions | temporal_diff+dog@large-6000 | v1 | 199 | 0.5025 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.9849 | 1.0000 | 0.9849 |
| submissions | temporal_diff+dog@large-6000 | v0 | 199 | 0.4998 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.9749 | 1.0000 | 0.9899 |
| submissions | dog_detector | v4 | 199 | 0.3834 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.9548 | 1.0000 | 0.9598 |
| submissions | dog_detector | v3 | 199 | 0.3719 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.8442 | 1.0000 | 0.9598 |
| submissions | dog_detector | v2 | 199 | 0.3680 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.8392 | 1.0000 | 0.9548 |
| submissions | dog_detector | v1 | 199 | 0.3611 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.8392 | 1.0000 | 0.9397 |
| submissions | dog_detector | v0 | 199 | 0.3577 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.8241 | 1.0000 | 0.9296 |
| submissions | log_detector | v2 | 199 | 0.2274 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.9849 | 1.0000 | 0.9849 |
| submissions | log_detector | v1 | 199 | 0.1040 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.8241 | 1.0000 | 0.8090 |
| submissions | log_detector | v0 | 199 | 0.0989 | NaN | NaN | 0.1500 | 0.0101 | 0.0101 | 1.0000 | 1.0000 | 0.8090 | 1.0000 | 0.8040 |

## 4. Total Structural Violation Counts Across Test Videos

| source_group | detector | tracker | total_a2_merges | total_a3_dup_ids | total_a4_div_rules | total_a5_temporal | total_a6_dist_gt_14um |
| --- | --- | --- | --- | --- | --- | --- | --- |
| submissions | temporal_diff+erode@small | v4 | 0 | 0 | 0 | 0 | 0 |
| submissions | temporal_diff+erode@small | v3 | 0 | 0 | 0 | 0 | 3 |
| submissions | temporal_diff+dog@small | v2 | 0 | 0 | 0 | 0 | 1 |
| submissions | temporal_diff+erode@small | v2 | 0 | 0 | 0 | 0 | 3 |
| submissions | temporal_diff+erode@small | v0 | 0 | 0 | 0 | 0 | 9 |
| submissions | temporal_diff+dog@small | v4 | 0 | 0 | 0 | 0 | 0 |
| submissions | temporal_diff+erode@small | v1 | 0 | 0 | 0 | 0 | 8 |
| submissions | temporal_diff+dog@small | v0 | 0 | 0 | 1 | 0 | 3 |
| submissions | temporal_diff+dog@small | v3 | 0 | 0 | 0 | 0 | 1 |
| submissions | temporal_diff+dog@small | v1 | 0 | 0 | 1 | 0 | 2 |
| submissions | raw_only+dog+edge+erode@small | v4 | 0 | 0 | 0 | 0 | 2 |
| submissions | raw_only+dog+edge+erode@small | v2 | 0 | 0 | 0 | 0 | 4 |
| submissions | raw_only+dog+edge+erode@small | v3 | 0 | 0 | 0 | 0 | 3 |
| submissions | raw_only+dog+edge+erode@small | v1 | 0 | 0 | 1 | 0 | 5 |
| submissions | raw_only+dog+edge+erode@small | v0 | 0 | 0 | 3 | 0 | 5 |
| submissions | temporal_stack+erode@small | v4 | 0 | 0 | 0 | 0 | 0 |
| submissions | temporal_stack+erode@small | v3 | 0 | 0 | 0 | 0 | 3 |
| submissions | temporal_stack+erode@small | v2 | 0 | 0 | 1 | 0 | 3 |
| submissions | temporal_stack+erode@small | v1 | 0 | 0 | 1 | 0 | 3 |
| submissions | temporal_stack+erode@small | v0 | 0 | 0 | 1 | 0 | 3 |
| submissions | temporal_diff+erode@medium-4000 | v3 | 0 | 0 | 1 | 0 | 2 |
| submissions | temporal_diff+erode@medium-4000 | v4 | 0 | 0 | 1 | 0 | 3 |
| submissions | temporal_diff+erode@medium-4000 | v2 | 0 | 0 | 1 | 0 | 4 |
| submissions | temporal_stack+erode@large-6000 | v4 | 0 | 0 | 0 | 0 | 1 |
| submissions | temporal_diff+erode@medium-4000 | v0 | 0 | 0 | 1 | 0 | 13 |
| submissions | temporal_diff+erode@medium-4000 | v1 | 0 | 0 | 2 | 0 | 12 |
| submissions | temporal_stack+erode@large-6000 | v3 | 0 | 0 | 1 | 0 | 1 |
| submissions | temporal_stack+erode@large-6000 | v2 | 0 | 0 | 2 | 0 | 1 |
| submissions | temporal_stack+erode@large-6000 | v1 | 0 | 0 | 1 | 0 | 11 |
| submissions | temporal_stack+erode@large-6000 | v0 | 0 | 0 | 6 | 0 | 13 |
| submissions | raw_only+dog+edge+erode@medium-4000 | v4 | 0 | 0 | 0 | 0 | 0 |
| submissions | raw_only+dog+edge+erode@medium-4000 | v3 | 0 | 0 | 0 | 0 | 1 |
| submissions | raw_only+dog+edge+erode@medium-4000 | v2 | 0 | 0 | 1 | 0 | 1 |
| submissions | raw_only+dog+edge+erode@medium-4000 | v1 | 0 | 0 | 0 | 0 | 2 |
| submissions | raw_only+dog+edge+erode@large-6000 | v3 | 0 | 0 | 0 | 0 | 1 |
| submissions | raw_only+dog+edge+erode@large-6000 | v2 | 0 | 0 | 0 | 0 | 1 |
| submissions | raw_only+dog+edge+erode@large-6000 | v4 | 0 | 0 | 0 | 0 | 2 |
| submissions | raw_only+dog+edge+erode@large-6000 | v1 | 0 | 0 | 0 | 0 | 1 |
| submissions | raw_only+dog+edge+erode@large-6000 | v0 | 0 | 0 | 0 | 0 | 3 |
| submissions | raw_only+dog+edge+erode@medium | v4 | 0 | 0 | 0 | 0 | 2 |
| submissions | raw_only+dog+edge+erode@medium | v3 | 0 | 0 | 0 | 0 | 3 |
| submissions | raw_only+dog+edge+erode@medium | v2 | 0 | 0 | 0 | 0 | 3 |
| submissions | raw_only+dog+edge+erode@large | v4 | 0 | 0 | 0 | 0 | 0 |
| submissions | raw_only+dog+edge+erode@medium | v1 | 0 | 0 | 0 | 0 | 4 |
| submissions | temporal_stack+erode@medium-4000 | v4 | 0 | 0 | 11 | 0 | 1 |
| submissions | raw_only+dog+edge+erode@large | v3 | 0 | 0 | 0 | 0 | 2 |
| submissions | raw_only+dog+edge+erode@medium-4000 | v0 | 0 | 0 | 3 | 0 | 5 |
| submissions | temporal_stack+erode@medium-4000 | v3 | 0 | 0 | 9 | 0 | 2 |
| submissions | raw_only+dog+edge+erode@large | v2 | 0 | 0 | 0 | 0 | 2 |
| submissions | temporal_stack+erode@medium-4000 | v2 | 0 | 0 | 14 | 0 | 2 |
| submissions | raw_only+dog+edge+erode@medium | v0 | 0 | 0 | 1 | 0 | 3 |
| submissions | temporal_stack+erode@medium-4000 | v1 | 0 | 0 | 9 | 0 | 2 |
| submissions | raw_only+dog+edge+erode@large | v1 | 0 | 0 | 0 | 0 | 2 |
| submissions | temporal_stack+erode@medium-4000 | v0 | 0 | 0 | 21 | 0 | 6 |
| submissions | temporal_diff+erode@medium | v4 | 0 | 0 | 6 | 0 | 3 |
| submissions | temporal_diff+erode@medium | v3 | 0 | 0 | 6 | 0 | 2 |
| submissions | temporal_diff+erode@medium | v2 | 0 | 0 | 7 | 0 | 2 |
| submissions | temporal_stack+erode@large | v4 | 0 | 0 | 0 | 0 | 2 |
| submissions | temporal_stack+erode@large | v3 | 0 | 0 | 1 | 0 | 3 |
| submissions | temporal_stack+erode@large | v2 | 0 | 0 | 2 | 0 | 3 |
| submissions | raw_only+dog+edge+erode@large | v0 | 0 | 0 | 2 | 0 | 2 |
| submissions | temporal_stack+erode@large | v1 | 0 | 0 | 3 | 0 | 12 |
| submissions | temporal_diff+erode@medium | v1 | 0 | 0 | 9 | 0 | 12 |
| submissions | temporal_diff+erode@medium | v0 | 0 | 0 | 8 | 0 | 19 |
| submissions | temporal_stack+erode@large | v0 | 0 | 0 | 9 | 0 | 14 |
| submissions | temporal_diff+dog@large | v3 | 0 | 0 | 2 | 0 | 1 |
| submissions | temporal_diff+dog@large | v4 | 0 | 0 | 2 | 0 | 1 |
| submissions | temporal_diff+dog@large | v2 | 0 | 0 | 2 | 0 | 2 |
| submissions | temporal_diff+dog@large | v1 | 0 | 0 | 3 | 0 | 16 |
| submissions | temporal_diff+dog@large | v0 | 0 | 0 | 4 | 0 | 19 |
| submissions | temporal_stack+erode@medium | v4 | 0 | 0 | 0 | 0 | 0 |
| submissions | temporal_stack+erode@medium | v3 | 0 | 0 | 0 | 0 | 0 |
| submissions | temporal_stack+erode@medium | v2 | 0 | 0 | 2 | 0 | 0 |
| submissions | temporal_stack+erode@medium | v1 | 0 | 0 | 5 | 0 | 2 |
| submissions | temporal_stack+erode@medium | v0 | 0 | 0 | 10 | 0 | 1 |
| submissions | temporal_diff+dog@medium | v2 | 0 | 0 | 3 | 0 | 5 |
| submissions | temporal_diff+dog@medium | v4 | 0 | 0 | 2 | 0 | 0 |
| submissions | temporal_diff+dog@medium | v3 | 0 | 0 | 2 | 0 | 4 |
| submissions | temporal_diff+dog@medium | v0 | 0 | 0 | 1 | 0 | 19 |
| submissions | temporal_diff+dog@medium | v1 | 0 | 0 | 1 | 0 | 14 |
| submissions | temporal_diff+dog@medium-4000 | v4 | 0 | 0 | 10 | 0 | 1 |
| submissions | temporal_diff+dog@medium-4000 | v3 | 0 | 0 | 8 | 0 | 2 |
| submissions | temporal_diff+dog@medium-4000 | v1 | 0 | 0 | 18 | 0 | 2 |
| submissions | temporal_diff+dog@medium-4000 | v2 | 0 | 0 | 11 | 0 | 2 |
| submissions | temporal_diff+dog@medium-4000 | v0 | 0 | 0 | 26 | 0 | 2 |
| submissions | temporal_diff+dog@large-6000 | v4 | 0 | 0 | 0 | 0 | 2 |
| submissions | temporal_diff+dog@large-6000 | v3 | 0 | 0 | 2 | 0 | 5 |
| submissions | temporal_diff+dog@large-6000 | v2 | 0 | 0 | 3 | 0 | 4 |
| submissions | temporal_diff+dog@large-6000 | v1 | 0 | 0 | 3 | 0 | 3 |
| submissions | temporal_diff+dog@large-6000 | v0 | 0 | 0 | 5 | 0 | 2 |
| submissions | dog_detector | v4 | 0 | 0 | 10 | 0 | 13 |
| submissions | dog_detector | v3 | 0 | 0 | 76 | 0 | 13 |
| submissions | dog_detector | v2 | 0 | 0 | 83 | 0 | 13 |
| submissions | dog_detector | v1 | 0 | 0 | 112 | 0 | 34 |
| submissions | dog_detector | v0 | 0 | 0 | 144 | 0 | 48 |
| fiji_tracks | temporal_diff+dog@medium-4000 | fiji_lap | 0 | 0 | 638 | 0 | 428 |
| fiji_tracks | temporal_stack+erode@large-6000 | fiji_lap | 0 | 0 | 594 | 0 | 363 |
| fiji_tracks | temporal_diff+dog@medium | fiji_lap | 0 | 0 | 419 | 0 | 505 |
| fiji_tracks | temporal_diff+dog@large-6000 | fiji_lap | 0 | 0 | 532 | 0 | 558 |
| fiji_tracks | dog_detector | fiji_lap | 0 | 0 | 763 | 0 | 542 |
| fiji_tracks | temporal_diff+dog@large | fiji_lap | 0 | 0 | 416 | 0 | 404 |
| fiji_tracks | temporal_diff+erode@medium | fiji_lap | 0 | 0 | 412 | 0 | 291 |
| fiji_tracks | temporal_diff+erode@medium-4000 | fiji_lap | 0 | 0 | 449 | 0 | 280 |
| fiji_tracks | temporal_stack+erode@medium | fiji_lap | 0 | 0 | 447 | 0 | 475 |
| fiji_tracks | temporal_stack+erode@large | fiji_lap | 0 | 0 | 434 | 0 | 443 |
| submissions | log_detector | v2 | 0 | 0 | 9 | 0 | 27 |
| fiji_tracks | temporal_stack+erode@small | fiji_lap | 0 | 0 | 462 | 0 | 349 |
| fiji_tracks | temporal_diff+dog@small | fiji_lap | 0 | 0 | 411 | 0 | 253 |
| fiji_tracks | raw_only+dog+edge+erode@medium-4000 | fiji_lap | 0 | 0 | 469 | 0 | 538 |
| fiji_tracks | raw_only+dog+edge+erode@large | fiji_lap | 0 | 0 | 455 | 0 | 508 |
| fiji_tracks | raw_only+dog+edge+erode@medium | fiji_lap | 0 | 0 | 391 | 0 | 411 |
| fiji_tracks | raw_only+dog+edge+erode@small | fiji_lap | 0 | 0 | 483 | 0 | 418 |
| fiji_tracks | temporal_diff+erode@small | fiji_lap | 0 | 0 | 457 | 0 | 291 |
| fiji_tracks | temporal_stack+erode@medium-4000 | fiji_lap | 0 | 0 | 412 | 0 | 375 |
| fiji_tracks | raw_only+dog+edge+erode@large-6000 | fiji_lap | 0 | 0 | 385 | 0 | 517 |
| fiji_tracks | log_detector | fiji_lap | 0 | 0 | 1898 | 0 | 1984 |
| submissions | log_detector | v1 | 0 | 0 | 139 | 0 | 298 |
| submissions | log_detector | v0 | 0 | 0 | 168 | 0 | 456 |

