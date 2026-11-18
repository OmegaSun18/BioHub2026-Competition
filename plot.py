from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ==============================================================================
# 1. Paths & Visual Styling (Matched to plot.py)
# ==============================================================================
DATA_DIR = Path("/workspace")
PER_VIDEO_CSV = DATA_DIR / "evaluation_per_video_metrics.csv"

# Support both spellings of the summary table filename
SUMMARY_CSV = DATA_DIR / "evaluation_summary_table.csv"
if not SUMMARY_CSV.exists() and (DATA_DIR / "evlauation_summary_table.csv").exists():
    SUMMARY_CSV = DATA_DIR / "evlauation_summary_table.csv"

CORRECTED_SUMMARY_CSV = DATA_DIR / "evaluation_summary_table_corrected.csv"

INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

# Tracker palette: v0-v4 from plot.py + distinct warm accent for fiji_lap
VC_ALL = {
    "fiji_lap": "#d95f02",
    "v0": "#a7c6ee",
    "v1": "#6ea3e6",
    "v2": "#2a78d6",
    "v3": "#1c5aa8",
    "v4": "#0f3a73",
}
VL_ALL = {
    "fiji_lap": "Fiji LAP",
    "v0": "v0 Kalman only",
    "v1": "v1 +raw",
    "v2": "v2 +dog",
    "v3": "v3 +dog+raw",
    "v4": "v4 +dog+raw+erode",
}
SUB_VARIANTS = ["v0", "v1", "v2", "v3", "v4"]

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "text.color": INK,
        "axes.labelcolor": INK2,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "figure.facecolor": SURF,
        "axes.facecolor": SURF,
    }
)


def short_label(name: str) -> str:
    return name.replace("raw_only+dog+edge+erode", "raw_only+…")


# ==============================================================================
# 2. Load Data & Recompute Corrected Assertion Pass Rates on the 40 Test Videos
# ==============================================================================
summary_df = pd.read_csv(SUMMARY_CSV)
per_vid_df = pd.read_csv(PER_VIDEO_CSV)

# Identify the 40 actual test videos (union of datasets with predicted nodes > 0)
test_40_datasets = sorted(per_vid_df.loc[per_vid_df["n_pred_nodes"] > 0, "dataset"].unique())
pv40 = per_vid_df[per_vid_df["dataset"].isin(test_40_datasets)].copy()

# Corrected per-video metrics:
# 1) Structural all-pass (A2 & A3 & A4 & A5 & A6) — excludes un-tuned A1 (t=0.15)
pv40["structural_all_pass"] = (
    pv40["a2_pass"] & pv40["a3_pass"] & pv40["a4_pass"] & pv40["a5_pass"] & pv40["a6_pass"]
)
# 2) Average pass rate across all 6 assertions per video
pv40["mean_all6_pass"] = pv40[["a1_pass", "a2_pass", "a3_pass", "a4_pass", "a5_pass", "a6_pass"]].mean(axis=1)
# 3) Average pass rate across the 5 structural assertions (A2-A6)
pv40["mean_struct5_pass"] = pv40[["a2_pass", "a3_pass", "a4_pass", "a5_pass", "a6_pass"]].mean(axis=1)

# Note: log_detector v2 only has predictions on 3/40 test videos; score empty videos as False
def agg_40(grp: pd.DataFrame) -> pd.Series:
    n_active = int((grp["n_pred_nodes"] > 0).sum())
    denom = float(len(test_40_datasets))  # Always normalize by all 40 test videos
    return pd.Series(
        {
            "n_test_videos": len(test_40_datasets),
            "n_active_videos": n_active,
            "structural_all_pass_rate": grp["structural_all_pass"].sum() / denom,
            "mean_all6_pass_rate": grp["mean_all6_pass"].sum() / denom,
            "mean_struct5_pass_rate": grp["mean_struct5_pass"].sum() / denom,
            "a1_pass_40": grp["a1_pass"].sum() / denom,
            "a2_pass_40": grp["a2_pass"].sum() / denom,
            "a3_pass_40": grp["a3_pass"].sum() / denom,
            "a4_pass_40": grp["a4_pass"].sum() / denom,
            "a5_pass_40": grp["a5_pass"].sum() / denom,
            "a6_pass_40": grp["a6_pass"].sum() / denom,
        }
    )


corrected_assertions = (
    pv40.groupby(["source_group", "detector", "tracker"], as_index=False)
    .apply(agg_40, include_groups=False)
)

df = summary_df[
    ["source_group", "detector", "tracker", "aggregate_score", "threshold_t", "overall_assertion_pass_rate"]
].merge(corrected_assertions, on=["source_group", "detector", "tracker"], how="inner")

df.to_csv(CORRECTED_SUMMARY_CSV, index=False)
print(f"Saved corrected 40-video summary table -> {CORRECTED_SUMMARY_CSV} ({len(df)} rows)")


# ==============================================================================
# 3. Helper for Figure 1 & Figure 3 Style (Horizontal Dumbbell + Scatter Plot)
# ==============================================================================
def plot_detector_by_variant(
    df_in: pd.DataFrame,
    value_col: str,
    xlabel: str,
    title: str,
    callout_header: str,
    callout_xy: tuple[float, float],
    xlim: tuple[float, float],
    out_path: Path,
) -> None:
    P = df_in.pivot(index="detector", columns="tracker", values=value_col)
    order = P.max(axis=1).sort_values().index.tolist()  # Best detector at the top
    top5 = df_in.sort_values(value_col, ascending=False).head(5)

    fig, ax = plt.subplots(figsize=(12.5, 8.4), dpi=150)

    for i, m in enumerate(order):
        r = P.loc[m].dropna()
        if len(r) > 0:
            ax.plot([r.min(), r.max()], [i, i], color=GRID, lw=2.5, zorder=1, solid_capstyle="round")

    for v in VC_ALL:
        if v not in P.columns:
            continue
        xs = [P.loc[m, v] for m in order]
        ys = list(range(len(order)))
        marker = "D" if v == "fiji_lap" else "o"
        size = 64 if v == "fiji_lap" else 72
        ax.scatter(
            xs,
            ys,
            s=size,
            marker=marker,
            color=VC_ALL[v],
            edgecolor=SURF,
            linewidth=1.5,
            zorder=3,
            label=VL_ALL[v],
        )

    txt = callout_header + "\n" + "\n".join(
        f"{n}. {r.detector} ({r.tracker})   {r[value_col]:.4f}"
        for n, (_, r) in enumerate(top5.iterrows(), 1)
    )
    ax.text(
        callout_xy[0],
        callout_xy[1],
        txt,
        fontsize=9.5,
        va="top",
        ha="left",
        color=INK,
        linespacing=1.6,
        bbox=dict(boxstyle="round,pad=0.7", fc=SURF, ec=GRID),
    )

    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=9.5)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_xlim(*xlim)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)

    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="y", length=0)

    ax.legend(title="tracker variant", loc="lower right", frameon=False, fontsize=9.5, title_fontsize=10)
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold", pad=14)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved -> {out_path}")


# ==============================================================================
# 4. Figure 1 & Figure 2: Ground-Truth Score Across All Detectors & Trackers
# ==============================================================================
plot_detector_by_variant(
    df_in=df,
    value_col="aggregate_score",
    xlabel="official competition score on test (40 videos)",
    title="Every detector and tracker variant by test score, best detector at the top",
    callout_header="Top 5 detector + tracker pairs (test score)",
    callout_xy=(0.10, len(df["detector"].unique()) - 1.1),
    xlim=(0.08, 0.89),
    out_path=DATA_DIR / "fig1_score_by_variant.png",
)

# Figure 2: Variant effect on score (v0..v4 across complete 19 detectors, matching plot.py)
sub_df = df[df["source_group"] == "submissions"].copy()
P_score = sub_df.pivot(index="detector", columns="tracker", values="aggregate_score")[SUB_VARIANTS]
P_score_complete = P_score.dropna()
G_score = P_score_complete.sub(P_score_complete["v0"], axis=0)

fig, (a, b) = plt.subplots(1, 2, figsize=(12.5, 5.5), dpi=150, gridspec_kw={"width_ratios": [1.5, 1]})
x = range(len(SUB_VARIANTS))
for m in P_score_complete.index:
    a.plot(x, G_score.loc[m].values, color="#c3c2b7", lw=1.1, zorder=1)
a.plot(
    x,
    G_score.mean().values,
    color=VC_ALL["v4"],
    lw=3,
    marker="o",
    ms=8,
    markeredgecolor=SURF,
    markeredgewidth=2,
    zorder=3,
)
for k, v in enumerate(G_score.columns):
    a.text(k, G_score[v].mean() + 0.0018, f"{G_score[v].mean():+.4f}", ha="center", fontsize=9.5, fontweight="bold", color=INK)

mx = G_score["v4"].idxmax()
a.annotate(
    short_label(mx),
    (4, G_score.loc[mx, "v4"]),
    xytext=(-8, -2),
    textcoords="offset points",
    ha="right",
    fontsize=8.5,
    color=INK2,
)
a.set_xticks(list(x))
a.set_xticklabels([VL_ALL[v].replace(" ", "\n", 1) for v in G_score.columns], fontsize=9)
a.set_ylabel("change in test score vs v0", fontsize=10)
a.grid(axis="y", color=GRID, lw=0.8)
a.set_axisbelow(True)
a.set_title(
    f"Appearance features consistently improve tracking score\nThin gray = each of {len(P_score_complete)} detectors, bold = mean",
    loc="left",
    fontsize=12,
    fontweight="bold",
)

best_score = P_score.idxmax(axis=1).value_counts().reindex(SUB_VARIANTS).fillna(0)
b.bar(range(len(SUB_VARIANTS)), best_score.values, color=[VC_ALL[v] for v in SUB_VARIANTS], width=0.62)
for k, v in enumerate(best_score.values):
    b.text(k, v + 0.25, str(int(v)), ha="center", fontsize=10, fontweight="bold")
b.set_xticks(range(len(SUB_VARIANTS)))
b.set_xticklabels(SUB_VARIANTS)
b.set_ylabel("detectors where this variant is best", fontsize=10)
b.set_title("Which variant wins per detector (score)", loc="left", fontsize=12, fontweight="bold")

for ax_ in (a, b):
    for s in ["top", "right"]:
        ax_.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax_.spines[s].set_color(GRID)

fig.tight_layout()
fig.savefig(DATA_DIR / "fig2_score_variant_effect.png")
plt.close(fig)
print(f"Saved -> {DATA_DIR / 'fig2_score_variant_effect.png'}")


# ==============================================================================
# 5. Figure 3 & Figure 4: Corrected Assertion Pass Rates (40 Test Videos)
# ==============================================================================
plot_detector_by_variant(
    df_in=df,
    value_col="structural_all_pass_rate",
    xlabel="corrected structural assertion pass rate (A2–A6 all-pass across 40 test videos)",
    title="Structural assertion pass rate (A2–A6) by detector and tracker, best at top",
    callout_header="Top 5 detector + tracker pairs (A2–A6 pass)",
    callout_xy=(0.02, len(df["detector"].unique()) - 1.1),
    xlim=(-0.03, 1.05),
    out_path=DATA_DIR / "fig3_assertion_pass_by_variant.png",
)

# Figure 4: Left = variant effect on structural assertion pass rate (vs v0);
#           Right = per-assertion (A1-A6) pass rate breakdown across tracker variants
P_assert = sub_df.pivot(index="detector", columns="tracker", values="structural_all_pass_rate")[SUB_VARIANTS]
P_assert_complete = P_assert.dropna()
G_assert = P_assert_complete.sub(P_assert_complete["v0"], axis=0)

fig, (a, b) = plt.subplots(1, 2, figsize=(13.2, 5.6), dpi=150, gridspec_kw={"width_ratios": [1.35, 1.15]})

for m in P_assert_complete.index:
    a.plot(x, G_assert.loc[m].values, color="#c3c2b7", lw=1.1, zorder=1)
a.plot(
    x,
    G_assert.mean().values,
    color=VC_ALL["v4"],
    lw=3,
    marker="o",
    ms=8,
    markeredgecolor=SURF,
    markeredgewidth=2,
    zorder=3,
)
for k, v in enumerate(G_assert.columns):
    a.text(k, G_assert[v].mean() + 0.008, f"{G_assert[v].mean():+.4f}", ha="center", fontsize=9.5, fontweight="bold", color=INK)

a.axhline(0, color=GRID, lw=1.2, ls="--", zorder=0)
a.set_xticks(list(x))
a.set_xticklabels([VL_ALL[v].replace(" ", "\n", 1) for v in G_assert.columns], fontsize=9)
a.set_ylabel("change in A2–A6 pass rate vs v0", fontsize=10)
a.grid(axis="y", color=GRID, lw=0.8)
a.set_axisbelow(True)
a.set_title(
    f"Appearance features reduce division & distance violations\nThin gray = each of {len(P_assert_complete)} detectors, bold = mean",
    loc="left",
    fontsize=12,
    fontweight="bold",
)

# Right panel: Mean pass rate per individual assertion (A1..A6) across tracker variants
assertion_cols = ["a1_pass_40", "a2_pass_40", "a3_pass_40", "a4_pass_40", "a5_pass_40", "a6_pass_40"]
assertion_labels = ["A1\n(Count\nt=0.15)", "A2\n(No\nMerges)", "A3\n(Unique\nIDs)", "A4\n(Binary\nDiv.)", "A5\n(Time\nOrder)", "A6\n(Dist\n≤14µm)"]
tracker_order = ["fiji_lap", "v0", "v1", "v2", "v3", "v4"]
mean_by_tracker = df.groupby("tracker")[assertion_cols].mean().reindex(tracker_order)

x_a = np.arange(len(assertion_cols))
bar_w = 0.13
for idx_t, tr in enumerate(tracker_order):
    offset = (idx_t - (len(tracker_order) - 1) / 2.0) * bar_w
    b.bar(
        x_a + offset,
        mean_by_tracker.loc[tr].values,
        width=bar_w,
        color=VC_ALL[tr],
        edgecolor=SURF,
        linewidth=0.6,
        label=tr,
    )

b.set_xticks(x_a)
b.set_xticklabels(assertion_labels, fontsize=8.8)
b.set_ylim(0, 1.08)
b.set_ylabel("mean pass rate across 40 test videos", fontsize=10)
b.grid(axis="y", color=GRID, lw=0.8)
b.set_axisbelow(True)
b.legend(title="tracker", ncol=3, loc="upper left", frameon=False, fontsize=8.5, title_fontsize=9)
b.set_title("Per-assertion pass rate breakdown (40 test videos)", loc="left", fontsize=12, fontweight="bold")

for ax_ in (a, b):
    for s in ["top", "right"]:
        ax_.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax_.spines[s].set_color(GRID)

fig.tight_layout()
fig.savefig(DATA_DIR / "fig4_assertion_breakdown_and_effect.png")
plt.close(fig)
print(f"Saved -> {DATA_DIR / 'fig4_assertion_breakdown_and_effect.png'}")