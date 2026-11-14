import xml.etree.ElementTree as ET
import numpy as np
import zarr
import matplotlib
matplotlib.use('Agg')  # Headless backend
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# Physical voxel scales (um) used during TrackMate calibration
PX_X, PX_Y, PX_Z = 0.40625, 0.40625, 1.625

ZARR_PATH = '/kaggle/input/biohub-cell-tracking-during-development/test/44b6_0113de3b.zarr'
XML_PATH = '/kaggle/working/sanity_test_tracks.xml'
OUT_PNG = '/kaggle/working/trackmate_overlay_summary.png'
OUT_GIF = '/kaggle/working/trackmate_overlay.gif'

# 1. Parse spots and track edges from TrackMate XML
root = ET.parse(XML_PATH).getroot()

spots = {}
for spot in root.findall('.//AllSpots/SpotsInFrame/Spot'):
    sid = int(spot.attrib['ID'])
    spots[sid] = {
        'frame': int(spot.attrib['FRAME']),
        'x_px': float(spot.attrib['POSITION_X']) / PX_X,
        'y_px': float(spot.attrib['POSITION_Y']) / PX_Y,
        'z_px': float(spot.attrib['POSITION_Z']) / PX_Z,
    }

edges = []
for track in root.findall('.//AllTracks/Track'):
    tid = int(track.attrib['TRACK_ID'])
    for edge in track.findall('Edge'):
        edges.append((
            tid,
            int(edge.attrib['SPOT_SOURCE_ID']),
            int(edge.attrib['SPOT_TARGET_ID']),
        ))

# 2. Load the matching frames from Zarr and compute 2D Max-Intensity Projections (T, Y, X)
n_frames = max(s['frame'] for s in spots.values()) + 1
z_vol = np.asarray(zarr.open(ZARR_PATH, mode='r')['0'][:n_frames])  # (T, Z, Y, X)
mip = z_vol.max(axis=1)  # Max projection along Z -> (T, Y, X)
vmin, vmax = np.percentile(mip, (1, 99.5))

# 3. Save a side-by-side multi-frame PNG grid
fig, axes = plt.subplots(1, n_frames, figsize=(4 * n_frames, 4), dpi=150)
if n_frames == 1:
    axes = [axes]

cmap = plt.get_cmap('tab20')
for t, ax in enumerate(axes):
    ax.imshow(mip[t], cmap='gray', vmin=vmin, vmax=vmax)
    # Plot spots at frame t
    pts = [(s['x_px'], s['y_px']) for s in spots.values() if s['frame'] == t]
    if pts:
        xs, ys = zip(*pts)
        ax.scatter(xs, ys, s=18, facecolors='none', edgecolors='cyan', linewidths=0.8)
    # Plot track segments up to frame t
    for tid, src, dst in edges:
        if spots[dst]['frame'] <= t:
            ax.plot(
                [spots[src]['x_px'], spots[dst]['x_px']],
                [spots[src]['y_px'], spots[dst]['y_px']],
                color=cmap(tid % 20), linewidth=1.2
            )
    ax.set_title(f'Frame {t}')
    ax.axis('off')

plt.tight_layout()
plt.savefig(OUT_PNG, bbox_inches='tight')
plt.close(fig)
print(f"Saved static frame-by-frame overlay to {OUT_PNG}")

# 4. Save an animated GIF across frames
fig_anim, ax_anim = plt.subplots(figsize=(6, 6), dpi=120)
def update(t):
    ax_anim.clear()
    ax_anim.imshow(mip[t], cmap='gray', vmin=vmin, vmax=vmax)
    pts = [(s['x_px'], s['y_px']) for s in spots.values() if s['frame'] == t]
    if pts:
        xs, ys = zip(*pts)
        ax_anim.scatter(xs, ys, s=25, facecolors='none', edgecolors='cyan', linewidths=1.0)
    for tid, src, dst in edges:
        if spots[dst]['frame'] <= t:
            ax_anim.plot(
                [spots[src]['x_px'], spots[dst]['x_px']],
                [spots[src]['y_px'], spots[dst]['y_px']],
                color=cmap(tid % 20), linewidth=1.5
            )
    ax_anim.set_title(f'TrackMate Overlay — Frame {t}/{n_frames - 1}')
    ax_anim.axis('off')

anim = FuncAnimation(fig_anim, update, frames=n_frames, interval=500)
anim.save(OUT_GIF, writer=PillowWriter(fps=2))
plt.close(fig_anim)
print(f"Saved animated tracking GIF to {OUT_GIF}")