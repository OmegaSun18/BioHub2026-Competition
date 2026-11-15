# %% [code] — Standalone Setup & Multi-Pipeline tracks.csv Evaluation
import os
import sys
import glob
import shutil
import subprocess
from pathlib import Path

# ==============================================================================
# 1. ENVIRONMENT & REPOSITORY SETUP (Adapted from Cell 0)
# ==============================================================================
INTERNET_AVAILABLE = True
WORK_DIR = Path("/kaggle/working")  # Change to Path(".") if running locally
REPO_DIR = WORK_DIR / "kaggle-cell-tracking-competition"
REPO_SRC = REPO_DIR / "src"

if INTERNET_AVAILABLE:
    if not REPO_DIR.exists():
        subprocess.run(
            ["git", "clone", "https://github.com/royerlab/kaggle-cell-tracking-competition.git", str(REPO_DIR)],
            check=True,
        )
    # Install competition package (tracking-cellmot) and zarr
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", str(REPO_DIR), "zarr"], check=True)
    # Lock NumPy back to a version compatible with preinstalled SciPy C-extensions
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "--force-reinstall", "--no-deps", "numpy>=1.26.4,<2.1.0"],
        check=True,
    )
    print("Dependencies installed and NumPy locked.")
else:
    for wp in ["/kaggle/input/cell-tracking-wheels", "/kaggle/input/Zarr-Wheels"]:
        if os.path.exists(wp):
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q", "--no-index", f"--find-links={wp}", "zarr"],
                check=True,
            )

if REPO_SRC.exists() and str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

# Verify C-extensions are healthy; auto-restart kernel if NumPy was reloaded mid-session
try:
    import scipy.ndimage
    import tracking_cellmot
    print("Environment healthy! Starting evaluation...")
except ImportError:
    print("\n[Auto-Restarting Kernel to reload clean C-extensions — re-run this cell once after restart]")
    os._exit(0)

# ==============================================================================
# 2. CONFIGURE PATHS (Ground Truth Test Dir & Pipeline Output Dir)
# ==============================================================================
BIOHUB_ROOT = Path(os.environ.get("BIOHUB_ROOT", "/kaggle/input/competitions/biohub-cell-tracking-during-development"))

# Directory containing ground-truth .geff files for your test split
GT_DIR = BIOHUB_ROOT / "test"

# Directory containing pipeline output folders or *tracks*.csv files
OUTPUT_DIR = WORK_DIR / "output"

# Temporary folder used to hold reconstructed .geff graphs during scoring
TMP_GEFF_ROOT = WORK_DIR / "_tmp_eval_geffs"

# ==============================================================================
# 3. DISCOVER & EVALUATE EACH PIPELINE'S tracks.csv
# ==============================================================================
def find_track_csvs(output_dir: Path) -> list[Path]:
    """Finds both output/<pipeline>/tracks.csv and output/*tracks*.csv."""
    if not output_dir.exists():
        return []
    matches = set(output_dir.rglob("tracks.csv")) | set(output_dir.rglob("*tracks*.csv"))
    return sorted(p for p in matches if p.is_file())


def evaluate_all_pipelines(gt_dir: Path, output_dir: Path):
    assert gt_dir.exists(), f"Ground truth directory not found: {gt_dir}"
    gt_geffs = list(gt_dir.glob("*.geff"))
    assert len(gt_geffs) > 0, (
        f"No .geff ground-truth annotations found in {gt_dir}. "
        f"(Note: on the standard Kaggle dataset, .geff files live in {BIOHUB_ROOT / 'train'}.)"
    )

    csv_files = find_track_csvs(output_dir)
    assert len(csv_files) > 0, f"No tracks.csv files found under {output_dir}"

    csv_to_geffs_script = REPO_DIR / "scripts" / "csv_to_geffs.py"
    evaluate_script = REPO_DIR / "scripts" / "evaluate.py"

    print(f"Found {len(gt_geffs)} ground-truth .geff files in: {gt_dir}")
    print(f"Found {len(csv_files)} pipeline CSV(s) to evaluate:\n")

    results = {}
    for csv_path in csv_files:
        # Label pipeline by parent folder name if named 'tracks.csv', otherwise by file stem
        pipeline_name = csv_path.parent.name if csv_path.name == "tracks.csv" else csv_path.stem
        pred_geff_dir = TMP_GEFF_ROOT / pipeline_name

        if pred_geff_dir.exists():
            shutil.rmtree(pred_geff_dir)
        pred_geff_dir.mkdir(parents=True, exist_ok=True)

        print("=" * 72)
        print(f"PIPELINE: {pipeline_name}")
        print(f"CSV Path: {csv_path}")
        print("-" * 72)

        # Step A: Convert tracks.csv -> .geff graphs
        subprocess.run(
            [
                sys.executable,
                str(csv_to_geffs_script),
                "--csv", str(csv_path),
                "--out-dir", str(pred_geff_dir),
            ],
            check=True,
        )

        # Step B: Evaluate reconstructed .geff graphs against GT_DIR
        eval_proc = subprocess.run(
            [
                sys.executable,
                str(evaluate_script),
                "--pred-dir", str(pred_geff_dir),
                "--gt-dir", str(gt_dir),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        print(eval_proc.stdout.strip())
        if eval_proc.stderr.strip():
            print("[stderr]:", eval_proc.stderr.strip())
        results[pipeline_name] = eval_proc.stdout.strip()

    # Clean up temporary reconstructed .geff files
    if TMP_GEFF_ROOT.exists():
        shutil.rmtree(TMP_GEFF_ROOT)

    return results


pipeline_scores = evaluate_all_pipelines(GT_DIR, OUTPUT_DIR)