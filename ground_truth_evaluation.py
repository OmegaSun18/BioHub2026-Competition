# %% [code] — Standalone Setup & Multi-Pipeline tracks.csv Evaluation
import importlib.util
import os
import re
import sys
import shutil
import subprocess
from pathlib import Path

# ==============================================================================
# 1. ENVIRONMENT & REPOSITORY SETUP
# ==============================================================================
INTERNET_AVAILABLE = True
WORK_DIR = Path("/kaggle/working")  # Change to Path(".") if running locally
REPO_DIR = WORK_DIR / "kaggle-cell-tracking-competition"
REPO_SRC = REPO_DIR / "src"

def is_package_installed(module_name: str) -> bool:
    """Checks if a Python package is importable without triggering C-extension imports."""
    return importlib.util.find_spec(module_name) is not None

# Check if required packages and repository scripts are already present
packages_ready = is_package_installed("tracking_cellmot") and is_package_installed("zarr")
scripts_ready = (REPO_DIR / "scripts" / "csv_to_geffs.py").exists() and (REPO_DIR / "scripts" / "evaluate.py").exists()

if packages_ready and scripts_ready:
    print("Libraries (tracking_cellmot, zarr) and repository scripts already installed. Skipping clone and pip install.")
else:
    if INTERNET_AVAILABLE:
        if not REPO_DIR.exists():
            subprocess.run(
                ["git", "clone", "https://github.com/royerlab/kaggle-cell-tracking-competition.git", str(REPO_DIR)],
                check=True,
            )
        # Install competition package (tracking-cellmot) and zarr
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--break-system-packages", "-q", str(REPO_DIR), "zarr"],
            check=True,
        )
        # Lock NumPy back to a version compatible with preinstalled SciPy C-extensions
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "--force-reinstall", "--break-system-packages", "--no-deps", "numpy>=1.26.4,<2.1.0"],
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
# 2. CONFIGURE PATHS (Ground Truth Test Dir & Three Pipeline Output Dirs)
# ==============================================================================
BIOHUB_ROOT = Path(os.environ.get("BIOHUB_ROOT", "/kaggle/working/split_dataset"))

# Directory containing ground-truth .geff files for your test split
GT_DIR = BIOHUB_ROOT / "test"

# Explicit list of the three pipeline output directories
PIPELINE_DIRS = [
    WORK_DIR / "output_pipeline1",
    WORK_DIR / "output_pipeline2",
    WORK_DIR / "output_pipeline3",
]

# Temporary folder used to hold reconstructed .geff graphs during scoring
TMP_GEFF_ROOT = WORK_DIR / "_tmp_eval_geffs"

# ==============================================================================
# 3. LOCATE tracks.csv & EVALUATE EACH PIPELINE DIRECTORY
# ==============================================================================
def resolve_pipeline_csv(pipeline_dir: Path) -> Path | None:
    """Finds tracks.csv inside the given pipeline directory (with fallback to WORK_DIR/output/<name>)."""
    candidate_dirs = [pipeline_dir, WORK_DIR / "output" / pipeline_dir.name]
    for d in candidate_dirs:
        if not d.exists():
            continue
        direct_csv = d / "tracks.csv"
        if direct_csv.is_file():
            return direct_csv
        matches = sorted(p for p in d.rglob("*tracks*.csv") if p.is_file())
        if matches:
            return matches[0]
    return None


def extract_score_summary(eval_stdout: str) -> str:
    """Extracts key overall score/accuracy lines from evaluate.py output."""
    key_lines = [
        line.strip()
        for line in eval_stdout.splitlines()
        if re.search(r"(final|combined|overall|total|jaccard|score|accuracy)", line, re.IGNORECASE)
    ]
    return " | ".join(key_lines) if key_lines else eval_stdout.strip().splitlines()[-1]


def evaluate_pipelines(gt_dir: Path, pipeline_dirs: list[Path]) -> dict[str, str]:
    assert gt_dir.exists(), f"Ground truth directory not found: {gt_dir}"
    gt_geffs = list(gt_dir.glob("*.geff"))
    assert len(gt_geffs) > 0, (
        f"No .geff ground-truth annotations found in {gt_dir}. "
        f"(Note: on the standard Kaggle dataset, .geff files live in {BIOHUB_ROOT / 'train'}.)"
    )

    csv_to_geffs_script = REPO_DIR / "scripts" / "csv_to_geffs.py"
    evaluate_script = REPO_DIR / "scripts" / "evaluate.py"

    print(f"Found {len(gt_geffs)} ground-truth .geff files in: {gt_dir}\n")

    results = {}
    for pipeline_dir in pipeline_dirs:
        pipeline_name = pipeline_dir.name
        csv_path = resolve_pipeline_csv(pipeline_dir)

        print("=" * 72)
        print(f"PIPELINE: {pipeline_name}")
        if csv_path is None:
            print(f"[SKIPPED] No tracks.csv found in {pipeline_dir}")
            print("=" * 72 + "\n")
            continue

        print(f"CSV Path: {csv_path}")
        print("-" * 72)

        pred_geff_dir = TMP_GEFF_ROOT / pipeline_name
        if pred_geff_dir.exists():
            shutil.rmtree(pred_geff_dir)
        pred_geff_dir.mkdir(parents=True, exist_ok=True)

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
        stdout_clean = eval_proc.stdout.strip()
        print(stdout_clean)
        if eval_proc.stderr.strip():
            print("[stderr]:", eval_proc.stderr.strip())
        print()

        results[pipeline_name] = stdout_clean

    # Clean up temporary reconstructed .geff files
    if TMP_GEFF_ROOT.exists():
        shutil.rmtree(TMP_GEFF_ROOT)

    assert len(results) > 0, (
        "No tracks.csv files were found in any of: "
        + ", ".join(str(d) for d in pipeline_dirs)
    )

    # Print side-by-side summary across all evaluated pipelines
    print("=" * 72)
    print("FINAL EVALUATION SUMMARY ACROSS PIPELINES")
    print("=" * 72)
    for pipeline_name, raw_output in results.items():
        print(f"{pipeline_name:<20} -> {extract_score_summary(raw_output)}")
    print("=" * 72)

    return results


pipeline_scores = evaluate_pipelines(GT_DIR, PIPELINE_DIRS)