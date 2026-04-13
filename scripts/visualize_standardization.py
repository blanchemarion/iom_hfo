#!/usr/bin/env python3
"""
visualization / QC for standardized HFO detector inputs.

This script reads the .mat files produced by standardize.py
and creates basic sanity-check plots:
- stacked raw traces over a short time window
- channel x time heatmap for the same window
- amplitude histogram from a random subset of samples

Expected .mat structure:
  data.data          -> [samples x channels]
  montage.SampleRate -> scalar Hz

Usage examples:
  python scripts/visualize_standardization.py --output-dir standardized_hfo_inputs
  python scripts/visualize_standardization.py --output-dir standardized_hfo_inputs --limit 3
  python scripts/visualize_standardization.py --mat-file standardized_hfo_inputs\\sub-01_ses-SITUATION1A_task-acute.mat
  python scripts/visualize_standardization.py --output-dir standardized_hfo_inputs --seconds 2 --max-channels 6
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat


def load_detector_mat(mat_path: Path) -> tuple[np.ndarray, float]:
    """
    Load a detector-compatible .mat file and return:
      X  : [samples x channels]
      fs : sample rate in Hz
    """
    m = loadmat(str(mat_path), struct_as_record=False, squeeze_me=True)

    if "data" not in m or "montage" not in m:
        raise ValueError(f"Missing 'data' or 'montage' in {mat_path}")

    data_block = m["data"]
    montage_block = m["montage"]

    if not hasattr(data_block, "data"):
        raise ValueError(f"'data' struct has no .data field in {mat_path}")
    if not hasattr(montage_block, "SampleRate"):
        raise ValueError(f"'montage' struct has no .SampleRate in {mat_path}")

    X = np.asarray(data_block.data, dtype=np.float64)
    fs = float(np.asarray(montage_block.SampleRate).squeeze())

    if X.ndim != 2:
        raise ValueError(f"Expected 2D data.data, got shape {X.shape}")

    return X, fs


def load_manifest_ok_files(output_dir: Path) -> list[Path]:
    manifest_json = output_dir / "conversion_manifest.json"
    if not manifest_json.is_file():
        raise FileNotFoundError(f"Missing manifest: {manifest_json}")

    with manifest_json.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    mats = []
    for row in manifest:
        if row.get("status") == "ok":
            mats.append(Path(row["output_mat"]))
    return mats


def robust_channel_scale(x: np.ndarray) -> float:
    """Robust scale for one channel."""
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale <= 0:
        scale = np.std(x)
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    return float(scale)


def summarize_signal(X: np.ndarray, fs: float) -> dict:
    n_samples, n_channels = X.shape
    duration_s = n_samples / fs
    flat = X.reshape(-1)

    # avoid huge memory / runtime for percentiles on massive arrays
    if flat.size > 2_000_000:
        rng = np.random.default_rng(0)
        idx = rng.choice(flat.size, size=2_000_000, replace=False)
        flat_use = flat[idx]
    else:
        flat_use = flat

    summary = {
        "shape": X.shape,
        "fs_hz": fs,
        "duration_s": duration_s,
        "mean_uV": float(np.mean(flat_use)),
        "std_uV": float(np.std(flat_use)),
        "min_uV": float(np.min(flat_use)),
        "max_uV": float(np.max(flat_use)),
        "p01_uV": float(np.percentile(flat_use, 1)),
        "p99_uV": float(np.percentile(flat_use, 99)),
    }
    return summary


def plot_stacked_traces(
    X: np.ndarray,
    fs: float,
    out_path: Path,
    seconds: float = 5.0,
    start_sec: float = 0.0,
    max_channels: int = 8,
) -> None:
    n_samples, n_channels = X.shape
    start = int(start_sec * fs)
    stop = min(n_samples, start + int(seconds * fs))
    if start >= stop:
        raise ValueError("Invalid start_sec / seconds window")

    xw = X[start:stop, : min(n_channels, n_channels)]
    t = np.arange(start, stop) / fs

    fig, ax = plt.subplots(figsize=(10, 6))

    offsets = []
    current_offset = 0.0

    for ch in range(xw.shape[1]):
        trace = xw[:, ch]
        scale = robust_channel_scale(trace)
        trace_plot = trace / scale

        ax.plot(t, trace_plot + current_offset, linewidth=0.8)
        offsets.append(current_offset)
        current_offset += 6.0

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Channels (stacked, robust-scaled)")
    ax.set_title(f"Stacked traces: {out_path.stem}")
    ax.set_yticks(offsets)
    ax.set_yticklabels([f"ch{c}" for c in range(xw.shape[1])])
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_heatmap(
    X: np.ndarray,
    fs: float,
    out_path: Path,
    seconds: float = 5.0,
    start_sec: float = 0.0,
    max_channels: int = 16,
) -> None:
    n_samples, n_channels = X.shape
    start = int(start_sec * fs)
    stop = min(n_samples, start + int(seconds * fs))
    if start >= stop:
        raise ValueError("Invalid start_sec / seconds window")

    xw = X[start:stop, : min(max_channels, n_channels)].T  # [channels x time]

    # mild clipping for visualization only
    clip = np.percentile(np.abs(xw), 99)
    if clip <= 0 or not np.isfinite(clip):
        clip = 1.0
    xw = np.clip(xw, -clip, clip)

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(
        xw,
        aspect="auto",
        origin="lower",
        extent=[start / fs, stop / fs, 0, xw.shape[0]],
    )
    fig.colorbar(im, ax=ax, label="Amplitude (uV, clipped)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Channel index")
    ax.set_title(f"Heatmap: {out_path.stem}")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_histogram(
    X: np.ndarray,
    out_path: Path,
    n_points: int = 200_000,
) -> None:
    flat = X.reshape(-1)
    if flat.size > n_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(flat.size, size=n_points, replace=False)
        flat = flat[idx]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(flat, bins=100)
    ax.set_xlabel("Amplitude (uV)")
    ax.set_ylabel("Count")
    ax.set_title(f"Amplitude histogram: {out_path.stem}")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def process_one_mat(
    mat_path: Path,
    qc_dir: Path,
    seconds: float,
    start_sec: float,
    max_channels: int,
) -> None:
    X, fs = load_detector_mat(mat_path)
    summary = summarize_signal(X, fs)

    logging.info(
        "%s | shape=%s | fs=%.1f Hz | duration=%.2f s | mean=%.3f uV | std=%.3f uV | p01=%.3f | p99=%.3f",
        mat_path.name,
        summary["shape"],
        summary["fs_hz"],
        summary["duration_s"],
        summary["mean_uV"],
        summary["std_uV"],
        summary["p01_uV"],
        summary["p99_uV"],
    )

    stem = mat_path.stem
    plot_stacked_traces(
        X,
        fs,
        qc_dir / f"{stem}_stacked.png",
        seconds=seconds,
        start_sec=start_sec,
        max_channels=max_channels,
    )
    plot_heatmap(
        X,
        fs,
        qc_dir / f"{stem}_heatmap.png",
        seconds=seconds,
        start_sec=start_sec,
        max_channels=max(16, max_channels),
    )
    plot_histogram(
        X,
        qc_dir / f"{stem}_hist.png",
    )


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Visualize QC outputs from standardized HFO .mat files.")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("standardized_hfo_inputs"),
        help="Directory containing .mat files and conversion_manifest.json",
    )
    p.add_argument(
        "--mat-file",
        type=Path,
        default=None,
        help="Optional single .mat file to inspect. If omitted, uses manifest.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=5,
        help="How many successful .mat files to visualize when using the manifest",
    )
    p.add_argument(
        "--seconds",
        type=float,
        default=5.0,
        help="Window length in seconds for trace / heatmap plots",
    )
    p.add_argument(
        "--start-sec",
        type=float,
        default=0.0,
        help="Start time in seconds for plotted window",
    )
    p.add_argument(
        "--max-channels",
        type=int,
        default=8,
        help="Max number of channels for stacked traces",
    )
    p.add_argument(
        "--qc-dir",
        type=Path,
        default=None,
        help="Output directory for QC figures (default: <output-dir>/qc_plots)",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p


def main() -> int:
    args = build_argparser().parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s %(message)s")

    output_dir = args.output_dir.expanduser().resolve()
    qc_dir = (args.qc_dir or (output_dir / "qc_plots")).expanduser().resolve()
    qc_dir.mkdir(parents=True, exist_ok=True)

    if args.mat_file is not None:
        mat_files = [args.mat_file.expanduser().resolve()]
    else:
        mat_files = load_manifest_ok_files(output_dir)
        if not mat_files:
            logging.error("No successful .mat files found in manifest.")
            return 1
        mat_files = mat_files[: args.limit]

    logging.info("Visualizing %d file(s) into %s", len(mat_files), qc_dir)

    for mat_path in mat_files:
        try:
            process_one_mat(
                mat_path=mat_path,
                qc_dir=qc_dir,
                seconds=args.seconds,
                start_sec=args.start_sec,
                max_channels=args.max_channels,
            )
        except Exception as e:  # noqa: BLE001
            logging.error("Failed on %s: %s: %s", mat_path, type(e).__name__, e)

    logging.info("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())