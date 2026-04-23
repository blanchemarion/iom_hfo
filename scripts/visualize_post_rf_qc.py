#!/usr/bin/env python3
"""
QC visualization for post-RF Demo_Run.py outputs.

Focus:
  - sanity-check detector outputs and RF split (pseudo vs real)
  - inspect confidence, class morphology, time/channel concentration
  - inspect feature-space structure relative to predicted class

Expected files in --output-dir for a recording stem:
  <stem>_events.npz
  <stem>_features.npz
  <stem>_predictions.npz
  <stem>_manifest.json
Optional:
  <stem>_pipeline_post_rf.mat
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _resolve_stem(
    output_dir: Path,
    recording_stem: str | None,
    manifest_path: Path | None,
) -> tuple[str, Path]:
    if manifest_path is not None:
        m = json.loads(manifest_path.read_text(encoding="utf-8"))
        stem = Path(m["input_mat"]).stem
        return stem, manifest_path
    if recording_stem:
        mp = output_dir / f"{recording_stem}_manifest.json"
        return recording_stem, mp
    manifests = sorted(output_dir.glob("*_manifest.json"))
    if not manifests:
        raise FileNotFoundError(f"No *_manifest.json found in {output_dir}")
    mp = manifests[-1]
    return mp.name.replace("_manifest.json", ""), mp


def _save_figure(fig: plt.Figure, png_path: Path, save_svg: bool) -> None:
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    if save_svg:
        fig.savefig(png_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def _sample_indices(n: int, k: int, seed: int) -> np.ndarray:
    if n <= 0:
        return np.array([], dtype=int)
    if n <= k:
        return np.arange(n, dtype=int)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=k, replace=False))


def _zscore_columns(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    mu = np.mean(x, axis=0, keepdims=True)
    sd = np.std(x, axis=0, keepdims=True)
    return (x - mu) / np.maximum(sd, eps)


def _pca_2d(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    PCA via SVD on z-scored features.
    Returns:
      scores: [n_samples x 2]
      explained_ratio: [2]
    """
    xz = _zscore_columns(x)
    xz = xz - np.mean(xz, axis=0, keepdims=True)
    # xz = U S Vt, projected scores = U S
    u, s, _vt = np.linalg.svd(xz, full_matrices=False)
    k = min(2, s.size)
    scores = u[:, :k] * s[:k]
    var = (s**2) / max(xz.shape[0] - 1, 1)
    ratio = var / np.sum(var) if np.sum(var) > 0 else np.zeros_like(var)
    if k < 2:
        scores = np.column_stack([scores, np.zeros((scores.shape[0], 2 - k))])
        ratio = np.pad(ratio[:k], (0, 2 - k), mode="constant")
    return scores[:, :2], ratio[:2]


def _load_dictionary_layer1(path: Path) -> np.ndarray:
    m = loadmat(str(path), struct_as_record=False, squeeze_me=True)
    d = m["Dictionary"]
    layers = d.L2kSVD
    return np.asarray(layers[0], dtype=np.float64)


def _sliding_match_scores(
    events: np.ndarray,
    atoms: np.ndarray,
    hop: int = 4,
    near_constant_std_threshold: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """
    For each event, return best atom index and best abs normalized correlation score.
    events shape: [samples x events], atoms shape: [atom_len x n_atoms]
    """
    atom_std = np.std(atoms, axis=0)
    valid = atom_std > near_constant_std_threshold
    if not np.any(valid):
        raise ValueError("No valid atoms for template matching.")

    atom_len, n_atoms = atoms.shape
    n_samples, n_events = events.shape
    if atom_len > n_samples:
        raise ValueError("Atom length exceeds event length.")

    starts = np.arange(0, n_samples - atom_len + 1, hop, dtype=int)
    atom_n = atoms / np.maximum(np.linalg.norm(atoms, axis=0, keepdims=True), 1e-12)
    best_atom = np.zeros(n_events, dtype=int)
    best_score = np.zeros(n_events, dtype=np.float64)

    valid_idx = np.where(valid)[0]
    for i in range(n_events):
        ev = events[:, i]
        win = np.stack([ev[s : s + atom_len] for s in starts], axis=0)
        win = win / np.maximum(np.linalg.norm(win, axis=1, keepdims=True), 1e-12)
        corr = win @ atom_n
        abs_corr = np.abs(corr[:, valid_idx])
        j, k_local = np.unravel_index(np.argmax(abs_corr), abs_corr.shape)
        k = int(valid_idx[k_local])
        best_atom[i] = k
        best_score[i] = float(np.abs(corr[j, k]))
    return best_atom, best_score


def _plot_prediction_summary(
    pred: np.ndarray,
    confidence: np.ndarray | None,
    out_path: Path,
    save_svg: bool,
) -> dict:
    n = int(pred.size)
    n_p = int(np.sum(pred == 1))
    n_r = int(np.sum(pred == 2))
    frac_p = (n_p / n) if n else 0.0
    frac_r = (n_r / n) if n else 0.0
    conf_available = confidence is not None and confidence.size == n

    fig = plt.figure(figsize=(10, 4.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.2])
    ax_txt = fig.add_subplot(gs[0, 0])
    ax_hist = fig.add_subplot(gs[0, 1])

    summary_lines = [
        f"Total events: {n}",
        f"Pseudo-HFO (pred==1): {n_p} ({frac_p:.1%})",
        f"Real-HFO (pred==2): {n_r} ({frac_r:.1%})",
        f"Confidence available: {conf_available}",
    ]
    if conf_available:
        summary_lines.extend(
            [
                f"Confidence min/median/max: "
                f"{np.min(confidence):.3f} / {np.median(confidence):.3f} / {np.max(confidence):.3f}",
            ]
        )
    ax_txt.axis("off")
    ax_txt.text(
        0.02,
        0.98,
        "\n".join(summary_lines),
        va="top",
        ha="left",
        fontsize=11,
        family="monospace",
    )
    ax_txt.set_title("Prediction Summary")

    if conf_available:
        ax_hist.hist(confidence, bins=20)
        ax_hist.set_xlabel("Confidence (|p(real)-p(pseudo)|)")
        ax_hist.set_ylabel("Count")
        ax_hist.set_title("Confidence histogram")
        ax_hist.grid(alpha=0.3)
    else:
        ax_hist.axis("off")
        ax_hist.text(0.5, 0.5, "No confidence scores available", ha="center", va="center")

    fig.tight_layout()
    _save_figure(fig, out_path, save_svg)
    return {
        "total_events": n,
        "pseudo_count": n_p,
        "real_count": n_r,
        "pseudo_fraction": frac_p,
        "real_fraction": frac_r,
        "confidence_available": conf_available,
    }


def _plot_class_gallery(
    event_raw: np.ndarray,
    pred: np.ndarray,
    confidence: np.ndarray | None,
    channelinfo: np.ndarray,
    cls: int,
    fs: float,
    out_path: Path,
    seed: int,
    n_show: int = 36,
    y_lim: tuple[float, float] | None = None,
    save_svg: bool = False,
) -> int:
    idx_all = np.where(pred == cls)[0]
    if idx_all.size == 0:
        fig, ax = plt.subplots(figsize=(8, 2.8))
        ax.axis("off")
        ax.text(0.5, 0.5, f"No events predicted as class {cls}", ha="center", va="center")
        _save_figure(fig, out_path, save_svg)
        return 0

    loc = _sample_indices(idx_all.size, min(n_show, idx_all.size), seed)
    idx = idx_all[loc]
    rows, cols = 6, 6
    t_ms = (np.arange(event_raw.shape[0]) - event_raw.shape[0] // 2) / fs * 1e3
    fig, axes = plt.subplots(rows, cols, figsize=(12, 9), sharex=True, sharey=True)
    axes = axes.ravel()
    for i, ax in enumerate(axes):
        if i >= idx.size:
            ax.axis("off")
            continue
        ev_i = int(idx[i])
        ax.plot(t_ms, event_raw[:, ev_i], linewidth=0.8)
        ax.axvline(0.0, color="k", alpha=0.2, linewidth=0.7)
        if y_lim is not None:
            ax.set_ylim(y_lim)
        conf_txt = ""
        if confidence is not None and confidence.size == pred.size:
            conf_txt = f" c={confidence[ev_i]:.2f}"
        ax.set_title(f"#{ev_i} ch={int(channelinfo[ev_i])}{conf_txt}", fontsize=7)
    class_name = "pseudo-HFO" if cls == 1 else "real-HFO"
    fig.suptitle(f"Event gallery: {class_name}")
    fig.supxlabel("Time (ms)")
    fig.supylabel("Amplitude (uV)")
    fig.tight_layout()
    _save_figure(fig, out_path, save_svg)
    return int(idx.size)


def _plot_class_waveform_summary(
    event_raw: np.ndarray,
    pred: np.ndarray,
    fs: float,
    out_path: Path,
    save_svg: bool,
) -> None:
    t_ms = (np.arange(event_raw.shape[0]) - event_raw.shape[0] // 2) / fs * 1e3
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True, sharey=True)

    for ax, cls, title in (
        (axes[0], 1, "Pseudo-HFO (pred==1)"),
        (axes[1], 2, "Real-HFO (pred==2)"),
    ):
        idx = np.where(pred == cls)[0]
        if idx.size == 0:
            ax.text(0.5, 0.5, "No events", ha="center", va="center")
            ax.set_title(title)
            ax.grid(alpha=0.3)
            continue
        x = event_raw[:, idx]
        mu = np.mean(x, axis=1)
        med = np.median(x, axis=1)
        sd = np.std(x, axis=1)
        p10 = np.percentile(x, 10, axis=1)
        p90 = np.percentile(x, 90, axis=1)
        ax.fill_between(t_ms, p10, p90, alpha=0.25, label="P10-P90")
        ax.fill_between(t_ms, mu - sd, mu + sd, alpha=0.2, label="Mean±SD")
        ax.plot(t_ms, mu, linewidth=1.4, label="Mean")
        ax.plot(t_ms, med, linewidth=1.2, label="Median")
        ax.axvline(0.0, color="k", alpha=0.2, linewidth=0.8)
        ax.set_title(f"{title} (n={idx.size})")
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=8)

    fig.supxlabel("Time (ms)")
    fig.supylabel("Amplitude (uV)")
    fig.suptitle("Class-specific waveform summaries")
    fig.tight_layout()
    _save_figure(fig, out_path, save_svg)


def _plot_timeline(
    timestamp: np.ndarray,
    channelinfo: np.ndarray,
    pred: np.ndarray,
    confidence: np.ndarray | None,
    fs: float,
    out_path: Path,
    save_svg: bool,
) -> None:
    t_sec = timestamp / fs
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    ax0, ax1 = axes

    # Time-only timeline
    if confidence is not None and confidence.size == pred.size:
        alpha = np.clip(0.2 + 0.8 * confidence, 0.2, 1.0)
    else:
        alpha = np.full(pred.shape[0], 0.6)
    y = np.zeros_like(t_sec)
    mask_p = pred == 1
    mask_r = pred == 2
    # Matplotlib can fail on empty alpha arrays; guard class-wise scatter calls.
    if np.any(mask_p):
        ax0.scatter(
            t_sec[mask_p],
            y[mask_p] + 0.0,
            s=10,
            alpha=alpha[mask_p],
            label="pseudo (1)",
        )
    if np.any(mask_r):
        ax0.scatter(
            t_sec[mask_r],
            y[mask_r] + 1.0,
            s=10,
            alpha=alpha[mask_r],
            label="real (2)",
        )
    ax0.set_yticks([0, 1])
    ax0.set_yticklabels(["pseudo", "real"])
    ax0.set_ylabel("Predicted class")
    ax0.set_title("Detection timeline (color by class, alpha by confidence)")
    ax0.grid(alpha=0.3)
    handles, labels = ax0.get_legend_handles_labels()
    if handles:
        ax0.legend(loc="best")
    else:
        ax0.text(0.5, 0.5, "No classified events", transform=ax0.transAxes, ha="center", va="center")

    # Raster-like channel vs time
    c = np.where(mask_p, "tab:orange", "tab:blue")
    ax1.scatter(t_sec, channelinfo, c=c, s=12, alpha=0.7)
    ax1.set_xlabel("Time in recording (s)")
    ax1.set_ylabel("Channel index")
    ax1.set_title("Channel-time raster of detected events")
    ax1.grid(alpha=0.3)

    fig.tight_layout()
    _save_figure(fig, out_path, save_svg)


def _plot_channel_distribution(
    channelinfo: np.ndarray,
    pred: np.ndarray,
    out_path: Path,
    save_svg: bool,
) -> None:
    ch = np.asarray(channelinfo, dtype=int).reshape(-1)
    if ch.size == 0:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.axis("off")
        ax.text(0.5, 0.5, "No events.", ha="center", va="center")
        _save_figure(fig, out_path, save_svg)
        return
    ch_min = int(np.min(ch))
    ch_max = int(np.max(ch))
    bins = np.arange(ch_min, ch_max + 2) - 0.5
    counts_all, _ = np.histogram(ch, bins=bins)
    counts_p, _ = np.histogram(ch[pred == 1], bins=bins)
    counts_r, _ = np.histogram(ch[pred == 2], bins=bins)
    x = np.arange(ch_min, ch_max + 1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    ax0, ax1 = axes
    ax0.bar(x, counts_all, width=0.85, color="0.5")
    ax0.set_title("All candidate events per channel")
    ax0.set_xlabel("Channel index")
    ax0.set_ylabel("Count")
    ax0.grid(alpha=0.3, axis="y")

    w = 0.4
    ax1.bar(x - w / 2, counts_p, width=w, label="pseudo (1)", color="tab:orange")
    ax1.bar(x + w / 2, counts_r, width=w, label="real (2)", color="tab:blue")
    ax1.set_title("Per-channel counts by predicted class")
    ax1.set_xlabel("Channel index")
    ax1.set_ylabel("Count")
    ax1.grid(alpha=0.3, axis="y")
    ax1.legend(loc="best")

    fig.tight_layout()
    _save_figure(fig, out_path, save_svg)


def _plot_confidence_inspection(
    confidence: np.ndarray | None,
    pred: np.ndarray,
    event_raw: np.ndarray,
    out_path: Path,
    save_svg: bool,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    if confidence is None or confidence.size != pred.size:
        for ax in axes.ravel():
            ax.axis("off")
            ax.text(0.5, 0.5, "No confidence available.", ha="center", va="center")
        fig.tight_layout()
        _save_figure(fig, out_path, save_svg)
        return

    amp = np.ptp(event_raw, axis=0)
    energy = np.sum(event_raw**2, axis=0)
    mask_p = pred == 1
    mask_r = pred == 2

    axes[0, 0].hist(confidence, bins=20)
    axes[0, 0].set_title("Confidence histogram (all)")
    axes[0, 0].set_xlabel("Confidence")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].hist(confidence[mask_p], bins=20, alpha=0.65, label="pseudo")
    axes[0, 1].hist(confidence[mask_r], bins=20, alpha=0.65, label="real")
    axes[0, 1].set_title("Confidence by predicted class")
    axes[0, 1].set_xlabel("Confidence")
    axes[0, 1].set_ylabel("Count")
    axes[0, 1].legend(loc="best")
    axes[0, 1].grid(alpha=0.3)

    axes[1, 0].scatter(amp[mask_p], confidence[mask_p], s=16, alpha=0.6, label="pseudo")
    axes[1, 0].scatter(amp[mask_r], confidence[mask_r], s=16, alpha=0.6, label="real")
    axes[1, 0].set_title("Confidence vs peak-to-peak amplitude")
    axes[1, 0].set_xlabel("Peak-to-peak amplitude (uV)")
    axes[1, 0].set_ylabel("Confidence")
    axes[1, 0].legend(loc="best")
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].scatter(energy[mask_p], confidence[mask_p], s=16, alpha=0.6, label="pseudo")
    axes[1, 1].scatter(energy[mask_r], confidence[mask_r], s=16, alpha=0.6, label="real")
    axes[1, 1].set_title("Confidence vs event energy")
    axes[1, 1].set_xlabel("Energy (sum of squares)")
    axes[1, 1].set_ylabel("Confidence")
    axes[1, 1].legend(loc="best")
    axes[1, 1].grid(alpha=0.3)

    fig.tight_layout()
    _save_figure(fig, out_path, save_svg)


def _plot_feature_space(
    features: np.ndarray,
    feature_names: list[str],
    pred: np.ndarray,
    out_path: Path,
    save_svg: bool,
) -> None:
    # Heatmap sorted by class then confidence-like spread in first feature
    order = np.argsort(pred)
    x = features[order, :]
    xz = _zscore_columns(x)
    n_feat = xz.shape[1]

    scores, explained = _pca_2d(features)
    mask_p = pred == 1
    mask_r = pred == 2

    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.2, 1.0])
    ax_hm = fig.add_subplot(gs[0, 0])
    ax_pca = fig.add_subplot(gs[1, 0])

    im = ax_hm.imshow(xz.T, aspect="auto", origin="lower", cmap="coolwarm", vmin=-3, vmax=3)
    fig.colorbar(im, ax=ax_hm, label="Feature z-score")
    ax_hm.set_yticks(np.arange(n_feat))
    ax_hm.set_yticklabels(feature_names)
    ax_hm.set_xlabel("Events (sorted by predicted class)")
    ax_hm.set_ylabel("Feature")
    ax_hm.set_title("Feature heatmap by event")
    split = int(np.sum(pred == 1))
    ax_hm.axvline(split - 0.5, color="k", linestyle="--", alpha=0.5)

    ax_pca.scatter(scores[mask_p, 0], scores[mask_p, 1], s=22, alpha=0.7, label="pseudo (1)")
    ax_pca.scatter(scores[mask_r, 0], scores[mask_r, 1], s=22, alpha=0.7, label="real (2)")
    ax_pca.set_xlabel(f"PC1 ({explained[0]*100:.1f}% var)")
    ax_pca.set_ylabel(f"PC2 ({explained[1]*100:.1f}% var)")
    ax_pca.set_title("PCA of 12 post-ASLR features")
    ax_pca.grid(alpha=0.3)
    ax_pca.legend(loc="best")

    fig.tight_layout()
    _save_figure(fig, out_path, save_svg)


def _plot_hard_examples(
    event_raw: np.ndarray,
    pred: np.ndarray,
    confidence: np.ndarray | None,
    fs: float,
    out_path: Path,
    save_svg: bool,
    k_each: int = 6,
) -> None:
    fig, axes = plt.subplots(3, k_each, figsize=(2.3 * k_each, 8), sharex=True, sharey=True)
    t_ms = (np.arange(event_raw.shape[0]) - event_raw.shape[0] // 2) / fs * 1e3

    if confidence is None or confidence.size != pred.size:
        for ax in axes.ravel():
            ax.axis("off")
            ax.text(0.5, 0.5, "No confidence available.", ha="center", va="center")
        fig.tight_layout()
        _save_figure(fig, out_path, save_svg)
        return

    idx_real = np.where(pred == 2)[0]
    idx_pseudo = np.where(pred == 1)[0]
    idx_amb = np.argsort(confidence)[:k_each]
    idx_real_hi = idx_real[np.argsort(confidence[idx_real])[::-1][:k_each]] if idx_real.size else np.array([], dtype=int)
    idx_pseudo_hi = idx_pseudo[np.argsort(confidence[idx_pseudo])[::-1][:k_each]] if idx_pseudo.size else np.array([], dtype=int)

    rows = [
        (idx_real_hi, "Highest-confidence real-HFO"),
        (idx_pseudo_hi, "Highest-confidence pseudo-HFO"),
        (idx_amb, "Most ambiguous (lowest confidence)"),
    ]
    for r, (idx_row, title) in enumerate(rows):
        for c in range(k_each):
            ax = axes[r, c]
            if c >= idx_row.size:
                ax.axis("off")
                continue
            i = int(idx_row[c])
            ax.plot(t_ms, event_raw[:, i], linewidth=0.9)
            ax.axvline(0.0, color="k", alpha=0.2, linewidth=0.7)
            ax.set_title(f"#{i} p={int(pred[i])} c={confidence[i]:.2f}", fontsize=8)
            ax.grid(alpha=0.25)
            if c == 0:
                ax.set_ylabel(title)
    fig.supxlabel("Time (ms)")
    fig.supylabel("Amplitude (uV)")
    fig.suptitle("Hard and ambiguous examples")
    fig.tight_layout()
    _save_figure(fig, out_path, save_svg)


def _plot_dictionary_match_by_class(
    event_raw: np.ndarray,
    pred: np.ndarray,
    d1: np.ndarray,
    out_path: Path,
    save_svg: bool,
) -> dict:
    if event_raw.shape[1] == 0:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.axis("off")
        ax.text(0.5, 0.5, "No events.", ha="center", va="center")
        _save_figure(fig, out_path, save_svg)
        return {"available": False}

    best_atom, best_score = _sliding_match_scores(event_raw, d1, hop=4)
    mask_p = pred == 1
    mask_r = pred == 2

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].hist(best_score[mask_p], bins=20, alpha=0.7, label="pseudo")
    axes[0].hist(best_score[mask_r], bins=20, alpha=0.7, label="real")
    axes[0].set_title("Best layer-1 atom match score by class")
    axes[0].set_xlabel("Best |corr|")
    axes[0].set_ylabel("Count")
    axes[0].legend(loc="best")
    axes[0].grid(alpha=0.3)

    n_atoms = d1.shape[1]
    counts_p = np.bincount(best_atom[mask_p], minlength=n_atoms)
    counts_r = np.bincount(best_atom[mask_r], minlength=n_atoms)
    top = np.argsort(counts_p + counts_r)[::-1][:20]
    x = np.arange(top.size)
    axes[1].bar(x - 0.2, counts_p[top], width=0.4, label="pseudo")
    axes[1].bar(x + 0.2, counts_r[top], width=0.4, label="real")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([str(int(i)) for i in top], rotation=45)
    axes[1].set_title("Top matched layer-1 atoms by class")
    axes[1].set_xlabel("Atom index")
    axes[1].set_ylabel("Best-match count")
    axes[1].legend(loc="best")
    axes[1].grid(alpha=0.3, axis="y")

    fig.tight_layout()
    _save_figure(fig, out_path, save_svg)
    return {
        "available": True,
        "score_mean_pseudo": float(np.mean(best_score[mask_p])) if np.any(mask_p) else None,
        "score_mean_real": float(np.mean(best_score[mask_r])) if np.any(mask_r) else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Post-RF QC visualization for Demo_Run.py outputs."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_repo_root() / "output" / "demo_run_py",
        help="Directory containing post-RF files (<stem>_events/features/predictions/manifest).",
    )
    parser.add_argument(
        "--recording-stem",
        type=str,
        default=None,
        help="Recording stem, e.g., sub-01_ses-SITUATION1A_task-acute",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Path to a specific <stem>_manifest.json (overrides --recording-stem).",
    )
    parser.add_argument(
        "--qc-dir",
        type=Path,
        default=None,
        help="Output QC directory (default: <output-dir>/<stem>_post_rf_qc).",
    )
    parser.add_argument(
        "--dictionary",
        type=Path,
        default=_repo_root() / "Data" / "Dictionary_CRDL_ASLR.mat",
        help="Dictionary file for optional template-matching QC.",
    )
    parser.add_argument(
        "--with-dictionary-match",
        action="store_true",
        help="Enable optional layer-1 template match analysis split by class.",
    )
    parser.add_argument("--gallery-size", type=int, default=36)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--save-svg",
        action="store_true",
        help="Also save SVG versions of figures.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    manifest_path = args.manifest.resolve() if args.manifest else None
    stem, guessed_manifest = _resolve_stem(output_dir, args.recording_stem, manifest_path)
    if manifest_path is None:
        manifest_path = guessed_manifest

    events_path = output_dir / f"{stem}_events.npz"
    features_path = output_dir / f"{stem}_features.npz"
    pred_path = output_dir / f"{stem}_predictions.npz"
    if not events_path.is_file():
        raise FileNotFoundError(f"Missing file: {events_path}")
    if not features_path.is_file():
        raise FileNotFoundError(f"Missing file: {features_path}")
    if not pred_path.is_file():
        raise FileNotFoundError(f"Missing file: {pred_path}")

    qc_dir = args.qc_dir.resolve() if args.qc_dir else output_dir / f"{stem}_post_rf_qc"
    qc_dir.mkdir(parents=True, exist_ok=True)

    ev = np.load(events_path, allow_pickle=True)
    ft = np.load(features_path, allow_pickle=True)
    pr = np.load(pred_path, allow_pickle=True)

    # event_raw orientation is expected to be [samples x events]
    event_raw = np.asarray(ev["event_raw"], dtype=np.float64)
    timestamp = np.asarray(ev["timestamp"], dtype=np.float64).reshape(-1)
    channelinfo = np.asarray(ev["channelinformation"], dtype=int).reshape(-1)
    fs = float(np.asarray(ev["fs"]).squeeze())

    features = np.asarray(ft["features"], dtype=np.float64)
    feature_names = [str(x) for x in np.asarray(ft["feature_names"]).tolist()]

    pred = np.asarray(pr["pred"], dtype=int).reshape(-1)
    votes = np.asarray(pr["votes"], dtype=np.float64) if "votes" in pr.files else None
    vote_fraction = (
        np.asarray(pr["vote_fraction"], dtype=np.float64) if "vote_fraction" in pr.files else None
    )

    if event_raw.ndim != 2:
        raise ValueError(f"Expected event_raw [samples x events], got shape {event_raw.shape}")
    n_events = event_raw.shape[1]
    if pred.size != n_events:
        raise ValueError(f"pred size {pred.size} does not match n_events {n_events}")
    if features.shape[0] != n_events:
        raise ValueError(f"features rows {features.shape[0]} do not match n_events {n_events}")

    confidence = None
    if vote_fraction is not None and vote_fraction.ndim == 2 and vote_fraction.shape[1] >= 2:
        confidence = np.abs(vote_fraction[:, 1] - vote_fraction[:, 0])
    elif votes is not None and votes.ndim == 2 and votes.shape[1] >= 2:
        denom = np.maximum(np.sum(votes, axis=1), 1.0)
        confidence = np.abs(votes[:, 1] - votes[:, 0]) / denom

    global_y = float(np.max(np.abs(event_raw))) if n_events > 0 else 1.0
    y_lim = (-global_y, global_y)

    summary = _plot_prediction_summary(
        pred=pred,
        confidence=confidence,
        out_path=qc_dir / "prediction_summary.png",
        save_svg=args.save_svg,
    )
    n_gallery_real = _plot_class_gallery(
        event_raw=event_raw,
        pred=pred,
        confidence=confidence,
        channelinfo=channelinfo,
        cls=2,
        fs=fs,
        out_path=qc_dir / "event_gallery_real_hfo.png",
        seed=args.seed + 1,
        n_show=args.gallery_size,
        y_lim=y_lim,
        save_svg=args.save_svg,
    )
    n_gallery_pseudo = _plot_class_gallery(
        event_raw=event_raw,
        pred=pred,
        confidence=confidence,
        channelinfo=channelinfo,
        cls=1,
        fs=fs,
        out_path=qc_dir / "event_gallery_pseudo_hfo.png",
        seed=args.seed + 2,
        n_show=args.gallery_size,
        y_lim=y_lim,
        save_svg=args.save_svg,
    )
    _plot_class_waveform_summary(
        event_raw=event_raw,
        pred=pred,
        fs=fs,
        out_path=qc_dir / "class_waveform_summaries.png",
        save_svg=args.save_svg,
    )
    _plot_timeline(
        timestamp=timestamp,
        channelinfo=channelinfo,
        pred=pred,
        confidence=confidence,
        fs=fs,
        out_path=qc_dir / "detection_timeline.png",
        save_svg=args.save_svg,
    )
    _plot_channel_distribution(
        channelinfo=channelinfo,
        pred=pred,
        out_path=qc_dir / "channel_distribution_by_class.png",
        save_svg=args.save_svg,
    )
    _plot_confidence_inspection(
        confidence=confidence,
        pred=pred,
        event_raw=event_raw,
        out_path=qc_dir / "confidence_inspection.png",
        save_svg=args.save_svg,
    )
    _plot_feature_space(
        features=features,
        feature_names=feature_names,
        pred=pred,
        out_path=qc_dir / "feature_space_inspection.png",
        save_svg=args.save_svg,
    )
    _plot_hard_examples(
        event_raw=event_raw,
        pred=pred,
        confidence=confidence,
        fs=fs,
        out_path=qc_dir / "hard_ambiguous_examples.png",
        save_svg=args.save_svg,
    )

    dict_report = {"available": False}
    if args.with_dictionary_match:
        d_path = args.dictionary.resolve()
        if d_path.is_file():
            d1 = _load_dictionary_layer1(d_path)
            dict_report = _plot_dictionary_match_by_class(
                event_raw=event_raw,
                pred=pred,
                d1=d1,
                out_path=qc_dir / "dictionary_match_by_class.png",
                save_svg=args.save_svg,
            )

    report = {
        "recording_stem": stem,
        "manifest": str(manifest_path),
        "inputs": {
            "events_file": str(events_path),
            "features_file": str(features_path),
            "predictions_file": str(pred_path),
        },
        "n_events": int(n_events),
        "event_shape_samples_by_events": [int(event_raw.shape[0]), int(event_raw.shape[1])],
        "feature_shape_events_by_features": [int(features.shape[0]), int(features.shape[1])],
        "summary": summary,
        "gallery": {
            "n_shown_real": int(n_gallery_real),
            "n_shown_pseudo": int(n_gallery_pseudo),
        },
        "dictionary_match": dict_report,
        "outputs": {
            "prediction_summary": str(qc_dir / "prediction_summary.png"),
            "gallery_real_hfo": str(qc_dir / "event_gallery_real_hfo.png"),
            "gallery_pseudo_hfo": str(qc_dir / "event_gallery_pseudo_hfo.png"),
            "class_waveforms": str(qc_dir / "class_waveform_summaries.png"),
            "timeline": str(qc_dir / "detection_timeline.png"),
            "channel_distribution": str(qc_dir / "channel_distribution_by_class.png"),
            "confidence": str(qc_dir / "confidence_inspection.png"),
            "feature_space": str(qc_dir / "feature_space_inspection.png"),
            "hard_examples": str(qc_dir / "hard_ambiguous_examples.png"),
            "dictionary_match_by_class": str(qc_dir / "dictionary_match_by_class.png")
            if args.with_dictionary_match
            else None,
        },
    }
    (qc_dir / "post_rf_qc_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

