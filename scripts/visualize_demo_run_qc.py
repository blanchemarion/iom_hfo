#!/usr/bin/env python3
"""
QC visualization for partial Demo_Run.py outputs (pre-RF stage).

Focus:
  - visually inspect detected candidate events
  - compare event content to pretrained dictionary atoms
  - summarize feature distributions before RF inference

Expected Demo_Run.py artifacts in --output-dir:
  <stem>_events.npz
  <stem>_features.npz
  <stem>_manifest.json
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


def _load_dictionary_layers(path: Path) -> list[np.ndarray]:
    m = loadmat(str(path), struct_as_record=False, squeeze_me=True)
    d = m["Dictionary"]
    layers = d.L2kSVD
    return [np.asarray(layers[i], dtype=np.float64) for i in range(4)]


def _load_detector_input_mat(path: Path) -> tuple[np.ndarray, float]:
    """Load detector-compatible .mat and return (data.data [samples x channels], fs)."""
    m = loadmat(str(path), struct_as_record=False, squeeze_me=True)
    if "data" not in m or "montage" not in m:
        raise ValueError(f"Missing 'data' or 'montage' in {path}")
    d = m["data"]
    mg = m["montage"]
    if not hasattr(d, "data") or not hasattr(mg, "SampleRate"):
        raise ValueError(f"Invalid detector input structure in {path}")
    x = np.asarray(d.data, dtype=np.float64)
    fs = float(np.asarray(mg.SampleRate).squeeze())
    if x.ndim != 2:
        raise ValueError(f"Expected 2D data.data in {path}, got {x.shape}")
    return x, fs


def _extract_input_mat_path(events_npz: np.lib.npyio.NpzFile) -> Path | None:
    """Read input_mat path saved by Demo_Run.py from events npz, if available."""
    if "input_mat" not in events_npz.files:
        return None
    v = np.asarray(events_npz["input_mat"])
    if v.ndim == 0:
        return Path(str(v.item()))
    if v.size == 0:
        return None
    return Path(str(v.reshape(-1)[0]))


def _zscore_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    mu = np.mean(x, axis=1, keepdims=True)
    sd = np.std(x, axis=1, keepdims=True)
    return (x - mu) / np.maximum(sd, eps)


def _sample_indices(n: int, k: int, seed: int) -> np.ndarray:
    if n <= k:
        return np.arange(n, dtype=int)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=k, replace=False))


def _plot_event_gallery(
    event_raw: np.ndarray,
    fs: float,
    out_path: Path,
    n_show: int,
    seed: int,
) -> None:
    n_events = event_raw.shape[1]
    idx = _sample_indices(n_events, n_show, seed)
    rows, cols = 6, 8
    n_panels = rows * cols
    idx = idx[:n_panels]
    t = (np.arange(event_raw.shape[0]) - event_raw.shape[0] // 2) / fs * 1e3

    fig, axes = plt.subplots(rows, cols, figsize=(14, 8), sharex=True, sharey=True)
    axes = axes.ravel()
    for i, ax in enumerate(axes):
        if i >= len(idx):
            ax.axis("off")
            continue
        ev = event_raw[:, idx[i]]
        ax.plot(t, ev, linewidth=0.8)
        ax.axvline(0.0, color="k", alpha=0.2, linewidth=0.8)
        ax.set_title(f"#{idx[i]}", fontsize=8)
    fig.suptitle("Candidate event gallery (raw, 512 samples)")
    fig.supxlabel("Time relative to center (ms)")
    fig.supylabel("Amplitude (uV)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_event_summary(
    event_raw: np.ndarray,
    fs: float,
    out_path: Path,
) -> None:
    t = (np.arange(event_raw.shape[0]) - event_raw.shape[0] // 2) / fs * 1e3
    mu = np.mean(event_raw, axis=1)
    sd = np.std(event_raw, axis=1)
    med = np.median(event_raw, axis=1)
    p10 = np.percentile(event_raw, 10, axis=1)
    p90 = np.percentile(event_raw, 90, axis=1)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.fill_between(t, p10, p90, alpha=0.25, label="P10-P90")
    ax.fill_between(t, mu - sd, mu + sd, alpha=0.2, label="Mean±SD")
    ax.plot(t, mu, linewidth=1.5, label="Mean")
    ax.plot(t, med, linewidth=1.2, label="Median")
    ax.axvline(0.0, color="k", alpha=0.25, linewidth=0.8)
    ax.set_xlabel("Time relative to center (ms)")
    ax.set_ylabel("Amplitude (uV)")
    ax.set_title("Event pool waveform summary")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_event_metadata(
    timestamp: np.ndarray,
    channelinfo: np.ndarray,
    fs: float,
    out_path: Path,
) -> None:
    t_sec = np.asarray(timestamp, dtype=float) / fs
    ch = np.asarray(channelinfo, dtype=int)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].hist(t_sec, bins=min(80, max(10, len(t_sec) // 20)))
    axes[0].set_xlabel("Time in recording (s)")
    axes[0].set_ylabel("Event count")
    axes[0].set_title("Event timestamps")
    axes[0].grid(alpha=0.3)

    bins = np.arange(ch.min(), ch.max() + 2) - 0.5
    axes[1].hist(ch, bins=bins)
    axes[1].set_xlabel("Channel index (0-based)")
    axes[1].set_ylabel("Event count")
    axes[1].set_title("Event channel distribution")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_feature_heatmap(
    features: np.ndarray,
    feature_names: list[str],
    out_path: Path,
    max_events: int,
) -> None:
    n = features.shape[0]
    use = min(n, max_events)
    x = features[:use, :]
    xz = _zscore_rows(x.T).T

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(xz.T, aspect="auto", origin="lower", cmap="coolwarm", vmin=-3, vmax=3)
    fig.colorbar(im, ax=ax, label="Z-score across events")
    ax.set_xlabel("Event index")
    ax.set_ylabel("Feature")
    ax.set_yticks(np.arange(len(feature_names)))
    ax.set_yticklabels(feature_names)
    ax.set_title("Pre-RF feature heatmap")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_dictionary_atoms(
    layers: list[np.ndarray],
    out_path: Path,
    atoms_per_layer: int = 24,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()
    for li, d in enumerate(layers):
        ax = axes[li]
        n_atoms = d.shape[1]
        k = min(atoms_per_layer, n_atoms)
        show = d[:, :k]
        # normalize each atom for display
        show = show / np.maximum(np.linalg.norm(show, axis=0, keepdims=True), 1e-12)
        im = ax.imshow(show.T, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-0.35, vmax=0.35)
        ax.set_title(f"Layer {li+1}: {d.shape[0]}x{d.shape[1]} (first {k} atoms)")
        ax.set_xlabel("Sample")
        ax.set_ylabel("Atom")
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _sliding_match_scores(
    events: np.ndarray,
    atoms: np.ndarray,
    hop: int = 4,
    valid_atom_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Approximate template matching used by sparse coding:
    for each event and each atom, compute max absolute normalized dot product
    over sliding windows (length = atom length, step = hop).
    """
    n_samples, n_events = events.shape
    atom_len, n_atoms = atoms.shape
    if atom_len > n_samples:
        raise ValueError("Atom length exceeds event length.")

    atom_norm = atoms / np.maximum(np.linalg.norm(atoms, axis=0, keepdims=True), 1e-12)
    if valid_atom_mask is None:
        valid_atom_mask = np.ones(n_atoms, dtype=bool)
    else:
        valid_atom_mask = np.asarray(valid_atom_mask, dtype=bool).reshape(-1)
        if valid_atom_mask.size != n_atoms:
            raise ValueError("valid_atom_mask length must match number of atoms.")
    valid_idx = np.where(valid_atom_mask)[0]
    if valid_idx.size == 0:
        raise ValueError("No valid atoms available for matching after masking.")

    starts = np.arange(0, n_samples - atom_len + 1, hop, dtype=int)
    scores = np.zeros((n_events, n_atoms), dtype=np.float64)
    best_start = np.zeros(n_events, dtype=int)
    best_signed = np.zeros(n_events, dtype=np.float64)

    for ei in range(n_events):
        ev = events[:, ei]
        win = np.stack([ev[s : s + atom_len] for s in starts], axis=0)  # [nw, atom_len]
        wn = win / np.maximum(np.linalg.norm(win, axis=1, keepdims=True), 1e-12)
        # [nw, n_atoms], then max abs over windows
        corr = wn @ atom_norm
        scores[ei, :] = np.max(np.abs(corr), axis=0)
        abs_corr_valid = np.abs(corr[:, valid_idx])
        j, k_local = np.unravel_index(np.argmax(abs_corr_valid), abs_corr_valid.shape)
        k = int(valid_idx[k_local])
        best_start[ei] = int(starts[j])
        best_signed[ei] = float(corr[j, k])

    masked_scores = scores.copy()
    masked_scores[:, ~valid_atom_mask] = -np.inf
    best_idx = np.argmax(masked_scores, axis=1)
    best_val = scores[np.arange(n_events), best_idx]
    return best_idx, best_val, best_start, best_signed


def _plot_template_matching_qc(
    event_raw: np.ndarray,
    d1: np.ndarray,
    out_dir: Path,
    fs: float,
    max_events_match: int,
    seed: int,
    near_constant_std_threshold: float,
) -> tuple[dict, dict[str, np.ndarray]]:
    idx = _sample_indices(event_raw.shape[1], max_events_match, seed)
    ev = event_raw[:, idx]

    atom_std = np.std(d1, axis=0)
    valid_atoms = atom_std > near_constant_std_threshold
    if not np.any(valid_atoms):
        raise ValueError(
            "All layer-1 atoms considered near-constant; lower --near-constant-std-threshold."
        )

    best_atom_idx, best_score, best_start, best_signed = _sliding_match_scores(
        ev,
        d1,
        hop=4,
        valid_atom_mask=valid_atoms,
    )

    # Histogram of best match score
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(best_score, bins=30)
    ax.set_xlabel("Best |normalized correlation| to Layer-1 atoms (excluding near-constant)")
    ax.set_ylabel("Count")
    ax.set_title("Template-matching strength (near-constant atoms excluded)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "template_match_score_hist.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Atom usage bar chart
    counts = np.bincount(best_atom_idx, minlength=d1.shape[1])
    top = np.argsort(counts)[::-1][:20]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(np.arange(len(top)), counts[top])
    ax.set_xticks(np.arange(len(top)))
    ax.set_xticklabels([str(i) for i in top], rotation=45)
    ax.set_xlabel("Layer-1 atom index")
    ax.set_ylabel("Best-match count")
    ax.set_title("Most frequently matched Layer-1 atoms (near-constant excluded)")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_dir / "template_match_atom_usage.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Example overlays: best-matched event segment vs best atom
    atom_len = d1.shape[0]
    t_ms = np.arange(atom_len) / fs * 1e3
    show = min(12, len(idx))
    fig, axes = plt.subplots(3, 4, figsize=(12, 7), sharex=True, sharey=True)
    axes = axes.ravel()
    for i in range(show):
        ax = axes[i]
        e = ev[:, i]
        st = int(best_start[i])
        crop = e[st : st + atom_len]
        a = d1[:, best_atom_idx[i]] * (1.0 if best_signed[i] >= 0 else -1.0)
        crop_n = crop / max(np.linalg.norm(crop), 1e-12)
        a_n = a / max(np.linalg.norm(a), 1e-12)
        ax.plot(t_ms, crop_n, label="matched event segment", linewidth=1.0)
        ax.plot(t_ms, a_n, label="best atom", linewidth=1.0)
        ax.set_title(
            f"ev#{idx[i]} a{best_atom_idx[i]} st={st} s={best_score[i]:.2f}",
            fontsize=8,
        )
        ax.grid(alpha=0.3)
    for j in range(show, len(axes)):
        axes[j].axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.supxlabel("Time in matched 128-sample window (ms)")
    fig.supylabel("L2-normalized amplitude")
    fig.suptitle("Event-to-atom overlays (Layer-1 dictionary, near-constant excluded)")
    fig.tight_layout()
    fig.savefig(out_dir / "template_match_event_atom_overlays.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    report = {
        "n_events_scored": int(len(idx)),
        "near_constant_std_threshold": float(near_constant_std_threshold),
        "n_atoms_total": int(d1.shape[1]),
        "n_atoms_excluded_near_constant": int(np.sum(~valid_atoms)),
        "excluded_atom_indices": [int(i) for i in np.where(~valid_atoms)[0]],
        "n_atoms_used_for_matching": int(np.sum(valid_atoms)),
        "best_score_mean": float(np.mean(best_score)),
        "best_score_median": float(np.median(best_score)),
        "best_score_p90": float(np.percentile(best_score, 90)),
    }
    context = {
        "sampled_event_indices": idx.astype(int),
        "best_atom_index": best_atom_idx.astype(int),
        "best_score": best_score.astype(np.float64),
        "best_start": best_start.astype(int),
        "best_signed_corr": best_signed.astype(np.float64),
    }
    return report, context


def _plot_raw_event_template_qc(
    raw_data: np.ndarray,
    fs_raw: float,
    timestamp: np.ndarray,
    channelinfo: np.ndarray,
    event_raw: np.ndarray,
    d1: np.ndarray,
    tm_context: dict[str, np.ndarray],
    out_path: Path,
    n_show: int,
    seed: int,
) -> None:
    """
    Show, for sampled detections:
      (left) raw continuous trace around timestamp with detected window highlighted
      (right) matched event segment against matched dictionary template
    """
    sampled_ev = tm_context["sampled_event_indices"]
    if sampled_ev.size == 0:
        return
    pick_local = _sample_indices(sampled_ev.size, n_show, seed + 101)
    show_events = sampled_ev[pick_local]
    atom_idx = tm_context["best_atom_index"][pick_local]
    score = tm_context["best_score"][pick_local]
    start = tm_context["best_start"][pick_local]
    signed = tm_context["best_signed_corr"][pick_local]

    rows = len(show_events)
    fig, axes = plt.subplots(rows, 2, figsize=(13, max(2.8 * rows, 6)))
    if rows == 1:
        axes = np.asarray([axes])

    atom_len = d1.shape[0]
    event_len = event_raw.shape[0]
    half_event = event_len // 2
    pad = int(round(0.3 * fs_raw))

    for r in range(rows):
        ev_i = int(show_events[r])
        ch = int(channelinfo[ev_i])
        ts = int(round(timestamp[ev_i]))

        ax_l = axes[r, 0]
        lo = max(0, ts - pad)
        hi = min(raw_data.shape[0], ts + pad)
        t_ms = (np.arange(lo, hi) - ts) / fs_raw * 1e3
        y = raw_data[lo:hi, ch]
        ax_l.plot(t_ms, y, linewidth=0.9)
        # Highlight the exact 512-sample detected event window on raw data
        ev_lo = (ts - (half_event - 1) - ts) / fs_raw * 1e3
        ev_hi = (ts + half_event - ts) / fs_raw * 1e3
        ax_l.axvspan(ev_lo, ev_hi, color="orange", alpha=0.2, label="detected event window")
        ax_l.axvline(0.0, color="r", alpha=0.35, linewidth=1.0, label="event timestamp")
        ax_l.set_title(f"raw ch={ch} ev#{ev_i}")
        ax_l.set_xlabel("Time around detection (ms)")
        ax_l.set_ylabel("Amplitude (uV)")
        ax_l.grid(alpha=0.3)
        if r == 0:
            ax_l.legend(loc="upper right", fontsize=8)

        ax_r = axes[r, 1]
        e = event_raw[:, ev_i]
        st = int(start[r])
        seg = e[st : st + atom_len]
        tpl = d1[:, int(atom_idx[r])] * (1.0 if signed[r] >= 0 else -1.0)
        seg_n = seg / max(np.linalg.norm(seg), 1e-12)
        tpl_n = tpl / max(np.linalg.norm(tpl), 1e-12)
        full_t = (np.arange(event_len) - event_len // 2) / fs_raw * 1e3
        seg_t = (np.arange(atom_len) + st - event_len // 2) / fs_raw * 1e3
        full_n = e / max(np.max(np.abs(e)), 1e-12)

        ax_r.plot(full_t, 0.35 * full_n, color="0.75", linewidth=0.9, label="full detected event")
        ax_r.plot(seg_t, seg_n, linewidth=1.1, label="matched event segment")
        ax_r.plot(seg_t, tpl_n, linewidth=1.1, label="matched template")
        ax_r.axvline(0.0, color="k", alpha=0.2, linewidth=0.8)
        ax_r.set_title(f"a{int(atom_idx[r])} st={st} score={float(score[r]):.2f}")
        ax_r.set_xlabel("Time within detected event (ms)")
        ax_r.set_ylabel("L2-normalized amplitude")
        ax_r.grid(alpha=0.3)
        if r == 0:
            ax_r.legend(loc="upper right", fontsize=8)

    fig.suptitle("Raw detection and matched template (Layer-1)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="QC visualization for Demo_Run.py outputs (events/features/dictionary matching)."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_repo_root() / "output" / "demo_run_py",
        help="Directory containing <stem>_events.npz and <stem>_features.npz",
    )
    parser.add_argument(
        "--recording-stem",
        type=str,
        default=None,
        help="Recording stem, e.g., sub-01_ses-SITUATION2A_task-acute",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Path to a specific <stem>_manifest.json (overrides --recording-stem)",
    )
    parser.add_argument(
        "--dictionary",
        type=Path,
        default=_repo_root() / "Data" / "Dictionary_CRDL_ASLR.mat",
        help="Pretrained dictionary .mat",
    )
    parser.add_argument(
        "--qc-dir",
        type=Path,
        default=None,
        help="Output QC directory (default: <output-dir>/<stem>_qc)",
    )
    parser.add_argument("--max-event-gallery", type=int, default=48)
    parser.add_argument("--max-feature-events", type=int, default=400)
    parser.add_argument("--max-events-match", type=int, default=250)
    parser.add_argument("--max-raw-template-examples", type=int, default=8)
    parser.add_argument(
        "--near-constant-std-threshold",
        type=float,
        default=1e-6,
        help="Layer-1 atom std threshold below which atoms are excluded from matching.",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    manifest_path = args.manifest.resolve() if args.manifest else None
    stem, guessed_manifest = _resolve_stem(output_dir, args.recording_stem, manifest_path)
    if manifest_path is None:
        manifest_path = guessed_manifest

    events_npz = output_dir / f"{stem}_events.npz"
    features_npz = output_dir / f"{stem}_features.npz"
    if not events_npz.is_file():
        raise FileNotFoundError(f"Missing events file: {events_npz}")
    if not features_npz.is_file():
        raise FileNotFoundError(f"Missing features file: {features_npz}")

    qc_dir = (args.qc_dir.resolve() if args.qc_dir else output_dir / f"{stem}_qc")
    qc_dir.mkdir(parents=True, exist_ok=True)

    ev = np.load(events_npz, allow_pickle=True)
    ft = np.load(features_npz, allow_pickle=True)
    event_raw = np.asarray(ev["event_raw"], dtype=np.float64)
    timestamp = np.asarray(ev["timestamp"], dtype=np.float64)
    channelinfo = np.asarray(ev["channelinformation"], dtype=int)
    fs = float(np.asarray(ev["fs"]).squeeze())
    input_mat_path = _extract_input_mat_path(ev)

    features = np.asarray(ft["features"], dtype=np.float64)
    feature_names = [str(x) for x in ft["feature_names"].tolist()]

    if event_raw.ndim != 2:
        raise ValueError(f"event_raw should be 2D [samples x events], got {event_raw.shape}")
    if event_raw.shape[1] == 0:
        raise ValueError("No events in event_raw; run Demo_Run.py on a file with detected events.")

    dict_layers = _load_dictionary_layers(args.dictionary.resolve())

    _plot_event_gallery(
        event_raw=event_raw,
        fs=fs,
        out_path=qc_dir / "event_gallery.png",
        n_show=args.max_event_gallery,
        seed=args.seed,
    )
    _plot_event_summary(
        event_raw=event_raw,
        fs=fs,
        out_path=qc_dir / "event_waveform_summary.png",
    )
    _plot_event_metadata(
        timestamp=timestamp,
        channelinfo=channelinfo,
        fs=fs,
        out_path=qc_dir / "event_metadata_histograms.png",
    )
    _plot_feature_heatmap(
        features=features,
        feature_names=feature_names,
        out_path=qc_dir / "feature_heatmap.png",
        max_events=args.max_feature_events,
    )
    _plot_dictionary_atoms(
        layers=dict_layers,
        out_path=qc_dir / "dictionary_atoms_overview.png",
    )
    tm, tm_ctx = _plot_template_matching_qc(
        event_raw=event_raw,
        d1=dict_layers[0],
        out_dir=qc_dir,
        fs=fs,
        max_events_match=args.max_events_match,
        seed=args.seed,
        near_constant_std_threshold=args.near_constant_std_threshold,
    )
    raw_template_fig = None
    if input_mat_path is not None and input_mat_path.is_file():
        raw_data, fs_raw = _load_detector_input_mat(input_mat_path)
        _plot_raw_event_template_qc(
            raw_data=raw_data,
            fs_raw=fs_raw,
            timestamp=timestamp,
            channelinfo=channelinfo,
            event_raw=event_raw,
            d1=dict_layers[0],
            tm_context=tm_ctx,
            out_path=qc_dir / "raw_detection_template_examples.png",
            n_show=args.max_raw_template_examples,
            seed=args.seed,
        )
        raw_template_fig = str(qc_dir / "raw_detection_template_examples.png")

    report = {
        "recording_stem": stem,
        "manifest": str(manifest_path),
        "events_file": str(events_npz),
        "features_file": str(features_npz),
        "n_events": int(event_raw.shape[1]),
        "event_len_samples": int(event_raw.shape[0]),
        "n_features": int(features.shape[1]),
        "fs_hz": fs,
        "dictionary_shapes": [list(d.shape) for d in dict_layers],
        "template_matching_layer1": tm,
        "outputs": {
            "event_gallery": str(qc_dir / "event_gallery.png"),
            "event_summary": str(qc_dir / "event_waveform_summary.png"),
            "event_metadata": str(qc_dir / "event_metadata_histograms.png"),
            "feature_heatmap": str(qc_dir / "feature_heatmap.png"),
            "dictionary_overview": str(qc_dir / "dictionary_atoms_overview.png"),
            "match_score_hist": str(qc_dir / "template_match_score_hist.png"),
            "match_atom_usage": str(qc_dir / "template_match_atom_usage.png"),
            "match_overlays": str(qc_dir / "template_match_event_atom_overlays.png"),
            "raw_detection_template_examples": raw_template_fig,
        },
    }
    (qc_dir / "qc_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

