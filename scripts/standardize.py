#!/usr/bin/env python3
"""
Standardize BIDS iEEG (ds004944-style) EDF recordings to MATLAB .mat files for the
IOM-HFO detector (bipolar + .mat inference path).

The MATLAB function HFO_Initial_Detector_DemoVersion.m loads the file and expects:
  - data.data          : [samples x channels], double recommended, units microvolts
  - montage.SampleRate : sampling rate in Hz (scalar)

No other fields are required by the detector; optional metadata can be added later.

Dependencies (install at least one EDF backend):
  pip install numpy scipy pandas mne
  # or
  pip install numpy scipy pandas pyedflib

Usage:
  python scripts/standardize_ds004944_for_hfo.py --dataset-root ds004944 --output-dir standardized_hfo_inputs

If every run fails with "git-annex or Git LFS pointer", the large *.edf files were never
downloaded—only metadata/pointers are in the clone. Fetch binaries, then re-run, for example:
  - git annex get .          (from inside the dataset repo, if git-annex is enabled)
  - datalad install / datalad get <path>   (if you use DataLad)
  - OpenNeuro: download the dataset via the website or openneuro CLI so *.edf are real files
  - Or clone without annex: use a release .zip / S3 download per OpenNeuro docs
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

import numpy as np
import pandas as pd
from scipy.io import savemat

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXPECTED_FS_HZ = 2000.0
EDF_GLOB = "*_task-acute_ieeg.edf"
MICROVOLT_P99_HARD_LIMIT = 1e5
MICROVOLT_MAX_HARD_LIMIT = 1e6
MICROVOLT_AUTOSCALE_FACTOR = 1e-6


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ConversionRecord:
    """One row for the manifest and logging."""

    subject: str
    session: str
    input_edf: str
    output_mat: str
    n_samples: int
    n_channels: int
    sample_rate: float
    channel_names: list[str]
    unit_notes: str
    conversion_applied: str
    status: Literal["ok", "failed"]
    error_message: str = ""


@dataclass
class EDFReadResult:
    """Raw reader output before enforcing channels.tsv order."""

    data: np.ndarray  # shape [samples, channels], physical units as returned by reader
    sample_rate_hz: float
    channel_names_edf: list[str]
    reader_backend: str
    physical_units_per_channel: list[str] | None = None  # pyedflib only


# ---------------------------------------------------------------------------
# BIDS / filesystem helpers
# ---------------------------------------------------------------------------


def find_task_acute_edf_files(dataset_root: Path) -> list[Path]:
    """Recursively find all BIDS continuous iEEG EDF files for task-acute."""
    return sorted(dataset_root.rglob(EDF_GLOB))


def parse_subject_session_from_path(edf_path: Path) -> tuple[str, str]:
    """Extract BIDS subject and session folder names from a path under dataset_root."""
    parts = edf_path.parts
    subj = next((p for p in parts if p.startswith("sub-")), "unknown")
    sess = next((p for p in parts if p.startswith("ses-")), "unknown")
    return subj, sess


def sidecar_channels_tsv(edf_path: Path) -> Path:
    """Matching *_channels.tsv in the same ieeg/ folder."""
    stem = edf_path.name.replace("_ieeg.edf", "")
    return edf_path.parent / f"{stem}_channels.tsv"


def sidecar_ieeg_json(edf_path: Path) -> Path:
    stem = edf_path.name.replace("_ieeg.edf", "")
    return edf_path.parent / f"{stem}_ieeg.json"


def output_mat_path(edf_path: Path, output_dir: Path) -> Path:
    """sub-XX_ses-SITUATIONXA_task-acute.mat"""
    stem = edf_path.name.replace("_ieeg.edf", "")
    return output_dir / f"{stem}.mat"


# ---------------------------------------------------------------------------
# channels.tsv + JSON
# ---------------------------------------------------------------------------


def read_channels_tsv(path: Path) -> pd.DataFrame:
    """Read BIDS channels.tsv (tab-separated)."""
    df = pd.read_csv(path, sep="\t", dtype=str)
    required = {"name", "type", "units"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"channels.tsv missing columns {missing}: {path}")
    return df


def read_ieeg_json_optional(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_unit_string(u: str) -> str:
    """Normalize µ / uV variants for comparison."""
    u = u.strip().lower().replace("μ", "u")
    u = re.sub(r"\s+", "", u)
    return u


def classify_units_column(units: str) -> Literal["uV", "V", "mV", "unknown"]:
    n = normalize_unit_string(units)
    if n in ("uv", "microvolt", "microvolts"):
        return "uV"
    if n in ("v", "volt", "volts"):
        return "V"
    if n in ("mv", "millivolt", "millivolts"):
        return "mV"
    return "unknown"


# ---------------------------------------------------------------------------
# Placeholder / incomplete downloads (git-annex, Git LFS)
# ---------------------------------------------------------------------------


def describe_edf_placeholder(edf_path: Path) -> str | None:
    """
    If the path is a tiny pointer file instead of a real EDF, return a human hint.
    Otherwise return None. Does not parse valid EDF headers.
    """
    try:
        size = edf_path.stat().st_size
    except OSError:
        return None
    if size > 8192:
        return None
    try:
        snippet = edf_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    low = snippet.lower()
    if "git-annex" in low or "sha256e-" in low or "git-lfs" in low or "filter: git-lfs" in low:
        return (
            "This path looks like a git-annex or Git LFS pointer, not a binary EDF. "
            "Fetch the real files (e.g. git annex get <file>, git lfs pull) and re-run."
        )
    return None


def is_annex_or_lfs_pointer_failure(message: str) -> bool:
    """True if the error indicates the EDF path was a pointer, not local binary data."""
    m = message.lower()
    return (
        "git-annex" in m
        or "git lfs" in m
        or "lfs pointer" in m
        or "not a binary edf" in m
    )


# ---------------------------------------------------------------------------
# EDF reading: MNE (preferred) or pyedflib
# ---------------------------------------------------------------------------


def _read_edf_mne(edf_path: Path) -> EDFReadResult:
    import mne

    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="ERROR")
    # MNE stores data in Volts; shape (n_channels, n_samples)
    data_v = raw.get_data()
    ch_names = [c.strip() for c in raw.ch_names]
    fs = float(raw.info["sfreq"])
    data_sxc = np.ascontiguousarray(data_v.T)
    return EDFReadResult(
        data=data_sxc,
        sample_rate_hz=fs,
        channel_names_edf=ch_names,
        reader_backend="mne",
        physical_units_per_channel=["V"] * len(ch_names),
    )


def _read_edf_pyedflib(edf_path: Path) -> EDFReadResult:
    import pyedflib

    reader = pyedflib.EdfReader(str(edf_path))
    try:
        n = reader.signals_in_file
        labels = [reader.getLabel(i).strip() for i in range(n)]
        fs0 = reader.getSampleFrequency(0)
        if n > 1:
            for i in range(1, n):
                fi = reader.getSampleFrequency(i)
                if abs(fi - fs0) > 1e-6:
                    logging.warning(
                        "Per-channel sample rates differ in %s (ch0=%s, ch%s=%s); using ch0.",
                        edf_path,
                        fs0,
                        i,
                        fi,
                    )
        lengths = [reader.getNSamples()[i] for i in range(n)]
        if len(set(lengths)) != 1:
            raise ValueError(f"Inconsistent signal lengths in EDF: {lengths}")

        signals = [reader.readSignal(i) for i in range(n)]
        units = [reader.getPhysicalDimension(i).strip() for i in range(n)]
        data_sxc = np.ascontiguousarray(np.stack(signals, axis=1))
    finally:
        reader.close()

    return EDFReadResult(
        data=data_sxc,
        sample_rate_hz=float(fs0),
        channel_names_edf=labels,
        reader_backend="pyedflib",
        physical_units_per_channel=units,
    )


def read_edf_auto(edf_path: Path, backend: str) -> EDFReadResult:
    """Load EDF using mne, pyedflib, or auto."""
    if backend == "mne":
        return _read_edf_mne(edf_path)
    if backend == "pyedflib":
        return _read_edf_pyedflib(edf_path)
    if backend == "auto":
        mne_err: str | None = None
        try:
            return _read_edf_mne(edf_path)
        except ImportError:
            pass  # mne not installed — try pyedflib
        except Exception as e:  # noqa: BLE001
            mne_err = f"mne: {type(e).__name__}: {e}"
            logging.info("MNE failed on %s; trying pyedflib.", edf_path.name)
        try:
            return _read_edf_pyedflib(edf_path)
        except ImportError as e:
            install_msg = (
                "Install at least one EDF reader: pip install mne  OR  pip install pyedflib"
            )
            if mne_err:
                raise RuntimeError(
                    "mne failed to read this file and pyedflib is not available.\n  "
                    + mne_err
                    + f"\n  pyedflib import: {e}\n"
                    + install_msg
                ) from e
            raise ImportError(
                "Neither mne nor pyedflib could be imported.\n" + install_msg
            ) from e
        except Exception as e:  # noqa: BLE001
            py_err = f"pyedflib: {type(e).__name__}: {e}"
            if mne_err:
                raise RuntimeError(
                    "Could not read EDF with either backend.\n  "
                    + mne_err
                    + "\n  "
                    + py_err
                ) from e
            raise
    raise ValueError(f"Unknown backend: {backend}")


# ---------------------------------------------------------------------------
# Unit conversion to microvolts (saved to disk)
# ---------------------------------------------------------------------------


def convert_reader_output_to_microvolts(
    result: EDFReadResult,
    tsv_units_per_channel: list[str],
) -> tuple[np.ndarray, str]:
    """
    Return data in microvolts and a short description of what was applied.

    MNE: always Volts -> multiply by 1e6.

    pyedflib: uses EDF physical dimension per channel; map V, mV, uV accordingly.
    If TSV declares uV for all channels, trust consistency check against magnitude after conversion.
    """
    x = result.data.astype(np.float64, copy=True)
    backend = result.reader_backend

    if backend == "mne":
        x *= 1e6
        return x, "MNE_get_data_is_Volts_multiplied_by_1e6_to_uV"

    # pyedflib: physical values in stated dimension
    assert result.physical_units_per_channel is not None
    scales = []
    notes = []
    for i, dim in enumerate(result.physical_units_per_channel):
        cls = classify_units_column(dim)
        if cls == "uV":
            scales.append(1.0)
            notes.append("uV")
        elif cls == "V":
            scales.append(1e6)
            notes.append("V_to_uV")
        elif cls == "mV":
            scales.append(1e3)
            notes.append("mV_to_uV")
        else:
            # Fall back to TSV for this channel index if possible
            if i < len(tsv_units_per_channel):
                tcls = classify_units_column(tsv_units_per_channel[i])
                if tcls == "uV":
                    scales.append(1.0)
                    notes.append("unknown_dim_assumed_uV_from_tsv")
                elif tcls == "V":
                    scales.append(1e6)
                    notes.append("unknown_dim_used_V_from_tsv")
                elif tcls == "mV":
                    scales.append(1e3)
                    notes.append("unknown_dim_used_mV_from_tsv")
                else:
                    raise ValueError(
                        f"Cannot infer physical unit for channel {i} (EDF dim={dim!r}, TSV units={tsv_units_per_channel[i]!r})"
                    )
            else:
                raise ValueError(f"Cannot infer physical unit for channel {i} (EDF dim={dim!r})")

    for i, s in enumerate(scales):
        x[:, i] *= s
    summary = "pyedflib_physical_scale_per_channel:" + ",".join(notes)
    return x, summary


def enforce_microvolt_sanity(
    data_uv: np.ndarray,
    *,
    p99_limit_uv: float = MICROVOLT_P99_HARD_LIMIT,
    max_limit_uv: float = MICROVOLT_MAX_HARD_LIMIT,
    autoscale_factor: float = MICROVOLT_AUTOSCALE_FACTOR,
) -> tuple[np.ndarray, str]:
    """
    Strict sanity-check for detector-ready microvolt signals.

    If values look 1e6 too large, auto-rescale by 1e-6 and re-check.
    Raises ValueError when data still violates the hard limits after correction.
    """
    x = np.asarray(data_uv, dtype=np.float64, order="C")
    abs_x = np.abs(x)
    p99 = float(np.percentile(abs_x, 99))
    mx = float(np.max(abs_x))

    note = f"uv_sanity:p99={p99:.6g},max={mx:.6g}"
    if p99 <= p99_limit_uv and mx <= max_limit_uv:
        return x, note + ",action=none"

    # Values are implausibly large for microvolts; common cause is a second accidental 1e6 gain.
    x_scaled = x * autoscale_factor
    abs_scaled = np.abs(x_scaled)
    p99_scaled = float(np.percentile(abs_scaled, 99))
    mx_scaled = float(np.max(abs_scaled))

    if p99_scaled <= p99_limit_uv and mx_scaled <= max_limit_uv:
        logging.warning(
            "Auto-rescaled by %s due to microvolt sanity check: p99 %.6g -> %.6g, max %.6g -> %.6g",
            autoscale_factor,
            p99,
            p99_scaled,
            mx,
            mx_scaled,
        )
        return (
            x_scaled,
            note
            + f",action=auto_rescaled_{autoscale_factor:g},"
            + f"p99_after={p99_scaled:.6g},max_after={mx_scaled:.6g}",
        )

    raise ValueError(
        "Microvolt sanity-check failed after attempted auto-rescale "
        f"(p99={p99:.6g}, max={mx:.6g}, p99_after={p99_scaled:.6g}, max_after={mx_scaled:.6g})."
    )


# ---------------------------------------------------------------------------
# Align columns to channels.tsv order
# ---------------------------------------------------------------------------


def normalize_channel_key(name: str) -> str:
    return name.strip()


def align_channels(
    data_sxc: np.ndarray,
    edf_names: list[str],
    tsv_names: list[str],
) -> np.ndarray:
    """
    Reorder columns to match tsv_names order. Raises if any name is missing.
    """
    edf_map: dict[str, int] = {}
    for i, n in enumerate(edf_names):
        key = normalize_channel_key(n)
        if key in edf_map:
            logging.warning(
                "Duplicate EDF channel label %r at indices %s and %s; using first occurrence.",
                key,
                edf_map[key],
                i,
            )
        else:
            edf_map[key] = i

    cols: list[int] = []
    missing: list[str] = []
    for name in tsv_names:
        key = normalize_channel_key(name)
        if key not in edf_map:
            missing.append(name)
        else:
            cols.append(edf_map[key])

    if missing:
        raise ValueError(
            "Channels listed in channels.tsv not found in EDF labels: " + ", ".join(missing)
        )

    unused_idx = set(range(len(edf_names))) - set(cols)
    if unused_idx:
        ignored = [edf_names[i] for i in sorted(unused_idx)]
        logging.warning(
            "EDF contains %d channel(s) not listed in channels.tsv (ignored): %s",
            len(ignored),
            ignored[:30],
        )

    return np.ascontiguousarray(data_sxc[:, cols])


# ---------------------------------------------------------------------------
# Validation warnings
# ---------------------------------------------------------------------------


def validate_channels_dataframe(channels_df: pd.DataFrame, edf_path: Path) -> list[str]:
    """Return list of warning strings."""
    warns: list[str] = []
    for _, row in channels_df.iterrows():
        t = str(row["type"]).strip().upper()
        if t and t != "ECOG":
            warns.append(f"Non-ECOG channel type {row['type']!r} for channel {row['name']!r} in {edf_path.name}")

        u = classify_units_column(str(row["units"]))
        if u == "unknown":
            warns.append(
                f"Unrecognized units {row['units']!r} for channel {row['name']!r} in {edf_path.name}"
            )

    # sampling_frequency column (BIDS) optional consistency
    if "sampling_frequency" in channels_df.columns:
        freqs = channels_df["sampling_frequency"].astype(str).unique().tolist()
        for f in freqs:
            try:
                fv = float(f)
                if abs(fv - EXPECTED_FS_HZ) > 0.5:
                    warns.append(
                        f"channels.tsv sampling_frequency={fv} Hz differs from expected {EXPECTED_FS_HZ} ({edf_path.name})"
                    )
            except ValueError:
                warns.append(f"Non-numeric sampling_frequency {f!r} in channels.tsv ({edf_path.name})")
    return warns


def compare_fs_json_edf_tsv(
    fs_edf: float,
    channels_df: pd.DataFrame,
    ieeg_json: dict[str, Any] | None,
    edf_path: Path,
) -> list[str]:
    warns: list[str] = []
    if abs(fs_edf - EXPECTED_FS_HZ) > 0.5:
        warns.append(
            f"EDF-reported fs={fs_edf} Hz != expected {EXPECTED_FS_HZ} Hz ({edf_path.name}); detector assumes 2000 Hz with resampleState=No."
        )

    if "sampling_frequency" in channels_df.columns:
        try:
            fs_tsv = float(str(channels_df["sampling_frequency"].iloc[0]))
            if abs(fs_tsv - fs_edf) > 0.5:
                warns.append(
                    f"channels.tsv fs={fs_tsv} vs EDF fs={fs_edf} ({edf_path.name})"
                )
        except (ValueError, IndexError):
            pass

    if ieeg_json and "SamplingFrequency" in ieeg_json:
        fs_j = float(ieeg_json["SamplingFrequency"])
        if abs(fs_j - fs_edf) > 0.5:
            warns.append(
                f"ieeg.json SamplingFrequency={fs_j} vs EDF fs={fs_edf} ({edf_path.name})"
            )
    return warns


# ---------------------------------------------------------------------------
# MATLAB save (nested structs)
# ---------------------------------------------------------------------------


def verify_saved_mat_file(
    mat_path: Path,
    expected_shape: tuple[int, int],
    expected_fs: float,
) -> None:
    """
    Reload the .mat with scipy and assert nested structs match detector expectations.
    Raises ValueError on mismatch (caller may catch and downgrade to warning).
    """
    from scipy.io import loadmat

    m = loadmat(str(mat_path), struct_as_record=False, squeeze_me=True)
    if "data" not in m or "montage" not in m:
        raise ValueError(f"Missing top-level variables in {mat_path}: {list(m.keys())}")

    data_block = m["data"]
    if not hasattr(data_block, "data"):
        raise ValueError(f"'data' struct has no .data field in {mat_path}")
    arr = np.asarray(data_block.data)
    if arr.ndim != 2:
        raise ValueError(f"data.data must be 2D, got shape {arr.shape}")
    if arr.shape != expected_shape:
        raise ValueError(f"data.data shape {arr.shape} != expected {expected_shape}")

    montage_block = m["montage"]
    if not hasattr(montage_block, "SampleRate"):
        raise ValueError(f"'montage' struct has no .SampleRate in {mat_path}")
    fs = float(np.asarray(montage_block.SampleRate).squeeze())
    if abs(fs - expected_fs) > 1e-6:
        raise ValueError(f"montage.SampleRate={fs} != expected {expected_fs}")


def save_detector_mat(
    out_path: Path,
    data_samples_by_channels: np.ndarray,
    sample_rate_hz: float,
) -> None:
    """
    Write .mat with variables `data` and `montage` compatible with MATLAB load().

    MATLAB after load:
      data.data
      montage.SampleRate
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # float64 for MATLAB default double
    mat = {
        "data": {"data": np.ascontiguousarray(data_samples_by_channels, dtype=np.float64)},
        "montage": {"SampleRate": np.float64(sample_rate_hz)},
    }
    savemat(str(out_path), mat, format="5", do_compression=True)


# ---------------------------------------------------------------------------
# Single-file pipeline
# ---------------------------------------------------------------------------


def process_one_edf(
    edf_path: Path,
    output_dir: Path,
    backend: str,
) -> ConversionRecord:
    subj, sess = parse_subject_session_from_path(edf_path)
    out_mat = output_mat_path(edf_path, output_dir)
    tsv_path = sidecar_channels_tsv(edf_path)
    json_path = sidecar_ieeg_json(edf_path)

    if not tsv_path.is_file():
        return ConversionRecord(
            subject=subj,
            session=sess,
            input_edf=str(edf_path.resolve()),
            output_mat=str(out_mat.resolve()),
            n_samples=0,
            n_channels=0,
            sample_rate=0.0,
            channel_names=[],
            unit_notes="",
            conversion_applied="",
            status="failed",
            error_message=f"Missing channels.tsv: {tsv_path}",
        )

    try:
        channels_df = read_channels_tsv(tsv_path)
        tsv_names = [str(n) for n in channels_df["name"].tolist()]
        tsv_units = [str(u) for u in channels_df["units"].tolist()]

        for w in validate_channels_dataframe(channels_df, edf_path):
            logging.warning(w)

        placeholder_hint = describe_edf_placeholder(edf_path)
        if placeholder_hint:
            raise ValueError(placeholder_hint)

        ieeg_json = read_ieeg_json_optional(json_path)

        result = read_edf_auto(edf_path, backend=backend)

        if result.data.ndim != 2:
            raise ValueError(f"Expected 2D array from reader, got shape {result.data.shape}")

        n_samp, n_ch = result.data.shape
        if n_ch != len(result.channel_names_edf):
            raise ValueError("Channel name count does not match data width.")

        data_uv, conv_note = convert_reader_output_to_microvolts(result, tsv_units)

        data_uv = align_channels(data_uv, result.channel_names_edf, tsv_names)
        data_uv, sanity_note = enforce_microvolt_sanity(data_uv)
        conv_note = f"{conv_note}|{sanity_note}"

        for w in compare_fs_json_edf_tsv(result.sample_rate_hz, channels_df, ieeg_json, edf_path):
            logging.warning(w)

        # Final shape check
        assert data_uv.shape[0] == n_samp
        assert data_uv.shape[1] == len(tsv_names)

        save_detector_mat(out_mat, data_uv, float(result.sample_rate_hz))
        verify_saved_mat_file(out_mat, data_uv.shape, float(result.sample_rate_hz))

        return ConversionRecord(
            subject=subj,
            session=sess,
            input_edf=str(edf_path.resolve()),
            output_mat=str(out_mat.resolve()),
            n_samples=int(data_uv.shape[0]),
            n_channels=int(data_uv.shape[1]),
            sample_rate=float(result.sample_rate_hz),
            channel_names=tsv_names,
            unit_notes="BIDS channels.tsv units: " + ";".join(tsv_units),
            conversion_applied=conv_note,
            status="ok",
            error_message="",
        )
    except Exception as exc:  # noqa: BLE001 — top-level per-file catch
        err_text = f"{type(exc).__name__}: {exc}"
        if is_annex_or_lfs_pointer_failure(err_text):
            # Avoid 40+ identical ERROR lines when the whole dataset is still annex pointers.
            logging.debug("Skipped (pointer file, not local EDF): %s", edf_path)
        elif logging.getLogger().isEnabledFor(logging.DEBUG):
            logging.exception("Failed: %s", edf_path)
        else:
            logging.error("Failed: %s — %s", edf_path, err_text)
        return ConversionRecord(
            subject=subj,
            session=sess,
            input_edf=str(edf_path.resolve()),
            output_mat=str(out_mat.resolve()),
            n_samples=0,
            n_channels=0,
            sample_rate=0.0,
            channel_names=[],
            unit_notes="",
            conversion_applied="",
            status="failed",
            error_message=err_text,
        )


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def write_manifest_csv(path: Path, records: Iterable[ConversionRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(records)
    fieldnames = [
        "subject",
        "session",
        "input_edf",
        "output_mat",
        "n_samples",
        "n_channels",
        "sample_rate",
        "channel_names",
        "unit_notes",
        "conversion_applied",
        "status",
        "error_message",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    "subject": r.subject,
                    "session": r.session,
                    "input_edf": r.input_edf,
                    "output_mat": r.output_mat,
                    "n_samples": r.n_samples,
                    "n_channels": r.n_channels,
                    "sample_rate": r.sample_rate,
                    "channel_names": json.dumps(r.channel_names, ensure_ascii=False),
                    "unit_notes": r.unit_notes,
                    "conversion_applied": r.conversion_applied,
                    "status": r.status,
                    "error_message": r.error_message,
                }
            )


def write_manifest_json(path: Path, records: Iterable[ConversionRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = []
    for r in records:
        data.append(
            {
                "subject": r.subject,
                "session": r.session,
                "input_edf": r.input_edf,
                "output_mat": r.output_mat,
                "n_samples": r.n_samples,
                "n_channels": r.n_channels,
                "sample_rate": r.sample_rate,
                "channel_names": r.channel_names,
                "unit_notes": r.unit_notes,
                "conversion_applied": r.conversion_applied,
                "status": r.status,
                "error_message": r.error_message,
            }
        )
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Convert ds004944 BIDS EDF iEEG to .mat for HFO_Initial_Detector_DemoVersion (bipolar+mat)."
    )
    p.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("ds004944"),
        help="Root of the BIDS dataset (default: ds004944)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("standardized_hfo_inputs"),
        help="Directory for output .mat files and manifests (default: standardized_hfo_inputs)",
    )
    p.add_argument(
        "--backend",
        choices=["auto", "mne", "pyedflib"],
        default="auto",
        help="EDF reader backend (default: try mne, then pyedflib)",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s %(message)s",
    )

    dataset_root: Path = args.dataset_root.expanduser().resolve()
    output_dir: Path = args.output_dir.expanduser().resolve()

    if not dataset_root.is_dir():
        logging.error("Dataset root is not a directory: %s", dataset_root)
        return 1

    edf_files = find_task_acute_edf_files(dataset_root)
    if not edf_files:
        logging.error("No files matching %s under %s", EDF_GLOB, dataset_root)
        return 1

    logging.info("Found %d EDF recording(s).", len(edf_files))

    records: list[ConversionRecord] = []
    for edf_path in edf_files:
        logging.info("Processing %s", edf_path)
        rec = process_one_edf(edf_path, output_dir, backend=args.backend)
        records.append(rec)
        if rec.status == "ok":
            logging.info(
                "OK -> %s [%d samples x %d ch @ %.1f Hz]",
                rec.output_mat,
                rec.n_samples,
                rec.n_channels,
                rec.sample_rate,
            )

    manifest_csv = output_dir / "conversion_manifest.csv"
    manifest_json = output_dir / "conversion_manifest.json"
    write_manifest_csv(manifest_csv, records)
    write_manifest_json(manifest_json, records)
    logging.info("Wrote %s", manifest_csv)
    logging.info("Wrote %s", manifest_json)

    n_ok = sum(1 for r in records if r.status == "ok")
    n_fail = len(records) - n_ok
    logging.info("Done. Success: %d, Failed: %d", n_ok, n_fail)

    if n_ok == 0 and n_fail > 0:
        failed_msgs = [r.error_message for r in records if r.status == "failed"]
        if failed_msgs and all(is_annex_or_lfs_pointer_failure(m) for m in failed_msgs):
            logging.warning(
                "All %d recording(s) failed because *.edf files are git-annex/LFS pointers "
                "(large files not present on disk). The conversion script is fine; download "
                "the real EDF binaries into this dataset folder, then run again. "
                "Try: `git annex get .` from the dataset root, DataLad `get`, or an OpenNeuro "
                "download that includes raw data. Use --log-level DEBUG to list each path.",
                n_fail,
            )
    return 0 if n_fail == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
