#!/usr/bin/env python3
"""
Python analogue of Demo_Run.m — detector + ASLR feature extraction only (no RF).

Sections map to:
  - Demo_Run.m: configuration, load Dictionary, HFO_Initial_Detector_DemoVersion,
    ASLR_Feature_extraction_kSVD (no eval_RF, no 100-event truncation).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat

# Repo root: parent of scripts/
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS = _REPO_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from iom_hfo_pipeline.aslr import aslr_feature_extraction_ksvd
from iom_hfo_pipeline.detector import hfo_initial_detector_demo_version


def build_detection_param(fs_default: float = 2000.0) -> dict:
    """Mirrors Demo_Run.m param.detection (after config)."""
    resample_rate = 2000.0
    return {
        "DataType": "mat",
        "blockRange": [],
        "saveMat": "Yes",
        "derivationType": "bipolar",
        "signalUnit": "uV",
        "resampleState": "No",
        "resampleRate": resample_rate,
        "saveResults": True,
        "removeSideHFO": "Yes",
        "lowerBand": 1.0,
        "HFOBand": np.array([80.0, 600.0]),
        "RippleBand": np.array([80.0, 270.0]),
        "FastRippleBand": np.array([230.0, 600.0]),
        "frameLength": 128,
        "overlapLength": 0,
        "numFrames": int(round(60 / (128 / resample_rate))),
        "numOverlapFrames": int(round(30 / (128 / resample_rate))),
        "eventLength": 512,
        "FRThreshold": 100.0,
        "minThresholdRipple": 5.0,
        "minThresholdFastRipple": 4.0,
        "thresholdMultiplier": 3.0,
        "numCrossing": 6,
        "numSideCrossing": 4,
        "cutoffRipple": 80.0,
        "cutoffFastRipple": 250.0,
    }


def build_denoising_param() -> dict:
    """param.denoising from Demo_Run.m."""
    return {
        "NoA": np.array([6, 4, 3], dtype=int),
        "shifts": np.array([4, 4, 4], dtype=int),
        "sdRemoved": np.array([6, 8, 4], dtype=int),
        "methodology": ["OMP", "OMP", "OMP"],
    }


def load_dictionary(path: Path) -> list[np.ndarray]:
    """Dictionary.L2kSVD — four cell layers."""
    d = loadmat(str(path), struct_as_record=False, squeeze_me=True)
    dic = d["Dictionary"]
    layers = dic.L2kSVD
    return [np.asarray(layers[i], dtype=np.float64) for i in range(4)]


def main() -> int:
    p = argparse.ArgumentParser(
        description="Demo_Run.py — detector + ASLR features (no Random Forest)."
    )
    p.add_argument(
        "--input-mat",
        type=Path,
        required=True,
        help="Standardized recording .mat (data.data, montage.SampleRate)",
    )
    p.add_argument(
        "--dictionary",
        type=Path,
        default=_REPO_ROOT / "Data" / "Dictionary_CRDL_ASLR.mat",
        help="Dictionary_CRDL_ASLR.mat",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=_REPO_ROOT / "output" / "demo_run_py",
        help="Output directory for npz + manifest",
    )
    args = p.parse_args()

    input_mat = (args.input_mat if args.input_mat.is_absolute() else _REPO_ROOT / args.input_mat).resolve()
    dict_path = (args.dictionary if args.dictionary.is_absolute() else _REPO_ROOT / args.dictionary).resolve()
    out_dir = (args.output_dir if args.output_dir.is_absolute() else _REPO_ROOT / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = input_mat.stem

    # Quick load for fs + QC (matches expected bipolar .mat layout)
    m0 = loadmat(str(input_mat), struct_as_record=False, squeeze_me=True)
    fs = float(np.asarray(m0["montage"].SampleRate).squeeze())
    raw0 = np.asarray(m0["data"].data, dtype=np.float64)
    max_abs_uv = float(np.max(np.abs(raw0)))

    detection = build_detection_param(fs_default=fs)
    detection["fileName"] = str(input_mat)
    denoise = build_denoising_param()

    dict_layers = load_dictionary(dict_path)

    out = hfo_initial_detector_demo_version(str(input_mat), detection)
    pool = out.get("pool")
    if pool is None:
        print("No events detected (pool is empty).", file=sys.stderr)
        manifest = {
            "input_mat": str(input_mat),
            "dictionary": str(dict_path),
            "n_events": 0,
            "status": "no_events",
            "qc": {
                "max_abs_raw_assumed_uv": max_abs_uv,
                "note": "If max_abs is far beyond typical iEEG (e.g. >> 1e5 uV), "
                "check units; FR artifact rejection uses 100 uV on the FR band.",
            },
        }
        (out_dir / f"{stem}_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        return 1

    raw = pool["event"]["raw"]
    if raw.ndim == 1:
        raw = raw.reshape(-1, 1)
    # pool.event.raw: (512, n_events) as MATLAB
    event_raw = np.asarray(raw, dtype=np.float64)
    if event_raw.shape[1] == 0:
        print("No events detected (pool is empty).", file=sys.stderr)
        manifest = {
            "input_mat": str(input_mat),
            "dictionary": str(dict_path),
            "n_events": 0,
            "status": "no_events",
            "qc": {
                "max_abs_raw_assumed_uv": max_abs_uv,
                "note": "If max_abs is far beyond typical iEEG (e.g. >> 1e5 uV), "
                "check units; FR artifact rejection uses 100 uV on the FR band.",
            },
        }
        (out_dir / f"{stem}_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        return 1

    features = aslr_feature_extraction_ksvd(event_raw, dict_layers, fs, denoise)

    np.savez_compressed(
        out_dir / f"{stem}_events.npz",
        event_raw=event_raw,
        timestamp=pool["timestamp"],
        channelinformation=pool["channelinformation"],
        fs=pool["info"]["fs"],
        envelope_R=pool["info"]["envelopeSide"]["R"],
        envelope_FR=pool["info"]["envelopeSide"]["FR"],
        input_mat=str(input_mat),
    )

    feature_names = [
        "range0",
        "L1_F1a",
        "L1_F1b",
        "L1_F1c",
        "L1_F1d",
        "L2_F2a",
        "L2_F2b",
        "L2_F2c",
        "L2_F2d",
        "L3_F3a",
        "L3_F3b",
        "CE",
    ]

    np.savez_compressed(
        out_dir / f"{stem}_features.npz",
        features=features,
        feature_names=np.array(feature_names),
        denoising_NoA=denoise["NoA"],
        denoising_shifts=denoise["shifts"],
        denoising_sdRemoved=denoise["sdRemoved"],
    )

    # Optional .mat bundle for MATLAB (same workspace-style variables)
    savemat(
        str(out_dir / f"{stem}_pipeline_pre_rf.mat"),
        {
            "event_raw": event_raw,
            "Features": features,
            "timestamp": pool["timestamp"],
            "channelinformation": pool["channelinformation"],
            "fs": fs,
            "dictionary_path": str(dict_path),
            "input_mat": str(input_mat),
        },
    )

    manifest = {
        "input_mat": str(input_mat),
        "dictionary": str(dict_path),
        "n_events": int(event_raw.shape[1]),
        "n_features": int(features.shape[1]),
        "fs_hz": fs,
        "qc": {
            "max_abs_raw_assumed_uv": max_abs_uv,
        },
        "event_shape_samples_by_events": list(event_raw.shape),
        "feature_shape_events_by_feats": list(features.shape),
        "outputs": {
            "events_npz": str(out_dir / f"{stem}_events.npz"),
            "features_npz": str(out_dir / f"{stem}_features.npz"),
            "pre_rf_mat": str(out_dir / f"{stem}_pipeline_pre_rf.mat"),
        },
    }
    (out_dir / f"{stem}_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
