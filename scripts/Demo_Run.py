#!/usr/bin/env python3
"""
Python analogue of Demo_Run.m — detector + ASLR + MATLAB RF inference.

Sections map to:
  - Demo_Run.m: configuration, load Dictionary, HFO_Initial_Detector_DemoVersion,
    ASLR_Feature_extraction_kSVD, event cap to first 100, eval_RF(model{1}).
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
from iom_hfo_pipeline.rf_infer import eval_matlab_rf, load_matlab_rf_model


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
        description="Demo_Run.py — detector + ASLR features + MATLAB RF inference."
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
        "--rf-model",
        type=Path,
        default=_REPO_ROOT / "Data" / "RF_CRDL_ASLR_Model.mat",
        help="RF_CRDL_ASLR_Model.mat",
    )
    p.add_argument(
        "--rf-model-index",
        type=int,
        default=1,
        help="MATLAB 1-based model cell index (Demo_Run.m uses model{1}).",
    )
    p.add_argument(
        "--max-events",
        type=int,
        default=100,
        help=(
            "Maximum number of events for feature+RF stage (MATLAB demo truncates to 100). "
            "Ignored if --no-event-cap is provided."
        ),
    )
    p.add_argument(
        "--no-event-cap",
        action="store_true",
        help="Disable MATLAB-like event truncation and run on all detected events.",
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
    rf_model_path = (args.rf_model if args.rf_model.is_absolute() else _REPO_ROOT / args.rf_model).resolve()
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

    # MATLAB parity: Demo_Run.m truncates result.pool.event.raw to first 100 columns
    # before ASLR feature extraction and RF inference.
    detected_n_events = int(event_raw.shape[1])
    effective_max_events = None if args.no_event_cap else int(args.max_events)
    if effective_max_events is not None and effective_max_events > 0:
        n_used = min(detected_n_events, effective_max_events)
    else:
        n_used = detected_n_events

    event_raw = event_raw[:, :n_used]
    timestamp = np.asarray(pool["timestamp"]).reshape(-1)[:n_used]
    channelinformation = np.asarray(pool["channelinformation"]).reshape(-1)[:n_used]
    envelope_r = np.asarray(pool["info"]["envelopeSide"]["R"]).reshape(-1)[:n_used]
    envelope_fr = np.asarray(pool["info"]["envelopeSide"]["FR"]).reshape(-1)[:n_used]

    features = aslr_feature_extraction_ksvd(event_raw, dict_layers, fs, denoise)

    trees, rf_meta = load_matlab_rf_model(
        rf_model_path, model_cell_index_1based=args.rf_model_index
    )
    pred, votes, vote_fraction = eval_matlab_rf(features, trees, class_labels=(1, 2))

    pseudo_mask = pred == 1
    real_mask = pred == 2

    np.savez_compressed(
        out_dir / f"{stem}_events.npz",
        event_raw=event_raw,
        timestamp=timestamp,
        channelinformation=channelinformation,
        fs=pool["info"]["fs"],
        envelope_R=envelope_r,
        envelope_FR=envelope_fr,
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

    # MATLAB eval_RF-style post-feature outputs.
    np.savez_compressed(
        out_dir / f"{stem}_predictions.npz",
        pred=pred,
        votes=votes,
        vote_fraction=vote_fraction,
        class_labels=np.array([1, 2], dtype=int),
        pseudo_mask=pseudo_mask,
        real_mask=real_mask,
        pseudo_event_raw=event_raw[:, pseudo_mask],
        real_event_raw=event_raw[:, real_mask],
        pseudo_event_indices=np.where(pseudo_mask)[0],
        real_event_indices=np.where(real_mask)[0],
        rf_model_path=str(rf_model_path),
        rf_model_index_1based=int(args.rf_model_index),
    )

    # Optional .mat bundle for MATLAB (same workspace-style variables)
    savemat(
        str(out_dir / f"{stem}_pipeline_pre_rf.mat"),
        {
            "event_raw": event_raw,
            "Features": features,
            "timestamp": timestamp.reshape(-1, 1),
            "channelinformation": channelinformation.reshape(-1, 1),
            "fs": fs,
            "dictionary_path": str(dict_path),
            "input_mat": str(input_mat),
        },
    )

    # Post-RF MATLAB bundle.
    savemat(
        str(out_dir / f"{stem}_pipeline_post_rf.mat"),
        {
            "event_raw": event_raw,
            "Features": features,
            "pred": pred.reshape(-1, 1),
            "votes": votes,
            "vote_fraction": vote_fraction,
            "pseudo_event_raw": event_raw[:, pseudo_mask],
            "real_event_raw": event_raw[:, real_mask],
            "timestamp": timestamp.reshape(-1, 1),
            "channelinformation": channelinformation.reshape(-1, 1),
            "fs": fs,
            "dictionary_path": str(dict_path),
            "rf_model_path": str(rf_model_path),
            "input_mat": str(input_mat),
        },
    )

    pred_json = {
        "input_mat": str(input_mat),
        "rf_model": str(rf_model_path),
        "rf_model_index_1based": int(args.rf_model_index),
        "n_trees": int(rf_meta["n_trees"]),
        "n_events_detected": detected_n_events,
        "n_events_used_for_rf": int(event_raw.shape[1]),
        "event_cap_enabled": not bool(args.no_event_cap),
        "event_cap_max": None if args.no_event_cap else int(args.max_events),
        "class_semantics": {
            "1": "pseudo-HFO",
            "2": "real-HFO",
        },
        "pred": pred.astype(int).tolist(),
        "votes": votes.astype(int).tolist(),
        "vote_fraction": vote_fraction.astype(float).tolist(),
        "counts": {
            "pseudo_hfo": int(np.sum(pseudo_mask)),
            "real_hfo": int(np.sum(real_mask)),
        },
    }
    (out_dir / f"{stem}_predictions.json").write_text(
        json.dumps(pred_json, indent=2), encoding="utf-8"
    )

    manifest = {
        "input_mat": str(input_mat),
        "dictionary": str(dict_path),
        "rf_model": str(rf_model_path),
        "n_events_detected": detected_n_events,
        "n_events": int(event_raw.shape[1]),
        "n_features": int(features.shape[1]),
        "fs_hz": fs,
        "rf_model_info": rf_meta,
        "event_cap": {
            "enabled": not bool(args.no_event_cap),
            "max_events": None if args.no_event_cap else int(args.max_events),
        },
        "prediction_counts": {
            "pseudo_hfo": int(np.sum(pseudo_mask)),
            "real_hfo": int(np.sum(real_mask)),
        },
        "qc": {
            "max_abs_raw_assumed_uv": max_abs_uv,
        },
        "event_shape_samples_by_events": list(event_raw.shape),
        "feature_shape_events_by_feats": list(features.shape),
        "outputs": {
            "events_npz": str(out_dir / f"{stem}_events.npz"),
            "features_npz": str(out_dir / f"{stem}_features.npz"),
            "pre_rf_mat": str(out_dir / f"{stem}_pipeline_pre_rf.mat"),
            "predictions_npz": str(out_dir / f"{stem}_predictions.npz"),
            "predictions_json": str(out_dir / f"{stem}_predictions.json"),
            "post_rf_mat": str(out_dir / f"{stem}_pipeline_post_rf.mat"),
        },
    }
    (out_dir / f"{stem}_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
