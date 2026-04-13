"""
ASLR_Feature_extraction_kSVD.m — dictionary-based feature extraction (no RF).
"""

from __future__ import annotations

import numpy as np
from scipy.signal import filtfilt, firwin

from iom_hfo_pipeline.level_features import level_based_feature_extraction_ksvd
from iom_hfo_pipeline.matlab_compat import detrend_linear, tukeywin
from iom_hfo_pipeline.snake import snake_ksvd_reconst_all_method_omp, snake_ksvd_reconst_general


def _fir1_bandpass(numtaps: int, low_hz: float, high_hz: float, fs: float) -> np.ndarray:
    """MATLAB fir1(numtaps, [low high]/(fs/2)) bandpass — numtaps+1 coefficients."""
    return firwin(
        numtaps + 1,
        [low_hz, high_hz],
        pass_zero=False,
        fs=fs,
    )


def _fir1_lowpass(numtaps: int, cutoff_hz: float, fs: float) -> np.ndarray:
    nyq = fs / 2.0
    return firwin(numtaps + 1, cutoff_hz / nyq, fs=2.0)


def aslr_feature_extraction_ksvd(
    train_data: np.ndarray,
    dictionary_layers: list[np.ndarray],
    fs: float,
    config: dict,
) -> np.ndarray:
    """
    train_data: shape (N, n_events) with N=512 — same as MATLAB Train_Data columns = events.
    dictionary_layers: 4 arrays matching Dictionary.L2kSVD{1:4}
    config keys: NoA, shifts (Overlap), sdRemoved, methodology (unused; OMP only)
    Returns Features shape (n_events, n_feature_cols)
    """
    ripple_band = (80.0, 270.0)
    fast_ripple_band = (230.0, 600.0)
    hfo_band = (80.0, 600.0)

    b1 = _fir1_bandpass(64, ripple_band[0], ripple_band[1], fs)
    b2 = _fir1_bandpass(64, fast_ripple_band[0], fast_ripple_band[1], fs)
    b3 = _fir1_bandpass(64, hfo_band[0], hfo_band[1], fs)
    a = np.array([1.0], dtype=np.float64)

    n = train_data.shape[0]
    env_p = 3.0
    n_events = train_data.shape[1]
    env_th_r = np.zeros(n_events, dtype=np.float64)
    env_th_fr = np.zeros(n_events, dtype=np.float64)

    for k in range(n_events):
        col = train_data[:, k]
        bnd_r = filtfilt(b1, a, col)
        bnd_fr = filtfilt(b2, a, col)
        side_r = np.concatenate(
            [bnd_r[: int(round(n * 1 / 3))], bnd_r[int(round(2 * n / 3)) :]]
        )
        side_fr = np.concatenate(
            [bnd_fr[: int(round(n * 1 / 3))], bnd_fr[int(round(2 * n / 3)) :]]
        )
        env_side_r = _cal_envelope(side_r, fs)
        env_side_fr = _cal_envelope(side_fr, fs)
        env_th_r[k] = max(5.0, env_p * float(np.median(env_side_r)))
        env_th_fr[k] = max(4.0, env_p * float(np.median(env_side_fr)))

    temp = train_data.copy()
    train_proc = detrend_linear(train_data)
    alpha = 0.3
    w = tukeywin(n, alpha).reshape(-1, 1)
    train_proc = train_proc * w

    features0 = np.max(temp, axis=0) - np.min(temp, axis=0)

    noa = config["NoA"]
    shifts = config["shifts"]
    sd_rem = config["sdRemoved"]

    d1 = dictionary_layers[0]
    feats1 = np.zeros((n_events, 9), dtype=np.float64)
    residual1_w = np.zeros_like(train_proc)

    for i in range(n_events):
        col = train_proc[:, i]
        m_c1, _recon1, res1, _err, d_err1 = snake_ksvd_reconst_general(
            col, d1, noa[0], shifts[0], sd_rem[0]
        )
        res1_w = res1 * w.ravel()
        residual1_w[:, i] = res1_w
        f = level_based_feature_extraction_ksvd(
            col, m_c1, res1_w, d_err1
        )
        feats1[i, :] = f

    idx1 = np.array([0, 4, 6, 7], dtype=int)
    features1 = feats1[:, idx1]

    d2 = dictionary_layers[1]
    feats2 = np.zeros((n_events, 11), dtype=np.float64)
    residual2_w = np.zeros_like(train_proc)

    for i in range(n_events):
        r1 = residual1_w[:, i]
        m_c2, _m_c2n, _recon2, res2, _err2, d_err2, _mc, _le = (
            snake_ksvd_reconst_all_method_omp(
                r1, d2, noa[1], shifts[1], sd_rem[1], env_th_r[i]
            )
        )
        res2_w = res2 * w.ravel()
        residual2_w[:, i] = res2_w
        f = level_based_feature_extraction_ksvd(
            r1, m_c2, res2_w, d_err2, 32
        )
        feats2[i, :] = f

    idx2 = np.array([0, 3, 8, 9], dtype=int)
    features2 = feats2[:, idx2]

    d3a = dictionary_layers[2][:, :-1]
    d3 = np.hstack([d3a, dictionary_layers[3]])
    feats3 = np.zeros((n_events, 11), dtype=np.float64)
    residual3 = np.zeros_like(train_proc)

    for i in range(n_events):
        r2 = residual2_w[:, i]
        m_c3, _m_c3n, _recon3, res3, _err3, d_err3, _mc3, _le3 = (
            snake_ksvd_reconst_all_method_omp(
                r2, d3, noa[2], shifts[2], sd_rem[2], env_th_fr[i]
            )
        )
        res3_w = res3 * w.ravel()
        residual3[:, i] = res3_w
        f = level_based_feature_extraction_ksvd(
            r2, m_c3, res3_w, d_err3, 16
        )
        feats3[i, :] = f

    idx3 = np.array([0, 3], dtype=int)
    features3 = feats3[:, idx3]

    ce = np.zeros(n_events, dtype=np.float64)
    for i in range(n_events):
        r3 = residual3[:, i]
        # MATLAB CE: N/2-32 : N/2+31 (64 samples, 1-based inclusive)
        ce[i] = float(np.max(np.abs(r3[n // 2 - 32 : n // 2 + 32])))

    features = np.hstack(
        [
            features0.reshape(-1, 1),
            features1,
            features2,
            features3,
            ce.reshape(-1, 1),
        ]
    )
    return features


def _cal_envelope(sig: np.ndarray, fs: float) -> np.ndarray:
    """calEnvelope.m — abs(hilbert) then lowpass."""
    from scipy.signal import hilbert

    env = np.abs(hilbert(sig))
    f_s = round(fs / 2)
    cutoff = 8.0 / f_s
    b = firwin(3, cutoff, fs=2.0)
    return filtfilt(b, np.array([1.0]), env)
