"""
Level_based_Feature_extraction_kSVD.m and VFactor_Local.m
"""

from __future__ import annotations

import numpy as np
from scipy.stats import skew


def vfactor_local(
    data: np.ndarray,
    wnd: int,
    overlap: int,
    position: tuple[int, int] | None = None,
) -> tuple[float, float]:
    """VFactor_Local.m — returns V2, V3."""
    data = np.asarray(data, dtype=np.float64).reshape(-1)
    if position is None:
        position = (0, len(data) - 1)
    d = data[position[0] : position[1] + 1]
    if len(d) < wnd:
        wnd = len(d)
    starts = np.arange(0, len(d) - wnd + 1, overlap, dtype=int)
    if len(starts) == 0:
        s = float(np.std(d)) if len(d) > 1 else 1.0
        v2 = float((np.max(d) - np.min(d)) / s) if s > 0 else 0.0
        v3 = float((np.max(d) - np.min(d)) / np.std(d)) if np.std(d) > 0 else 0.0
        return v2, v3
    m = np.column_stack([d[s : s + wnd] for s in starts])
    col_stds = np.std(m, axis=0, ddof=1)
    s = float(np.median(col_stds))
    if s == 0:
        s = 1e-12
    v2 = float((np.max(d) - np.min(d)) / s)
    sd = float(np.std(d, ddof=1)) if len(d) > 1 else 1.0
    if sd == 0:
        sd = 1e-12
    v3 = float((np.max(d) - np.min(d)) / sd)
    return v2, v3


def level_based_feature_extraction_ksvd(
    data: np.ndarray,
    matrix_coeff: np.ndarray,
    residual: np.ndarray,
    d_error: np.ndarray | None = None,
    dx: int | None = None,
) -> np.ndarray:
    """
    Returns column vector of features (same ordering as MATLAB column stack).
    For RF pipeline we return row vector per event (horizontal concat in ASLR).
    """
    data = np.asarray(data, dtype=np.float64).reshape(-1)
    matrix_coeff = np.asarray(matrix_coeff, dtype=np.float64)
    residual = np.asarray(residual, dtype=np.float64).reshape(-1)
    dn = float(np.sqrt(np.sum(data**2)))
    if dn == 0:
        dn = 1e-20
    er = float(np.sqrt(np.sum(residual**2)) / dn * 100.0)
    d1 = float(np.sum(np.abs(data)))
    if d1 == 0:
        d1 = 1e-20
    l1er = float(np.sum(np.abs(residual)) / d1 * 100.0)
    v2, v3 = vfactor_local(residual, 128, 4)

    if d_error is None or d_error.size == 0:
        d_error = np.zeros((1, 1), dtype=np.float64)

    if dx is None:
        d_ev = np.sum(np.abs(np.asarray(d_error)), axis=0, keepdims=False)
        s = float(np.sum(d_ev))
        if s > 0:
            p = d_ev / s
            ent = float(-np.sum(p * np.log2(p + 1e-20)))
        else:
            ent = 0.0
        if np.isnan(ent):
            ent = 0.0

        mc = matrix_coeff[:-1, :] if matrix_coeff.shape[0] > 1 else matrix_coeff
        if mc.size == 0:
            evs = evt = max_max = 0.0
        else:
            sgn = np.sign(mc)
            temp_cov = np.cov(sgn, rowvar=False)
            sptl_cov = np.cov(sgn.T, rowvar=False)
            evs = float(np.max(np.linalg.eigvalsh(sptl_cov)))
            evt = float(np.max(np.linalg.eigvalsh(temp_cov)))
            max_max = float(
                np.max(np.abs(matrix_coeff[:-1, :])) / np.linalg.norm(data)
            )
        max_derror = float(np.max(np.abs(d_error)))
        feats = np.array(
            [er, l1er, v2, v3, evs, evt, max_max, max_derror, ent],
            dtype=np.float64,
        )
        return feats

    # nargin >= 5 with dx
    sp = len(data)
    half = sp // 2
    sl = slice(half - dx, half + dx + 1)
    le = float(
        np.sqrt(np.sum(residual[sl] ** 2))
        / np.sqrt(np.sum(data[sl] ** 2))
        * 100.0
    )
    sk = float(skew(residual[sl]))
    d_ev = np.sum(np.abs(d_error), axis=0)
    s = float(np.sum(d_ev))
    if s > 0:
        p = d_ev / s
        ent = float(-np.sum(p * np.log2(p + 1e-20)))
    else:
        ent = 0.0
    if np.isnan(ent):
        ent = 0.0

    mc = matrix_coeff[:-1, :]
    sgn = np.sign(mc)
    temp_cov = np.cov(sgn, rowvar=False)
    sptl_cov = np.cov(sgn.T, rowvar=False)
    evs = float(np.max(np.linalg.eigvalsh(sptl_cov)))
    evt = float(np.max(np.linalg.eigvalsh(temp_cov)))
    max_max = float(np.max(np.abs(matrix_coeff[:-1, :])) / np.linalg.norm(data))
    max_derror = float(np.max(np.abs(d_error)))

    feats = np.array(
        [
            er,
            l1er,
            le,
            v2,
            v3,
            abs(sk),
            evs,
            evt,
            max_max,
            max_derror,
            ent,
        ],
        dtype=np.float64,
    )
    return feats
