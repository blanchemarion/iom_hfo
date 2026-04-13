"""
MATLAB-compatible helpers: buffer, Tukey window, detrend.

Maps to MATLAB: buffer(...,'nodelay'), tukeywin, detrend(...,'linear').
"""

from __future__ import annotations

import numpy as np
from numpy.fft import irfft, rfft


def matlab_buffer_nodelay(x: np.ndarray, n: int, p: int) -> np.ndarray:
    """
    MATLAB: buffer(x, n, p, 'nodelay') for vector x.

    p = overlap between consecutive windows (MATLAB definition).
    Hop = n - p. Incomplete tail windows are dropped.
    Output shape (n, num_cols).
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    hop = n - p
    if hop <= 0:
        raise ValueError("Require n > p for positive hop.")
    cols: list[np.ndarray] = []
    i = 0
    while i + n <= len(x):
        cols.append(x[i : i + n])
        i += hop
    if not cols:
        return np.zeros((n, 0), dtype=np.float64)
    return np.column_stack(cols)


def tukeywin(n: int, alpha: float) -> np.ndarray:
    """Tukey window; matches scipy.signal.windows.tukey when alpha=alpha."""
    from scipy.signal.windows import tukey

    return tukey(n, alpha=alpha).astype(np.float64)


def detrend_linear(x: np.ndarray) -> np.ndarray:
    """MATLAB detrend(x,'linear') along axis 0 for 1D/2D."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        return _detrend_1d(x)
    out = np.empty_like(x)
    for c in range(x.shape[1]):
        out[:, c] = _detrend_1d(x[:, c])
    return out


def _detrend_1d(y: np.ndarray) -> np.ndarray:
    n = len(y)
    if n < 2:
        return y.copy()
    t = np.arange(n, dtype=np.float64)
    p = np.polyfit(t, y, 1)
    return y - (p[0] * t + p[1])


def interp1_linear_uniform(h_th: np.ndarray, n_out: int) -> np.ndarray:
    """
    Upsample threshold vector h_th to length n_out (fixes ambiguous MATLAB interp1 usage
    in get_adaptive_threshold: intended mapping from coarse grid to per-sample indices).
    """
    h_th = np.asarray(h_th, dtype=np.float64).reshape(-1)
    m = len(h_th)
    if m == 0:
        return np.zeros(n_out, dtype=np.float64)
    x_old = np.linspace(1.0, float(m), m)
    x_new = np.linspace(1.0, float(m), n_out)
    return np.interp(x_new, x_old, h_th)


interp1_linear = interp1_linear_uniform
