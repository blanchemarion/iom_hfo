"""
Orthogonal Matching Pursuit — port of Functions/OMP_Visualize.m

Dictionary D: columns are atoms.
X: single column vector (one segment).
"""

from __future__ import annotations

import numpy as np


def omp_visualize(
    d: np.ndarray,
    x_col: np.ndarray,
    l_max: int,
    s1: float,
    s2: float,
    s3: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      y: (loc_shft, j_iter) reconstruction at each OMP iteration (MATLAB y(:,j))
      coeff: (K, 1) sparse coefficients (final)
      loc: indices of selected atoms (final)
      residual: (loc_shft, j_max+1) — residual[:,j] = residual before iteration j+1
      error: length up to l_max — relative L2 error after each iteration
    """
    d = np.asarray(d, dtype=np.float64)
    x = np.asarray(x_col, dtype=np.float64).reshape(-1)
    _, k = d.shape
    residual = np.zeros((len(x), l_max + 1), dtype=np.float64)
    residual[:, 0] = x
    y = np.zeros((len(x), l_max), dtype=np.float64)
    coeff = np.zeros((k, 1), dtype=np.float64)
    indx: list[int] = []
    error = np.zeros(l_max, dtype=np.float64)
    error_diff = np.zeros(max(0, l_max - 1), dtype=np.float64)
    loc = np.array([], dtype=int)

    x_norm = np.sqrt(np.sum(x**2))
    if x_norm == 0:
        return y, coeff, loc, residual, error

    j_final = 0
    for j in range(1, l_max + 1):
        j_final = j
        proj = d.T @ residual[:, j - 1]
        pos = int(np.argmax(np.abs(proj)))
        indx.append(pos)
        d_sel = d[:, indx]
        a = np.linalg.pinv(d_sel) @ x
        residual[:, j] = x - d_sel @ a
        y[:, j - 1] = d_sel @ a

        error[j - 1] = np.sqrt(np.sum(residual[:, j] ** 2)) / x_norm
        temp = np.zeros(k, dtype=np.float64)
        temp[np.array(indx)] = a
        coeff[:, 0] = temp
        loc = np.array(indx, dtype=int)

        if j > 1:
            error_diff[j - 2] = error[j - 2] - error[j - 1]
        if j > s3:
            if (error[j - 1] < s1) or (j > 1 and error_diff[j - 2] < s2):
                break

    y_out = y[:, :j_final]
    err_out = error[:j_final]
    return y_out, coeff, loc, residual, err_out
