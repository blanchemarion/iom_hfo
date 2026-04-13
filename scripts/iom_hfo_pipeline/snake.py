"""
Snake kSVD reconstruction — ports of:
  Snake_kSVD_reconst_general.m
  Snake_kSVD_reconst_AllMethod.m (OMP branch; includes LE side residual)
"""

from __future__ import annotations

import numpy as np

from iom_hfo_pipeline.matlab_compat import matlab_buffer_nodelay
from iom_hfo_pipeline.omp import omp_visualize


def snake_ksvd_reconst_general(
    data: np.ndarray,
    dictionary: np.ndarray,
    no_atoms: int,
    overlap: int,
    sd_removed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    """
    Snake_kSVD_reconst_general(..., draw=0). smooth==0 path only.
    """
    data = np.asarray(data, dtype=np.float64).reshape(-1)
    dictionary = np.asarray(dictionary, dtype=np.float64)
    loc_shft = dictionary.shape[0]
    dic_rows = dictionary.shape[0]
    n = len(data)
    no_block = n // overlap
    no_seg = dic_rows // overlap

    segments = matlab_buffer_nodelay(data, loc_shft, loc_shft - overlap)
    n_seg_buf = segments.shape[1]

    coeff_list: list[np.ndarray] = []
    e_store = np.zeros((no_atoms, n_seg_buf), dtype=np.float64)
    rec1 = np.zeros((loc_shft, n_seg_buf), dtype=np.float64)
    coeff_m = np.zeros(n_seg_buf, dtype=np.float64)

    for no in range(n_seg_buf):
        seg = segments[:, no]
        rec, coeff, _, _, err = omp_visualize(
            dictionary, seg, no_atoms, 0.0, 0.0, 1
        )
        rec1[:, no] = rec[:, -1]
        coeff_list.append(np.asarray(coeff, dtype=np.float64).reshape(-1))
        e_full = np.zeros(no_atoms, dtype=np.float64)
        e_full[: len(err)] = err
        e_store[:, no] = e_full
        seg_norm = np.linalg.norm(seg)
        coeff_m[no] = np.max(np.abs(coeff)) / seg_norm if seg_norm > 0 else 0.0

    matrix_coeff = np.column_stack(coeff_list)
    if no_atoms == 1:
        d_error = e_store.copy()
    else:
        d_error = np.diff(e_store, axis=0)

    rec1_buff: list[np.ndarray] = []
    for no in range(n_seg_buf):
        col = rec1[:, no]
        rec1_buff.append(matlab_buffer_nodelay(col, overlap, 0))

    reconstruction = _snake_reconstruction_edges(
        rec1_buff, no_block, no_seg, dic_rows, overlap, sd_removed
    )

    residual = data - reconstruction
    err = float(
        np.sqrt(np.sum(residual**2)) / np.sqrt(np.sum(data**2)) * 100.0
    )
    return matrix_coeff, reconstruction, residual, err, d_error


def _snake_reconstruction_edges(
    rec1_buff: list[np.ndarray],
    no_block: int,
    no_seg: int,
    dic_rows: int,
    overlap: int,
    sd_removed: int,
) -> np.ndarray:
    """
    MATLAB smooth==0. rec1_buff[k] is 0-based segment index k, shape (overlap, ncols).
    """
    reconstruction: list[np.ndarray] = []
    counter = 0
    counter2 = 0
    counter3 = 0
    d_over_o = dic_rows // overlap

    for k in range(1, no_block + 1):
        rec_t: np.ndarray | None = None

        if k < d_over_o:
            cols: list[np.ndarray] = []
            for k2 in range(1 + counter2, k + counter2 + 1):
                bi = k2 - counter2 - 1
                ci = (k + 1 - k2 + counter2) - 1
                b = rec1_buff[bi]
                if 0 <= ci < b.shape[1]:
                    cols.append(b[:, ci])
            counter2 += 1
            if cols:
                rec_t = np.column_stack(cols)
                reconstruction.append(np.mean(rec_t, axis=1))

        elif k > no_block - d_over_o + 1:
            m = (no_block + 1) % k
            cols = []
            for k2 in range(1 + counter3, m + counter3 + 1):
                bi = (no_block - d_over_o + 1 - k2 + 1 + counter3) - 1
                ci = (k2 + 1) - 1
                if 0 <= bi < len(rec1_buff):
                    b = rec1_buff[bi]
                    if 0 <= ci < b.shape[1]:
                        cols.append(b[:, ci])
            counter3 += 1
            if cols:
                rec_t = np.column_stack(cols)
                reconstruction.append(np.mean(rec_t, axis=1))

        else:
            cols = []
            for k2 in range(1 + counter, no_seg + counter + 1):
                bi = k2 - 1
                ci = (no_seg + 1 - k2 + counter) - 1
                b = rec1_buff[bi]
                if 0 <= ci < b.shape[1]:
                    cols.append(b[:, ci])
            counter += 1
            if cols:
                rec_t = np.column_stack(cols)
                c0 = sd_removed
                c1 = no_seg - sd_removed
                reconstruction.append(np.mean(rec_t[:, c0:c1], axis=1))

    return np.concatenate(reconstruction)


def snake_ksvd_reconst_all_method_omp(
    data: np.ndarray,
    dictionary: np.ndarray,
    no_atoms: int,
    overlap: int,
    sd_removed: int,
    th: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Snake_kSVD_reconst_AllMethod — OMP branch only.
    Returns Matrix_Coeff, Matrix_CoeffN, Reconstruction, Residual, Error, dError,
    Max_Coeff, LE (5x1 as in MATLAB)
    """
    data = np.asarray(data, dtype=np.float64).reshape(-1)
    dictionary = np.asarray(dictionary, dtype=np.float64)
    loc_shft = dictionary.shape[0]
    dic_rows = dictionary.shape[0]
    n = len(data)
    no_block = n // overlap
    no_seg = dic_rows // overlap

    segments = matlab_buffer_nodelay(data, loc_shft, loc_shft - overlap)
    n_seg_buf = segments.shape[1]

    coeff_list: list[np.ndarray] = []
    coeff_n_list: list[np.ndarray] = []
    e_store = np.zeros((no_atoms, n_seg_buf), dtype=np.float64)
    rec1 = np.zeros((loc_shft, n_seg_buf), dtype=np.float64)
    coeff_m = np.zeros(n_seg_buf, dtype=np.float64)

    for no in range(n_seg_buf):
        seg = segments[:, no]
        rec, coeff, _, _, err = omp_visualize(
            dictionary, seg, no_atoms, 0.0, 0.0, 1
        )
        rec1[:, no] = rec[:, -1]
        c = np.asarray(coeff, dtype=np.float64).reshape(-1)
        coeff_list.append(c)
        seg_norm = np.linalg.norm(seg)
        coeff_n_list.append(c / seg_norm if seg_norm > 0 else c)
        e_full = np.zeros(no_atoms, dtype=np.float64)
        e_full[: len(err)] = err
        e_store[:, no] = e_full
        coeff_m[no] = np.max(np.abs(coeff)) / seg_norm if seg_norm > 0 else 0.0

    matrix_coeff = np.column_stack(coeff_list)
    matrix_coeff_n = np.column_stack(coeff_n_list)
    if no_atoms == 1:
        d_error = e_store.copy()
    else:
        d_error = np.diff(e_store, axis=0)

    rec1_buff: list[np.ndarray] = []
    for no in range(n_seg_buf):
        rec1_buff.append(matlab_buffer_nodelay(rec1[:, no], overlap, 0))

    reconstruction = _snake_reconstruction_edges(
        rec1_buff, no_block, no_seg, dic_rows, overlap, sd_removed
    )
    residual = data - reconstruction
    err = float(
        np.sqrt(np.sum(residual**2)) / np.sqrt(np.sum(data**2)) * 100.0
    )

    # LE from Snake_kSVD_reconst_AllMethod (lines 147-158)
    sz = dictionary.shape[0] // 2
    res2_side = np.concatenate([residual[0 : 256 - sz], residual[257 + sz :]])
    seg2_side = np.concatenate([data[0 : 256 - sz], data[257 + sz :]])
    le = np.zeros(5, dtype=np.float64)
    if np.max(np.abs(res2_side)) > th:
        le[0] = np.max(np.abs(res2_side))
        le[1] = np.max(res2_side**2)
        mask = np.abs(res2_side) > th
        le[2] = np.sum(np.abs(res2_side[mask]))
        le[3] = np.sum(res2_side[mask] ** 2)
        le[4] = float(
            _zerocross_count(np.abs(res2_side.reshape(1, -1)) - th)[0]
        )

    max_coeff = float(np.max(coeff_m))
    return (
        matrix_coeff,
        matrix_coeff_n,
        reconstruction,
        residual,
        err,
        d_error,
        max_coeff,
        le,
    )


def _zerocross_count(data: np.ndarray) -> tuple[int, np.ndarray]:
    """Port of zerocross_count.m (single row)."""
    x = data[0, :].astype(np.float64)
    x_sign = np.sign(x)
    x_sign[x_sign == 0] = 1
    d = np.diff(x_sign)
    idx = np.where(d != 0)[0]
    return len(idx), idx
