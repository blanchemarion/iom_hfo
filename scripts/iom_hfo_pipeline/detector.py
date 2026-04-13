"""
HFO_Initial_Detector_DemoVersion.m — initial dual-band HFO pool (bipolar + continuous .mat).
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, firwin
from scipy.io import loadmat

from iom_hfo_pipeline.matlab_compat import interp1_linear_uniform as interp1_linear


def load_bipolar_mat(path: str) -> tuple[np.ndarray, float]:
    """Load standardized .mat with data.data [samples x ch] and montage.SampleRate."""
    m = loadmat(path, struct_as_record=False, squeeze_me=True)
    d = m["data"]
    raw = np.asarray(d.data, dtype=np.float64)
    fs = float(np.asarray(m["montage"].SampleRate).squeeze())
    return raw, fs


def buffered_stats(
    data: np.ndarray, frame: int, overlap: int, stat: str
) -> tuple[np.ndarray, int]:
    """buffered_stats.m — buffer then column-wise std/mean/median/var."""
    # MATLAB buffer(x, frame, overlap): hop = frame - overlap
    buf = _matlab_buffer_nodelay(data, frame, overlap)
    if buf.size == 0:
        return np.array([]), 0
    if stat == "std":
        y = np.std(buf, axis=0, ddof=1)
    elif stat == "var":
        y = np.var(buf, axis=0, ddof=1)
    elif stat == "mean":
        y = np.mean(buf, axis=0)
    elif stat == "median":
        y = np.median(buf, axis=0)
    else:
        raise ValueError(stat)
    bs = buf.shape[1]
    return y, bs


def _matlab_buffer_nodelay(x: np.ndarray, n: int, p: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    hop = n - p
    if hop <= 0:
        raise ValueError("n > p required")
    cols = []
    i = 0
    while i + n <= len(x):
        cols.append(x[i : i + n])
        i += hop
    if not cols:
        return np.zeros((n, 0))
    return np.column_stack(cols)


def get_adaptive_threshold(
    data: np.ndarray,
    frame: int,
    overlap: int,
    h_frame: int,
    h_overlap: int,
    typ: str,
    param: float,
) -> tuple[np.ndarray, np.ndarray]:
    """get_adaptive_threshold.m — Std path."""
    if typ == "Std":
        v, _ = buffered_stats(data, frame, overlap, "std")
    else:
        v, _ = buffered_stats(data, frame, overlap, "var")
    lv = len(v)
    # If h_frame > len(v), MATLAB buffer yields empty; clamp (common for short clips / long h_frame).
    hf = int(min(h_frame, max(lv, 1)))
    ho = int(min(h_overlap, max(hf - 1, 0)))
    h_th, bs = buffered_stats(v, hf, ho, "median")
    n = len(data)
    nv = len(v)
    # get_adaptive_threshold.m: dt uses the same h_frame/h_overlap as buffered_stats on v
    dt = nv - (bs - 1) * (hf - ho)
    if dt < 0.75 * hf and len(h_th) >= 2:
        h_th = np.asarray(h_th, dtype=np.float64).copy()
        h_th[-1] = h_th[-2]
    th = interp1_linear(np.asarray(h_th, dtype=np.float64).reshape(-1), n) * param
    return v, th


def find_adaptive_event(
    data: np.ndarray,
    th: np.ndarray,
    id_type: int,
    th_type: int,
    reject: int,
) -> np.ndarray:
    """
    find_adaptive_event.m — id_type=2 (peak), th_type=1, reject=9.

    Groups are runs of sorted threshold indices with gaps <= 100 samples (split where
    diff > 100). For th_type==1, groups with length > 100 or <= 2 are removed.
    id_type==2: peak index is where data == max(data(group)) (MATLAB uses max, not abs).
    Discard pass: 512-sample window [stamp-255:stamp+256]; compare to reject*th(stamp).
    MATLAB empty catch on out-of-range window leaves discard=0 (event kept).
    """
    data = np.asarray(data, dtype=np.float64).reshape(-1)
    th = np.asarray(th, dtype=np.float64).reshape(-1)
    mask = (data > th) | (data < -th)
    index = np.where(mask)[0]
    if index.size == 0:
        return np.array([], dtype=np.int64)
    bad = (data[index] > reject * th[index]) | (data[index] < -reject * th[index])
    index = index[~bad]
    if index.size <= 1:
        return np.array([], dtype=np.int64)
    splits = np.where(np.diff(index) > 100)[0] + 1
    groups = np.split(index, splits)
    peaks: list[int] = []
    for g in groups:
        if g.size == 0:
            continue
        if th_type == 1 and (g.size > 100 or g.size <= 2):
            continue
        if id_type == 2:
            gmax = float(np.max(data[g]))
            hit = np.where(data[g] == gmax)[0]
            peaks.append(int(g[hit[0]]))
    stamp = np.array(peaks, dtype=np.int64)
    if stamp.size == 0:
        return stamp
    discard = np.zeros(len(stamp), dtype=bool)
    n = len(data)
    for i in range(len(stamp)):
        s = int(stamp[i])
        # MATLAB: temp=data(stamp(i)-255:stamp(i)+256) — 512 samples; invalid range → catch, keep
        if s - 255 < 0 or s + 256 >= n:
            continue
        temp = data[s - 255 : s + 257]
        ts = float(th[s])
        if np.any(temp > reject * ts) or np.any(temp < -reject * ts):
            discard[i] = True
    return stamp[~discard]


def getaligneddata(
    data: np.ndarray,
    index: np.ndarray,
    range_: tuple[int, int],
    artifact: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """getaligneddata.m — 0-based indices; data (samples, channels)."""
    if artifact is None:
        artifact = np.zeros((0, 2), dtype=np.float64)
    index = np.asarray(index, dtype=np.int64).reshape(-1)
    r0, r1 = int(round(range_[0])), int(round(range_[1]))
    n_samp = r1 - r0 + 1
    t = len(index)
    keep = np.zeros(t, dtype=bool)
    aligned = np.zeros((n_samp, data.shape[1], t), dtype=np.float64)
    aligned_index = np.zeros((t, n_samp), dtype=np.int64)
    ns = data.shape[0]
    for kidx in range(t):
        aligned_index[kidx, :] = index[kidx] + np.arange(r0, r1 + 1, dtype=np.int64)
        row = aligned_index[kidx, :]
        if np.any(row < 0) or np.any(row >= ns):
            continue
        aligned[:, :, kidx] = data[row, :]
        keep[kidx] = True
    aligned = aligned[:, :, keep]
    aligned_index = aligned_index[keep, :]
    return aligned, aligned_index, keep


def cal_envelope(sig: np.ndarray, fs: float) -> np.ndarray:
    """calEnvelope.m — fir1(2, cutoff) with cutoff = 8 / round(fs/2) as MATLAB normalized Wn → Hz."""
    from scipy.signal import hilbert

    env = np.abs(hilbert(sig))
    f_s = max(int(round(fs / 2)), 1)
    wn_norm = 8.0 / float(f_s)
    cutoff_hz = wn_norm * (fs / 2.0)
    b = firwin(3, cutoff_hz, fs=fs)
    return filtfilt(b, np.array([1.0]), env)


def zerocross_count_row(x: np.ndarray) -> int:
    """zerocross_count for single row."""
    x_sign = np.sign(x)
    x_sign[x_sign == 0] = 1
    d = np.diff(x_sign)
    return int(np.sum(d != 0))


def _zerocross_indices_matlab(x: np.ndarray) -> np.ndarray:
    """Indices i where sign changes between x[i] and x[i+1] (zerocross_count.m)."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    x_sign = np.sign(x)
    x_sign[x_sign == 0] = 1
    x_sign_diff = np.diff(x_sign)
    return np.where(x_sign_diff != 0)[0]


def hfo_amp_detector(
    x: np.ndarray,
    s_d: np.ndarray | None,
    th: float | np.ndarray,
    fs: float,
    fc: float,
    nc: int,
) -> bool:
    """hfo_amp_detector.m — returns True if pattern detected."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if s_d is not None and isinstance(s_d, np.ndarray) and float(np.max(s_d)) > 4:
        th = float(np.max(s_d))
    elif isinstance(th, np.ndarray):
        th = float(np.mean(th))
    else:
        th = float(th)
    n = int(round(1.0 / fc * fs))
    h = np.ones(nc - 1, dtype=np.float64)
    nc_max = 20

    for sign in (1, -1):
        z = x - sign * th
        if zerocross_count_row(z.reshape(1, -1)) == 0:
            continue
        idx = _zerocross_indices_matlab(z)
        if idx.size == 0:
            continue
        ix = np.diff(idx).astype(np.float64)
        if ix.size == 0:
            continue
        dx = np.convolve((ix < n).astype(np.float64), h, mode="full")
        if np.any(dx >= nc - 1) and ix.size < nc_max - 1:
            return True
    return False


def check_centralized_component(
    data: np.ndarray,
    center_range: float = 0.1,
    side_range: float = 1 / 3,
    wnd: int = 8,
    overlap: int = 4,
    param: float = 3.0,
) -> tuple[bool, float]:
    """Check_centralized_component.m"""
    n = len(data)
    sd_center = _temp_variance(
        data[int(round(n * center_range)) : int(round(n * (1 - center_range)))],
        wnd,
        overlap,
        2,
    )
    sd_side = np.concatenate(
        [
            _temp_variance(data[: int(round(n * side_range))], wnd, overlap, 2),
            _temp_variance(
                data[int(round(2 * n * side_range)) :], wnd, overlap, 2
            ),
        ]
    )
    th = float(param * np.median(sd_side))
    ff_mid = np.where(sd_center > th)[0]
    accept = bool(ff_mid.size > 0)
    return accept, th


def _temp_variance(
    data: np.ndarray, frame: int, overlap: int, typ: int
) -> np.ndarray:
    """temp_variance.m (simplified sliding window)."""
    data = np.asarray(data, dtype=np.float64).reshape(-1)
    win = int(frame)
    if len(data) < win:
        return np.array([np.std(data) if typ == 2 else np.var(data)])
    starts = np.arange(0, len(data) - win + 1, overlap, dtype=int)
    out = []
    for s in starts:
        seg = data[s : s + win]
        if typ == 2:
            out.append(np.std(seg, ddof=1))
        else:
            out.append(np.var(seg, ddof=1))
    return np.asarray(out, dtype=np.float64)


def hfo_initial_detector_demo_version(
    file_path: str, detection: dict
) -> dict:
    """
    Port of HFO_Initial_Detector_DemoVersion(fileName, param.detection).
    `detection` must include all keys from Demo_Run.m param.detection.
    """
    m = loadmat(file_path, struct_as_record=False, squeeze_me=True)
    d = m["data"]
    data = np.asarray(d.data, dtype=np.float64)
    fs = float(np.asarray(m["montage"].SampleRate).squeeze())

    if detection.get("signalUnit", "uV") == "mV":
        data = data * 1000.0

    b0, a0 = butter(2, detection["lowerBand"] / (fs / 2), btype="high")
    b1 = firwin(
        65,
        [detection["RippleBand"][0], detection["RippleBand"][1]],
        pass_zero=False,
        fs=fs,
    )
    b2 = firwin(
        65,
        [detection["FastRippleBand"][0], detection["FastRippleBand"][1]],
        pass_zero=False,
        fs=fs,
    )
    b3 = firwin(
        65,
        [detection["HFOBand"][0], detection["HFOBand"][1]],
        pass_zero=False,
        fs=fs,
    )
    a = np.array([1.0], dtype=np.float64)

    input_raw = filtfilt(b0, a0, data, axis=0)
    input_filtered_R = filtfilt(b1, a, input_raw, axis=0)
    input_filtered_FR = filtfilt(b2, a, input_raw, axis=0)
    input_filtered_HFO = filtfilt(b3, a, input_raw, axis=0)

    n = input_raw.shape[1]
    fl = detection["frameLength"]
    ol = detection["overlapLength"]
    nf = detection["numFrames"]
    nof = detection["numOverlapFrames"]

    timestamp_R: list[np.ndarray] = []
    for i in range(n):
        _, th_r = get_adaptive_threshold(
            input_filtered_R[:, i],
            fl,
            ol,
            nf,
            nof,
            "Std",
            5.0,
        )
        try:
            st = find_adaptive_event(input_filtered_R[:, i], th_r, 2, 1, 9)
            if st.size:
                ts = np.column_stack([st, np.full(len(st), i, dtype=int)])
            else:
                ts = np.zeros((0, 2), dtype=int)
        except Exception:
            ts = np.zeros((0, 2), dtype=int)
        timestamp_R.append(ts)

    # Merge R and FR streams (same structure as MATLAB)
    timestamp_FR: list[np.ndarray] = []
    for i in range(n):
        _, th_fr = get_adaptive_threshold(
            input_filtered_FR[:, i],
            fl,
            ol,
            nf,
            nof,
            "Std",
            5.0,
        )
        try:
            st = find_adaptive_event(input_filtered_FR[:, i], th_fr, 2, 1, 9)
            if st.size:
                ts = np.column_stack([st, np.full(len(st), i, dtype=int)])
            else:
                ts = np.zeros((0, 2), dtype=int)
        except Exception:
            ts = np.zeros((0, 2), dtype=int)
        timestamp_FR.append(ts)

    # Merge per-channel duplicate removal (MATLAB: remove R peaks near FR per channel)
    timestamp_all_ch: list[np.ndarray] = []
    for ch in range(n):
        tr = timestamp_R[ch]
        tfr = timestamp_FR[ch]
        if tr.size == 0:
            timestamp_all_ch.append(tfr)
            continue
        if tfr.size == 0:
            timestamp_all_ch.append(tr)
            continue
        half_w = int(round(detection["eventLength"] / 2 - 1))
        if tr.size and tfr.size:
            keep = np.ones(len(tr), dtype=bool)
            for i in range(len(tr)):
                if np.any(np.abs(tr[i, 0] - tfr[:, 0]) < half_w):
                    keep[i] = False
            tr = tr[keep]
        timestamp_all_ch.append(np.vstack([tr, tfr]) if tfr.size else tr)

    t_all = (
        np.vstack([t for t in timestamp_all_ch if t.size])
        if any(t.size for t in timestamp_all_ch)
        else np.zeros((0, 2), dtype=int)
    )
    if t_all.size:
        t_all = t_all[np.argsort(t_all[:, 0])]

    aligned_all, aligned_idx, K_all = getaligneddata(
        input_raw,
        t_all[:, 0],
        (
            -round(detection["eventLength"] / 2 - 1),
            round(detection["eventLength"] / 2),
        ),
    )
    t_all = t_all[K_all]

    # event.dataAll
    n_ev = aligned_all.shape[2]
    if n_ev == 0:
        return {"pool": None}

    event_length = detection["eventLength"]
    data_all = np.zeros((event_length, 5, n_ev), dtype=np.float64)
    for i in range(n_ev):
        ch = int(t_all[i, 1])
        idx_row = aligned_idx[i, :].astype(int)
        data_all[:, 0, i] = aligned_all[:, ch, i]
        data_all[:, 1, i] = input_filtered_HFO[idx_row, ch]
        data_all[:, 2, i] = input_filtered_R[idx_row, ch]
        data_all[:, 3, i] = input_filtered_FR[idx_row, ch]
        data_all[:, 4, i] = idx_row.astype(np.float64)

    # FR artifact rejection
    accepted_fr = np.zeros(n_ev, dtype=bool)
    for ev_no in range(n_ev):
        mx = np.max(data_all[:, 3, ev_no])
        mn = np.min(data_all[:, 3, ev_no])
        accepted_fr[ev_no] = (mx < detection["FRThreshold"]) and (
            mn > -detection["FRThreshold"]
        )
    data_all = data_all[:, :, accepted_fr]
    t_all = t_all[accepted_fr]

    n_ev = data_all.shape[2]
    env_th_r = np.zeros(n_ev, dtype=np.float64)
    env_th_fr = np.zeros(n_ev, dtype=np.float64)
    for ev_no in range(n_ev):
        temp_r = data_all[:, 2, ev_no]
        temp_fr = data_all[:, 3, ev_no]
        el = event_length
        env_r_side = cal_envelope(
            np.concatenate(
                [
                    temp_r[: int(round(el * 1 / 3))],
                    temp_r[int(round(2 * el / 3)) :],
                ]
            ),
            fs,
        )
        env_fr_side = cal_envelope(
            np.concatenate(
                [
                    temp_fr[: int(round(el * 1 / 3))],
                    temp_fr[int(round(2 * el / 3)) :],
                ]
            ),
            fs,
        )
        env_th_r[ev_no] = max(
            detection["minThresholdRipple"],
            detection["thresholdMultiplier"] * float(np.median(env_r_side)),
        )
        env_th_fr[ev_no] = max(
            detection["minThresholdFastRipple"],
            detection["thresholdMultiplier"] * float(np.median(env_fr_side)),
        )

    if detection["removeSideHFO"] != "Yes":
        # Not used in Demo_Run default
        pass

    accepted_osc = np.zeros(n_ev, dtype=bool)
    for ev_no in range(n_ev):
        p_r = hfo_amp_detector(
            data_all[:, 2, ev_no],
            None,
            env_th_r[ev_no],
            fs,
            detection["cutoffRipple"],
            detection["numCrossing"],
        )
        p_r_l = hfo_amp_detector(
            data_all[: int(round(event_length / 3)), 2, ev_no],
            None,
            env_th_r[ev_no],
            fs,
            detection["cutoffRipple"],
            detection["numSideCrossing"],
        )
        p_r_r = hfo_amp_detector(
            data_all[int(round(2 * event_length / 3)) :, 2, ev_no],
            None,
            env_th_r[ev_no],
            fs,
            detection["cutoffRipple"],
            detection["numSideCrossing"],
        )
        p_fr = hfo_amp_detector(
            data_all[:, 3, ev_no],
            None,
            env_th_fr[ev_no],
            fs,
            detection["cutoffFastRipple"],
            detection["numCrossing"],
        )
        p_fr_l = hfo_amp_detector(
            data_all[: int(round(event_length / 3)), 3, ev_no],
            None,
            env_th_fr[ev_no],
            fs,
            detection["cutoffFastRipple"],
            detection["numSideCrossing"],
        )
        p_fr_r = hfo_amp_detector(
            data_all[int(round(2 * event_length / 3)) :, 3, ev_no],
            None,
            env_th_fr[ev_no],
            fs,
            detection["cutoffFastRipple"],
            detection["numSideCrossing"],
        )
        if p_fr or p_r:
            if p_fr_l + p_fr_r + p_r_l + p_r_r == 0:
                accepted_osc[ev_no] = True
            else:
                accepted_osc[ev_no] = False
        else:
            accepted_osc[ev_no] = False

    data_all = data_all[:, :, accepted_osc]
    t_all = t_all[accepted_osc]
    env_th_r = env_th_r[accepted_osc]
    env_th_fr = env_th_fr[accepted_osc]

    accepted_env = np.zeros(data_all.shape[2], dtype=bool)
    for ev_no in range(data_all.shape[2]):
        acc, _ = check_centralized_component(data_all[:, 1, ev_no])
        accepted_env[ev_no] = acc

    data_all = data_all[:, :, accepted_env]
    t_all = t_all[accepted_env]
    env_th_r = env_th_r[accepted_env]
    env_th_fr = env_th_fr[accepted_env]

    if data_all.shape[2] == 0:
        return {"pool": None}

    pool = {
        "event": {
            "raw": np.squeeze(data_all[:, 0, :]),
            "hfo": np.squeeze(data_all[:, 1, :]),
            "R": np.squeeze(data_all[:, 2, :]),
            "FR": np.squeeze(data_all[:, 3, :]),
        },
        "timestamp": t_all[:, 0].astype(np.float64),
        "channelinformation": t_all[:, 1].astype(int),
        "info": {
            "envelopeSide": {"R": env_th_r, "FR": env_th_fr},
            "fs": fs,
            "file_name": file_path,
            "signal_range": "uV",
            "Number_of_channels": n,
            "montage": None,
        },
        "parameters": {
            "EventSize": detection["eventLength"],
            "Ripple_band": detection["RippleBand"],
            "FastRipple_band": detection["FastRippleBand"],
            "HFO_band": detection["HFOBand"],
        },
    }
    if pool["event"]["raw"].ndim == 1:
        pool["event"]["raw"] = pool["event"]["raw"].reshape(-1, 1)
    return {"pool": pool}
