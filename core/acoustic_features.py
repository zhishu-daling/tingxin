"""
听心 · 声学基元提取器 (A1 + A2)

从每个音频片段提取多维声学特征，用于：
- 说话人区分（声纹特征）
- 情感分析（能量/语速/音高变化）
- 事件检测（笑声/停顿等）
"""
import numpy as np
import librosa
from scipy import signal


def extract_mfcc(audio: np.ndarray, sr: int, n_mfcc: int = 13) -> np.ndarray:
    """MFCC - 声道形状特征"""
    mfcc = librosa.feature.mfcc(
        y=audio, sr=sr, n_mfcc=n_mfcc,
        n_fft=1024, hop_length=512
    )
    return mfcc  # (n_mfcc, n_frames)


def extract_pitch(audio: np.ndarray, sr: int) -> tuple:
    """
    基频 F0 估计
    返回: (f0_hz: np.ndarray, voiced_flag: np.ndarray)
    """
    f0, voiced_flag, _ = librosa.pyin(
        y=audio, fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C7'), sr=sr,
        fill_na=0.0
    )
    return f0, voiced_flag


def extract_spectral_features(audio: np.ndarray, sr: int) -> dict:
    """光谱特征集"""
    return {
        "centroid": librosa.feature.spectral_centroid(y=audio, sr=sr)[0],
        "bandwidth": librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0],
        "rolloff": librosa.feature.spectral_rolloff(y=audio, sr=sr)[0],
        "contrast": np.mean(librosa.feature.spectral_contrast(y=audio, sr=sr), axis=0),
    }


def extract_energy(audio: np.ndarray, sr: int) -> np.ndarray:
    """RMS 能量包络"""
    return librosa.feature.rms(y=audio)[0]


def extract_zcr(audio: np.ndarray) -> np.ndarray:
    """过零率 - 噪音/声门特征"""
    return librosa.feature.zero_crossing_rate(y=audio)[0]


def extract_formants(audio: np.ndarray, sr: int, n_formants: int = 4) -> np.ndarray:
    """
    共振峰估计（对中频段做 LPC 分析）
    不同声道形状产生不同共振峰分布 -> 说话人格印
    """
    # LPC 阶数 = 采样率/1000 + 2 的经验公式
    order = sr // 1000 + 2
    lpc_coeffs = librosa.lpc(audio, order=order)

    # 寻找共振峰（LPC 极点频率）
    roots = np.roots(np.r_[1, lpc_coeffs[1:]])
    roots = roots[np.abs(roots) < 1.0]  # 稳定极点

    angles = np.angle(roots)
    freqs = angles * sr / (2 * np.pi)
    freqs = freqs[freqs > 0]  # 正频率

    # 取前 n_formants 个
    return np.sort(freqs)[:n_formants]


def extract_jitter(audio: np.ndarray, sr: int, f0: np.ndarray = None, voiced: np.ndarray = None) -> dict:
    """
    提取基频微抖动特征（不自主喉部肌肉信号）
    训练有素的骗子和天生沉稳者也无法完全抑制。

    参数:
        audio: 音频信号
        sr: 采样率
        f0: 可选，已计算的基频数组。为None时自动调用extract_pitch
        voiced: 可选，已计算的有声段标记。为None时自动调用extract_pitch

    返回:
    {
        "jitter_ppq5": float -> 五周期基频扰动系数
        "jitter_rap": float -> 三周期基频扰动系数
        "shimmer_db": float -> 振幅扰动（dB）
        "shimmer_apq5": float -> 五周期振幅扰动系数
    }
    """
    if f0 is None or voiced is None:
        f0, voiced = extract_pitch(audio, sr)
    if np.sum(voiced) < 10:
        return {"jitter_ppq5": 0.0, "jitter_rap": 0.0,
                "shimmer_db": 0.0, "shimmer_apq5": 0.0}

    pitch_vals = f0[voiced]

    # Jitter: 基频周期偏差
    periods = sr / (pitch_vals + 1e-10)
    diffs = np.abs(np.diff(periods))
    if len(diffs) < 5:
        jitter_ppq5 = 0.0
        jitter_rap = 0.0
    else:
        # 五周期扰动（PPQ5）：每5个周期的平均偏差
        ppq5 = np.array([
            np.mean(np.abs(periods[i:i+5] - np.mean(periods[i:i+5])))
            for i in range(len(periods)-4)
        ])
        jitter_ppq5 = float(np.mean(ppq5) / (np.mean(periods) + 1e-10))
        # 三周期扰动（RAP）
        rap = np.array([
            np.abs(periods[i+1] - (periods[i] + periods[i+2])/2)
            for i in range(len(periods)-2)
        ])
        jitter_rap = float(np.mean(rap) / (np.mean(periods) + 1e-10))

    # Shimmer: 振幅扰动（基于能量包络）
    energy = extract_energy(audio, sr)
    # 重采样到跟pitch帧数对齐
    from scipy import signal as _signal
    energy_resampled = _signal.resample(energy, len(pitch_vals))
    amp_vals = np.sqrt(np.maximum(energy_resampled, 1e-15))

    if len(amp_vals) < 5:
        shimmer_db = 0.0
        shimmer_apq5 = 0.0
    else:
        # shimmer in dB
        amp_db = 20 * np.log10(amp_vals + 1e-15)
        shimmer_db = float(np.std(amp_db))
        # 五周期振幅扰动
        apq5 = np.array([
            np.mean(np.abs(amp_vals[i:i+5] - np.mean(amp_vals[i:i+5])))
            for i in range(len(amp_vals)-4)
        ])
        shimmer_apq5 = float(np.mean(apq5) / (np.mean(amp_vals) + 1e-10))

    return {
        "jitter_ppq5": round(jitter_ppq5, 6),
        "jitter_rap": round(jitter_rap, 6),
        "shimmer_db": round(shimmer_db, 4),
        "shimmer_apq5": round(shimmer_apq5, 6),
    }


def extract_decay_time(audio: np.ndarray, sr: int) -> float:
    """
    句尾能量衰减时间。
    从末尾能量峰值90%下降到10%的时间（秒）。
    急停=短，自然释放=适中。
    """
    n = len(audio)
    if n < sr // 5:  # 小于200ms不分析
        return 0.0

    energy = np.abs(audio)
    # 末尾25%区域找衰减
    tail = energy[3*n//4:]
    if len(tail) < sr // 20:
        return 0.0

    # 平滑
    from scipy.ndimage import uniform_filter1d
    smooth = uniform_filter1d(tail, size=max(1, sr//500))

    peak = np.max(smooth)
    if peak < 1e-10:
        return 0.0

    valley = np.min(smooth[len(smooth)//2:])  # 后半段最小值
    if valley < 0:
        valley = 0

    threshold_90 = peak * 0.9
    threshold_10 = valley + (peak - valley) * 0.1

    # 从后往前找穿越点
    cross_90 = None
    cross_10 = None
    for i in range(len(smooth)-1, -1, -1):
        if cross_10 is None and smooth[i] <= threshold_10:
            cross_10 = i
        if cross_90 is None and smooth[i] >= threshold_90:
            cross_90 = i
            break

    if cross_90 is None or cross_10 is None:
        return 0.0

    decay_samples = cross_10 - cross_90
    # 采样率折算: tail是25%原音频，需对应原始采样
    # tail = audio[3n/4:], 帧数是 len(tail)
    # 每个smooth索引对应 tail 中的一个采样
    decay_time_s = max(0, decay_samples) / sr
    return round(decay_time_s, 4)


def detect_micro_pause_end(audio: np.ndarray, sr: int,
                           window_ms: int = 250) -> dict:
    """
    检测句尾微停顿特征。
    自然说话在句尾会有50-200ms的放松停顿。
    伪装者会消除这个停顿或进入"完美沉默"

    返回:
    {
        "has_micro_pause": bool,
        "pause_duration_ms": float,
        "post_speech_energy_drop_db": float,  # 句尾能量骤降幅
    }
    """
    n = len(audio)
    if n < sr // 10:
        return {"has_micro_pause": False, "pause_duration_ms": 0.0,
                "post_speech_energy_drop_db": 0.0}

    # 句尾最后window_ms段
    window_len = int(sr * window_ms / 1000)
    tail = audio[max(0, n-window_len):]

    energy = np.abs(tail)
    from scipy.ndimage import uniform_filter1d
    smooth = uniform_filter1d(energy, size=max(1, sr//1000))  # 1ms平滑

    # 找能量快速下降到接近0的点
    mean_energy = np.mean(smooth)
    tail_end = smooth[-min(len(smooth)//4, 50):]
    end_mean = np.mean(tail_end)

    # 能量骤降幅度
    peak_tail = np.max(smooth)
    drop_db = 20 * np.log10(
        (peak_tail + 1e-10) / (end_mean + 1e-10)
    ) if end_mean > 1e-15 else 40.0

    # 找第一个连续静音段（能量<10%峰值）
    silent_thresh = peak_tail * 0.1 if peak_tail > 1e-10 else 1e-10
    silent_start = None
    for i in range(len(smooth)):
        if smooth[i] < silent_thresh:
            if silent_start is None:
                silent_start = i
        else:
            silent_start = None

    pause_ms = 0.0
    has_pause = False
    if silent_start is not None:
        pause_samples = len(smooth) - silent_start
        pause_ms = pause_samples / sr * 1000
        has_pause = 50 <= pause_ms <= 200  # 自然停顿范围

    return {
        "has_micro_pause": has_pause,
        "pause_duration_ms": round(pause_ms, 1),
        "post_speech_energy_drop_db": round(drop_db, 2),
    }


def extract_segment_features(audio: np.ndarray, sr: int) -> dict:
    """
    对一个音频片段提取完整的说话人/情感/伪装特征向量

    返回 dict:
    {
        "mfcc_mean": (13,) -> 声道形状
        "mfcc_std": (13,) -> 声道变化
        "pitch_mean": float -> 平均音高
        "pitch_std": float -> 音高变化
        "spectral_centroid_mean": float -> 音色亮度
        "energy_mean": float -> 平均能量
        "energy_std": float -> 能量变化
        "zcr_mean": float -> 噪音特征
        "formants": (n_formants,) -> 共振峰
        "speaking_rate": float -> 语速（每秒音节数近似）
        # 新增：伪装检测基元
        "jitter_ppq5": float -> 五周期基频扰动
        "jitter_rap": float -> 三周期基频扰动
        "shimmer_db": float -> 振幅扰动
        "shimmer_apq5": float -> 五周期振幅扰动
        "decay_time_s": float -> 句尾能量衰减时间
        "has_micro_pause": bool -> 句尾是否有自然微停顿
        "pause_duration_ms": float -> 停顿时长
        "post_speech_energy_drop_db": float -> 句尾能量骤降幅度
    }
    """
    if len(audio) < sr // 10:  # 太短的片段不分析
        return {}

    mfcc = extract_mfcc(audio, sr)
    f0, voiced = extract_pitch(audio, sr)
    spectral = extract_spectral_features(audio, sr)
    energy = extract_energy(audio, sr)
    zcr = extract_zcr(audio)
    formants = extract_formants(audio, sr)

    # 有效音高
    pitch_vals = f0[voiced] if np.any(voiced) else np.array([0.0])

    # 伪装检测基元（复用已算好的基频，避免重复pyin）
    jitter = extract_jitter(audio, sr, f0=f0, voiced=voiced)
    decay = extract_decay_time(audio, sr)
    micro = detect_micro_pause_end(audio, sr)

    # 打包
    features = {
        "mfcc_mean": np.mean(mfcc, axis=1).tolist(),
        "mfcc_std": np.std(mfcc, axis=1).tolist(),
        "pitch_mean": float(np.mean(pitch_vals)) if len(pitch_vals) > 0 else 0.0,
        "pitch_std": float(np.std(pitch_vals)) if len(pitch_vals) > 0 else 0.0,
        "spectral_centroid_mean": float(np.mean(spectral["centroid"])),
        "energy_mean": float(np.mean(energy)),
        "energy_std": float(np.std(energy)),
        "zcr_mean": float(np.mean(zcr)),
        "formants": formants.tolist(),
        # 伪装检测
        "jitter_ppq5": jitter["jitter_ppq5"],
        "jitter_rap": jitter["jitter_rap"],
        "shimmer_db": jitter["shimmer_db"],
        "shimmer_apq5": jitter["shimmer_apq5"],
        "decay_time_s": decay,
        "has_micro_pause": micro["has_micro_pause"],
        "pause_duration_ms": micro["pause_duration_ms"],
        "post_speech_energy_drop_db": micro["post_speech_energy_drop_db"],
    }

    return features


def segment_distance(feat_a: dict, feat_b: dict) -> float:
    """
    计算两个音频片段的声学距离
    越小越可能是同一个人
    """
    from scipy.spatial.distance import cosine, euclidean

    weights = {
        "mfcc_mean": 0.4,     # 声道形状最重要
        "pitch_mean": 0.2,    # 音高
        "pitch_std": 0.05,
        "spectral_centroid_mean": 0.1,
        "energy_mean": 0.05,
        "zcr_mean": 0.05,
        "formants": 0.15,     # 共振峰
    }

    dist = 0.0
    for key, weight in weights.items():
        if key not in feat_a or key not in feat_b:
            continue
        va, vb = feat_a[key], feat_b[key]
        if isinstance(va, list):
            va, vb = np.array(va), np.array(vb)
            d = cosine(va, vb) if len(va) > 1 and np.linalg.norm(va) > 0 and np.linalg.norm(vb) > 0 else euclidean(va, vb)
        else:
            d = abs(va - vb) / (abs(va) + abs(vb) + 1e-10)
        dist += weight * d

    return dist
