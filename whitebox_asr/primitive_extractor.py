"""
L2 声学基元提取层（优化版）

复用听心思路，但F0改用自相关法（比pyin快100倍+）。
"""

import numpy as np
from typing import Dict, List, Optional, Tuple

SAMPLE_RATE = 16000
N_FFT = 1024
HOP_LENGTH = 512


def _autocorrelation_f0(audio: np.ndarray, sr: int,
                         fmin: float = 60, fmax: float = 500) -> Tuple[float, bool]:
    """
    自相关法基频估计（极速版，适合干净语音）
    
    返回: (f0_hz, voiced)
    """
    n = len(audio)
    if n < sr * 0.01:  # 小于10ms
        return 0.0, False
    
    # 预加重
    audio = audio - 0.97 * np.pad(audio[:-1], (1, 0))
    
    # 加窗
    window = np.hanning(n)
    audio = audio * window
    
    # 自相关
    r = np.correlate(audio, audio, mode='full')
    r = r[n-1:]  # 取后半
    
    # 搜索范围
    min_lag = int(sr / fmax)
    max_lag = int(sr / fmin)
    min_lag = max(1, min_lag)
    max_lag = min(len(r) - 1, max_lag)
    
    if max_lag <= min_lag:
        return 0.0, False
    
    search = r[min_lag:max_lag+1]
    peak_idx = np.argmax(search)
    peak_val = search[peak_idx]
    
    # 判断有声/无声
    energy = np.sum(audio ** 2)
    threshold = energy * 0.1
    voiced = peak_val > threshold
    
    if not voiced:
        return 0.0, False
    
    lag = min_lag + peak_idx
    f0 = sr / lag if lag > 0 else 0.0
    
    return f0, True


def extract_frame_features(audio: np.ndarray, sr: int = SAMPLE_RATE) -> Dict:
    """
    逐帧提取声学特征（轻量版，不用librosa）
    
    返回: Dict of frame-level features
    """
    frame_len = int(0.025 * sr)   # 25ms
    hop = int(0.010 * sr)         # 10ms
    n_fft = min(N_FFT, frame_len)
    
    n_frames = max(1, (len(audio) - frame_len) // hop + 1)
    
    # 预分配
    rms_energy = np.zeros(n_frames)
    zcr = np.zeros(n_frames)
    f0 = np.zeros(n_frames)
    voiced = np.zeros(n_frames, dtype=bool)
    spectral_centroid = np.zeros(n_frames)
    spectral_bandwidth = np.zeros(n_frames)
    
    for i in range(n_frames):
        start = i * hop
        end = start + frame_len
        if end > len(audio):
            end = len(audio)
            start = end - frame_len
        if start < 0:
            continue
        frame = audio[start:end]
        
        # RMS能量
        rms_energy[i] = float(np.sqrt(np.mean(frame ** 2)))
        
        # 过零率
        zcr[i] = float(np.mean(np.abs(np.diff(np.sign(frame))) > 0))
        
        # FFT特征
        fft_frame = frame * np.hanning(len(frame))
        fft_abs = np.abs(np.fft.rfft(fft_frame))
        fft_len = len(fft_abs)
        freqs = np.fft.rfftfreq(len(fft_frame), d=1.0/sr)
        
        # 光谱质心
        if np.sum(fft_abs) > 1e-10:
            spectral_centroid[i] = float(np.sum(freqs * fft_abs) / np.sum(fft_abs))
        else:
            spectral_centroid[i] = 0.0
        
        # 光谱带宽
        if spectral_centroid[i] > 0 and np.sum(fft_abs) > 1e-10:
            diff = freqs - spectral_centroid[i]
            spectral_bandwidth[i] = float(np.sqrt(np.sum(diff**2 * fft_abs) / np.sum(fft_abs)))
        else:
            spectral_bandwidth[i] = 0.0
        
        # 自相关F0（每10帧算一次提速）
        if i % 10 == 0:
            f0_val, v = _autocorrelation_f0(frame, sr)
            f0[i] = f0_val
            voiced[i] = v
        else:
            f0[i] = f0[i-1] if i > 0 else 0
            voiced[i] = voiced[i-1] if i > 0 else False
    
    # 计算简单MFCC（基于DCT的近似）
    mfcc = _compute_simple_mfcc(audio, sr, n_frames, frame_len, hop)
    
    return {
        "n_frames": n_frames,
        "frame_len": frame_len,
        "hop": hop,
        "sr": sr,
        "rms_energy": rms_energy,
        "zcr": zcr,
        "f0": f0,
        "voiced": voiced,
        "spectral_centroid": spectral_centroid,
        "spectral_bandwidth": spectral_bandwidth,
        "mfcc": mfcc,
    }


def _compute_simple_mfcc(audio, sr, n_frames, frame_len, hop, n_mels=26, n_mfcc=13):
    """
    快速MFCC计算（不用librosa，纯numpy）
    """
    import numpy as np
    
    # Mel滤波器组
    low_freq = 0
    high_freq = sr / 2
    n_fft = min(512, frame_len)
    
    def hz_to_mel(hz):
        return 2595 * np.log10(1 + hz / 700.0)
    
    def mel_to_hz(mel):
        return 700 * (10 ** (mel / 2595.0) - 1)
    
    mel_points = np.linspace(hz_to_mel(low_freq), hz_to_mel(high_freq), n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    
    fbank = np.zeros((n_mels, n_fft // 2 + 1))
    for m in range(1, n_mels + 1):
        f_m_minus = bin_points[m - 1]
        f_m = bin_points[m]
        f_m_plus = bin_points[m + 1]
        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - f_m_minus) / (f_m - f_m_minus)
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (f_m_plus - k) / (f_m_plus - f_m)
    
    # 逐帧计算
    mfccs = np.zeros((n_frames, n_mfcc))
    for i in range(n_frames):
        start = i * hop
        end = start + frame_len
        if end > len(audio):
            end = len(audio)
            start = max(0, end - frame_len)
        if start < 0:
            continue
        frame = audio[start:end] * np.hanning(min(frame_len, end-start))
        
        # FFT → 功率谱
        fft = np.abs(np.fft.rfft(frame, n=n_fft)) ** 2
        
        # Mel滤波
        mel_energy = np.dot(fbank, fft)
        mel_energy = np.maximum(mel_energy, 1e-10)
        log_mel = np.log(mel_energy)
        
        # DCT
        dct_out = np.zeros(n_mfcc)
        for j in range(n_mfcc):
            dct_out[j] = np.sum(log_mel * np.cos(np.pi * j * (np.arange(n_mels) + 0.5) / n_mels))
        mfccs[i] = dct_out
    
    return mfccs.T  # (n_mfcc, n_frames)


def extract_segment_features(frames: Dict) -> Dict:
    """
    片段特征聚合
    
    输入: extract_frame_features的输出
    返回: 聚合后的特征向量
    """
    n = frames["n_frames"]
    if n == 0:
        return {}
    
    sr = frames["sr"]
    rms = frames["rms_energy"]
    f0 = frames["f0"]
    voiced = frames["voiced"]
    centroid = frames["spectral_centroid"]
    bandwidth = frames["spectral_bandwidth"]
    zcr = frames["zcr"]
    
    # 能量特征
    energy_mean = float(np.mean(rms))
    energy_std = float(np.std(rms))
    energy_slope = float(np.polyfit(np.arange(n), rms, 1)[0]) if n > 2 else 0.0
    
    # 能量包络阶跃检测（用于爆破音）
    energy_diff = np.diff(rms)
    max_jump = float(np.max(energy_diff)) if len(energy_diff) > 0 else 0.0
    max_jump_idx = int(np.argmax(energy_diff)) if len(energy_diff) > 0 else 0
    
    # F0特征
    voiced_f0 = f0[voiced]
    pitch_mean = float(np.mean(voiced_f0)) if len(voiced_f0) > 0 else 0.0
    pitch_std = float(np.std(voiced_f0)) if len(voiced_f0) > 0 else 0.0
    pitch_contour = f0.copy()  # 完整F0轨迹（给声调用）
    
    # 光谱特征
    centroid_mean = float(np.mean(centroid))
    centroid_std = float(np.std(centroid))
    bandwidth_mean = float(np.mean(bandwidth))
    
    # 高频能量占比（>4kHz）
    # 用光谱质心近似：质心高 → 高频能量多
    high_freq_ratio = float(np.mean(centroid > 4000)) if centroid_mean > 0 else 0.0
    
    # 过零率特征
    zcr_mean = float(np.mean(zcr))
    zcr_std = float(np.std(zcr))
    
    # MFCC近似（用光谱特征简化）
    # 真正的MFCC用librosa，这里用几维谱特征近似
    mfcc_approx = [
        float(np.mean(centroid)) / 4000,           # 近似MFCC1
        float(np.mean(bandwidth)) / 4000,           # 近似MFCC2
        float(high_freq_ratio),                     # 近似MFCC3
        float(np.mean(zcr)) * 10,                   # 近似MFCC4
    ]
    
    return {
        "n_frames": n,
        "duration_s": n * frames["hop"] / sr,
        "energy_mean": energy_mean,
        "energy_std": energy_std,
        "energy_slope": energy_slope,
        "energy_max_jump": max_jump,
        "energy_max_jump_idx": max_jump_idx,
        "rms_envelope": rms.tolist(),
        "pitch_mean": pitch_mean,
        "pitch_std": pitch_std,
        "pitch_contour": pitch_contour,
        "voiced_ratio": float(np.mean(voiced)),
        "spectral_centroid_mean": centroid_mean,
        "spectral_centroid_std": centroid_std,
        "spectral_bandwidth_mean": bandwidth_mean,
        "high_freq_ratio": high_freq_ratio,
        "zcr_mean": zcr_mean,
        "zcr_std": zcr_std,
        "mfcc_approx": mfcc_approx,
    }
