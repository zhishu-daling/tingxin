"""
L1 音频预处理层 + 音节切分

VAD切出有声段后，再用量子onset检测切分为音节级片段。
"""

import numpy as np
import sys, os

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_TINGXIN_DIR = os.path.dirname(_THIS_DIR)
if _TINGXIN_DIR not in sys.path:
    sys.path.insert(0, _TINGXIN_DIR)

from core.preprocess import load_audio, SAMPLE_RATE


def _spectral_flux_onsets(audio: np.ndarray, sr: int) -> np.ndarray:
    """
    基于频谱通量（spectral flux）的音节onset检测
    
    思路：相邻帧频谱的正变化量累积 → 找峰值 → 就是音节起始点
    
    返回: onset时间点（秒）的array
    """
    frame_len = int(0.025 * sr)   # 25ms
    hop = int(0.010 * sr)         # 10ms
    
    n_frames = max(1, (len(audio) - frame_len) // hop + 1)
    
    # 计算每帧的频谱
    flux = np.zeros(n_frames)
    prev_spectrum = None
    
    for i in range(n_frames):
        start = i * hop
        end = min(start + frame_len, len(audio))
        if end - start < frame_len // 2:
            break
        frame = audio[start:end] * np.hanning(end - start)
        
        # FFT幅值谱
        spectrum = np.abs(np.fft.rfft(frame))
        
        if prev_spectrum is not None:
            # 正变化量之和 = 频谱通量
            diff = spectrum - prev_spectrum
            flux[i] = np.sum(diff[diff > 0]) / max(np.sum(spectrum), 1e-10)
        
        prev_spectrum = spectrum
    
    # 找flux峰值（要做自适应阈值）
    if len(flux) < 3:
        return np.array([])
    
    # 平滑
    from scipy.ndimage import uniform_filter1d
    flux_smooth = uniform_filter1d(flux, size=3)
    
    # 自适应阈值：均值 + 0.8 * 标准差（提高阈值，减少过切）
    threshold = np.mean(flux_smooth) + 0.8 * np.std(flux_smooth)
    
    # 找局部峰值
    onsets = []
    for i in range(2, len(flux_smooth) - 2):
        if (flux_smooth[i] > threshold and
            flux_smooth[i] > flux_smooth[i-1] and
            flux_smooth[i] > flux_smooth[i-2] and
            flux_smooth[i] >= flux_smooth[i+1] and
            flux_smooth[i] >= flux_smooth[i+2]):
            # 两个onset之间至少间隔150ms（防止过切）
            t = (i * hop) / sr
            if not onsets or t - onsets[-1] > 0.15:
                onsets.append(t)
    
    return np.array(onsets)


def _vad_simple(audio: np.ndarray, sr: int,
                silence_thresh_db: float = -35,
                min_silence_ms: int = 200,
                min_voice_ms: int = 100) -> list:
    """
    简化的VAD：按能量切出有声段
    比听心的_silence_segments更快
    """
    frame_len = int(0.030 * sr)  # 30ms
    hop = frame_len
    n_frames = max(1, (len(audio) - frame_len) // hop + 1)
    
    rms = np.array([
        np.sqrt(np.mean(audio[i*hop:i*hop+frame_len]**2))
        for i in range(n_frames)
    ])
    rms_db = 20 * np.log10(np.maximum(rms, 1e-10))
    is_voice = rms_db > silence_thresh_db
    
    # 状态机合并
    segments = []
    i = 0
    while i < n_frames:
        if is_voice[i]:
            start = i * hop / sr
            while i < n_frames and is_voice[i]:
                i += 1
            end = i * hop / sr
            dur = end - start
            if dur >= min_voice_ms / 1000:
                segments.append({
                    "start_s": round(start, 3),
                    "end_s": round(end, 3),
                    "duration_s": round(dur, 3),
                })
        else:
            i += 1
    
    return segments


def segment_audio(audio_path: str, sr: int = SAMPLE_RATE) -> dict:
    """
    完整音频分割管道：
    1. 加载+重采样
    2. VAD切出有声段
    3. 每个有声段内做onset检测 → 切分为音节级片段
    
    返回:
    {
        "file": str,
        "sr": int,
        "duration_s": float,
        "voice_segments": [...],    # VAD有声段
        "syllables": [              # 音节级片段
            {
                "index": int,
                "start_s": float,
                "end_s": float,
                "duration_s": float,
                "audio": np.ndarray,
            }
        ]
    }
    """
    audio_path = str(audio_path)
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"音频文件不存在: {audio_path}")
    
    audio = load_audio(audio_path, sr=sr)
    duration_s = len(audio) / sr
    
    # VAD
    voice_segments = _vad_simple(audio, sr)
    
    # 每个有声段内做onset检测 → 切音节
    syllables = []
    MIN_SYLLABLE_DUR = 0.08  # 最短音节80ms（短于此合并）
    MAX_SYLLABLE_DUR = 0.45  # 最长音节450ms（长于此再切）
    for vs in voice_segments:
        start_s = vs["start_s"]
        end_s = vs["end_s"]
        start_sample = int(start_s * sr)
        end_sample = int(min(end_s * sr, len(audio)))
        seg_audio = audio[start_sample:end_sample].copy()
        
        # 检测onset
        onsets = _spectral_flux_onsets(seg_audio, sr)
        
        if len(onsets) == 0:
            # 没有检测到onset → 整段当一个音节
            syllables.append({
                "index": len(syllables),
                "start_s": start_s,
                "end_s": end_s,
                "duration_s": end_s - start_s,
                "audio": seg_audio,
            })
        else:
            # 在onset处切分
            cut_points = [0.0] + onsets.tolist() + [end_s - start_s]
            raw_segments = []
            for j in range(len(cut_points) - 1):
                seg_start = start_s + cut_points[j]
                seg_end = start_s + cut_points[j + 1]
                dur = seg_end - seg_start
                if dur < 0.02:  # 太短直接跳过
                    continue
                s_sample = int(seg_start * sr) - start_sample
                s_sample = max(0, s_sample)
                e_sample = int(seg_end * sr) - start_sample
                e_sample = min(e_sample, len(seg_audio))
                syl_audio = seg_audio[s_sample:e_sample].copy()
                raw_segments.append({
                    "start_s": seg_start,
                    "end_s": seg_end,
                    "duration_s": dur,
                    "audio": syl_audio,
                })
            
            # 合并过短的片段
            merged = []
            for seg in raw_segments:
                if not merged:
                    merged.append(seg)
                elif seg["duration_s"] < MIN_SYLLABLE_DUR or merged[-1]["duration_s"] < MIN_SYLLABLE_DUR:
                    # 当前段或前一段太短 → 合并
                    merged[-1]["end_s"] = seg["end_s"]
                    merged[-1]["duration_s"] = merged[-1]["end_s"] - merged[-1]["start_s"]
                    # 合并audio
                    merged[-1]["audio"] = np.concatenate([merged[-1]["audio"], seg["audio"]])
                else:
                    merged.append(seg)
            
            for seg in merged:
                if seg["duration_s"] < MIN_SYLLABLE_DUR * 0.5:
                    continue  # 仍然太短就丢弃
                syllables.append({
                    "index": len(syllables),
                    "start_s": round(seg["start_s"], 3),
                    "end_s": round(seg["end_s"], 3),
                    "duration_s": round(seg["duration_s"], 3),
                    "audio": seg["audio"],
                })
    
    return {
        "file": os.path.abspath(audio_path),
        "sr": sr,
        "duration_s": duration_s,
        "voice_segments": voice_segments,
        "syllables": syllables,
    }


# 兼容旧接口
def load_and_vad(audio_path, sr=SAMPLE_RATE):
    result = segment_audio(audio_path, sr)
    # 转成旧格式
    return {
        "file": result["file"],
        "sr": result["sr"],
        "duration_s": result["duration_s"],
        "segments": result["syllables"],
        "vad_details": result["voice_segments"],
    }

def load_and_vad_from_array(audio, sr=SAMPLE_RATE):
    """简化的数组版本（只做VAD，不做onset切片）"""
    duration_s = len(audio) / sr
    voice_segments = _vad_simple(audio, sr)
    segments = []
    for i, vs in enumerate(voice_segments):
        ss = int(vs["start_s"] * sr)
        se = int(min(vs["end_s"] * sr, len(audio)))
        segments.append({
            "index": i,
            "start_s": vs["start_s"],
            "end_s": vs["end_s"],
            "duration_s": vs["duration_s"],
            "audio": audio[ss:se].copy(),
        })
    return {
        "file": None, "sr": sr, "duration_s": duration_s,
        "segments": segments, "vad_details": voice_segments,
    }
