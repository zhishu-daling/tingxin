"""
C1.3 说话人切换检测（重构版）

基于昨晚验证过的 MFCC + K-Means 方案
增量模式：每来一批新片段，对全部已收集片段重新聚类
"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.fft import dct

from .preprocess import load_audio, SAMPLE_RATE


def _mel_filterbank(n_mels, n_freqs, sr):
    def hz_to_mel(f): return 2595 * np.log10(1 + f / 700)
    def mel_to_hz(m): return 700 * (10**(m / 2595) - 1)
    mel_min, mel_max = hz_to_mel(0), hz_to_mel(sr / 2)
    mel_pts = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_pts = mel_to_hz(mel_pts)
    bins = np.floor((n_freqs - 1) * hz_pts / (sr / 2)).astype(int)
    filters = np.zeros((n_mels, n_freqs))
    for m in range(1, n_mels + 1):
        for k in range(bins[m - 1], bins[m + 1]):
            if k < bins[m]:
                filters[m - 1, k] = (k - bins[m - 1]) / (bins[m] - bins[m - 1])
            elif k == bins[m]:
                filters[m - 1, k] = 1.0
            else:
                filters[m - 1, k] = (bins[m + 1] - k) / (bins[m + 1] - bins[m])
    return filters


def compute_mfcc_mean(audio_segment, sr=16000, n_mfcc=13):
    """对一段音频计算MFCC均值向量"""
    if len(audio_segment) < sr * 0.05:
        return None
    x = np.append(audio_segment[0], audio_segment[1:] - 0.97 * audio_segment[:-1])
    frame_size, hop = int(0.025 * sr), int(0.010 * sr)
    frames = []
    for i in range(0, len(x) - frame_size, hop):
        frame = x[i:i+frame_size] * np.hanning(frame_size)
        frames.append(frame)
    frames = np.array(frames)
    if len(frames) == 0:
        return None
    fft_size = 512; n_mels = 26
    mel_w = _mel_filterbank(n_mels, fft_size // 2 + 1, sr)
    mfccs = []
    for frame in frames:
        spec = np.abs(np.fft.rfft(frame, n=fft_size))**2
        mel_spec = mel_w @ spec + 1e-10
        log_mel = np.log(mel_spec)
        mfcc = dct(log_mel, type=2, norm='ortho')[:n_mfcc]
        mfccs.append(mfcc)
    return np.mean(mfccs, axis=0) if mfccs else None


def cluster_speakers(segments, audio_data, sr=SAMPLE_RATE, n_speakers=2):
    """
    对片段列表做 K-Means 说话人聚类
    返回: [{start, end, text, speaker}, ...]
    """
    if len(segments) < 2:
        for s in segments: s["speaker"] = "SPK1"
        return segments

    # 提取 MFCC 均值
    features = []
    valid_indices = []
    for i, seg in enumerate(segments):
        s = int(seg["start"] * sr)
        e = int(seg["end"] * sr)
        s = max(0, s); e = min(len(audio_data), e)
        chunk = audio_data[s:e]
        mfcc = compute_mfcc_mean(chunk, sr)
        if mfcc is not None:
            features.append(mfcc)
            valid_indices.append(i)

    if len(features) < 2:
        for s in segments: s["speaker"] = "SPK1"
        return segments

    # 标准化 + K-Means
    feat_mat = np.array(features)
    scaler = StandardScaler()
    feat_scaled = scaler.fit_transform(feat_mat)

    k = min(n_speakers, len(features))
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(feat_scaled)

    # 重排序（SPK1 = 第一个说话人）
    order = {}
    for i, idx in enumerate(valid_indices):
        l = labels[i]
        if l not in order:
            order[l] = f"SPK{len(order)+1}"
        segments[idx]["speaker"] = order[l]

    return segments


def diarize(segments, audio_path, target_sr=SAMPLE_RATE):
    """批处理：加载音频 → MFCC → K-Means → 说话人标签"""
    audio = load_audio(audio_path, sr=target_sr)
    return _merge_consecutive(cluster_speakers(segments, audio, target_sr, n_speakers=2))


def _merge_consecutive(segments):
    if not segments: return []
    merged = [dict(segments[0])]
    for seg in segments[1:]:
        if seg["speaker"] == merged[-1]["speaker"]:
            merged[-1]["end"] = seg["end"]
            merged[-1]["text"] += seg["text"]
        else:
            merged.append(dict(seg))
    return merged


# ===== 增量式（兼容旧接口）=====
class IncrementalDiarizer:
    """
    基于MFCC的增量说话人分离器
    每来一批新片段 -> 对全部积累片段做 K-Means -> 刷新所有标签
    """
    def __init__(self, n_speakers=2, threshold=0.3):
        self.n_speakers = n_speakers
        self._audio_data = None
        self._sr = SAMPLE_RATE

    def set_audio(self, audio):
        self._audio_data = audio

    def add_segment(self, segment, features):
        """新增一个片段（兼容旧接口，实际在refine时统一聚类）"""
        return None

    def refine(self, all_segments):
        """对所有已积累片段做聚类"""
        if self._audio_data is None or len(all_segments) < 2:
            return all_segments
        return cluster_speakers(all_segments, self._audio_data, self._sr, self.n_speakers)

    def batch_refine(self, segments, audio_path):
        """完全体精校"""
        return diarize(segments, audio_path, self._sr)
