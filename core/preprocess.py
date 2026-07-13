"""
A4 预处理模块
加载音频 → 重采样 → 单声道 → 归一化 → 分段
"""
import io
import os
import subprocess
import numpy as np
import soundfile as sf
from pathlib import Path


# 优先查找本地 ffmpeg
_FFMPEG_DIRS = [
    r"C:\ffmpeg\ffmpeg-8.0.1-essentials_build\bin",
]
_FFMPEG_PATH = None
for _d in _FFMPEG_DIRS:
    _p = os.path.join(_d, "ffmpeg.exe")
    if os.path.exists(_p):
        _FFMPEG_PATH = _d
        break
if _FFMPEG_PATH:
    os.environ["PATH"] = _FFMPEG_PATH + os.pathsep + os.environ.get("PATH", "")

SAMPLE_RATE = 16000  # Whisper 最佳输入采样率


def load_audio(path: str, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    加载音频文件，自动转单声道+目标采样率
    支持 mp3/wav/m4a/flac 等常见格式

    返回: (samples,) 归一化到 [-1, 1] 的 float32 数组
    """
    path = str(path)

    # 用 ffmpeg 解码（支持几乎所有格式）
    ffmpeg_exe = os.path.join(_FFMPEG_PATH, "ffmpeg.exe") if _FFMPEG_PATH else "ffmpeg"
    cmd = [
        ffmpeg_exe, "-i", path,
        "-f", "wav",
        "-acodec", "pcm_s16le",
        "-ac", "1",  # 单声道
        "-ar", str(sr),  # 目标采样率
        "-vn",  # 无视频
        "-y",  # 覆盖输出
        "-"
    ]

    try:
        result = subprocess.run(
            cmd, capture_output=True, check=True, timeout=300
        )
        buffer = io.BytesIO(result.stdout)
        audio, _ = sf.read(
            buffer, dtype="float32", always_2d=False
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"音频解码超时: {path}")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"音频解码失败: {path}\n{e.stderr.decode(errors='ignore')}")

    # 归一化
    max_val = np.max(np.abs(audio))
    if max_val > 1e-10:
        audio = audio / max_val

    return audio.astype(np.float32)


def get_duration(path: str) -> float:
    """获取音频时长（秒）"""
    ffprobe_exe = os.path.join(_FFMPEG_PATH, "ffprobe.exe") if _FFMPEG_PATH else "ffprobe"
    cmd = [
        ffprobe_exe, "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(result.stdout.strip())


def split_audio(audio: np.ndarray, sr: int, segment_ms: int = 30000) -> list:
    """
    将长音频按固定时长切分（用于长音频 ASR 不丢帧）
    返回 [(start_sample, end_sample, audio_segment), ...]
    """
    segment_len = int(sr * segment_ms / 1000)
    chunks = []
    for start in range(0, len(audio), segment_len):
        end = min(start + segment_len, len(audio))
        chunks.append((start, end, audio[start:end]))
    return chunks
