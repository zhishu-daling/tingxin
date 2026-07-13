"""
听心 · 声音基元 (A0)

不经解码、不经分析，只输出原始声音基元的 JSON。
- 无 ASR 转写
- 无情绪分析
- 无说话人分离
- 无音乐/风格识别
- 只有原始声学基元的提取与打包

支持单文件和批量处理。
"""
import json
import os
import time
import numpy as np
from pathlib import Path

from .preprocess import load_audio, get_duration, SAMPLE_RATE


# ========== 原始基元提取 ==========

def _silence_segments(audio: np.ndarray, sr: int,
                      frame_ms: int = 30,
                      silence_thresh_db: float = -40,
                      min_silence_ms: int = 300,
                      min_voice_ms: int = 200) -> list:
    """
    纯能量阈值分割，将音频切分为[静音段, 有声段]交替。

    参数:
        audio: 原始波形
        sr: 采样率
        frame_ms: 帧长(ms)
        silence_thresh_db: 静音阈值 dB
        min_silence_ms: 最小静音长度(ms)，低于此不切
        min_voice_ms: 最小有声长度(ms)，低于此合并进静音

    返回: [{"type":"silence","start_s":...,"end_s":...,"duration_s":...},
            {"type":"voice","start_s":...,"end_s":...,"duration_s":...}, ...]
    """
    frame_len = int(sr * frame_ms / 1000)
    hop = frame_len
    n_frames = (len(audio) - frame_len) // hop + 1 if len(audio) >= frame_len else 0

    if n_frames < 1:
        return [{"type": "silence" if len(audio) == 0 else "voice",
                 "start_s": 0.0, "end_s": len(audio) / sr, "duration_s": len(audio) / sr}]

    # 每帧 RMS dB
    rms = np.array([
        np.sqrt(np.mean(audio[i * hop:i * hop + frame_len] ** 2))
        for i in range(n_frames)
    ])
    rms_db = 20 * np.log10(np.maximum(rms, 1e-10))

    is_voice = rms_db > silence_thresh_db

    # 状态机：合并相邻帧
    segments = []
    cur_type = "voice" if is_voice[0] else "silence"
    cur_start = 0.0
    for i in range(1, n_frames):
        ft = "voice" if is_voice[i] else "silence"
        if ft != cur_type:
            seg_end = i * hop / sr
            seg_dur = seg_end - cur_start
            if seg_dur >= min_voice_ms / 1000 or cur_type == "voice":
                segments.append({"type": cur_type, "start_s": round(cur_start, 3),
                                 "end_s": round(seg_end, 3), "duration_s": round(seg_dur, 3)})
                cur_type = ft
                cur_start = seg_end

    # 最后一段
    final_end = len(audio) / sr
    final_dur = final_end - cur_start
    segments.append({"type": cur_type, "start_s": round(cur_start, 3),
                     "end_s": round(final_end, 3), "duration_s": round(final_dur, 3)})

    # 合并短段
    merged = []
    for seg in segments:
        if seg["duration_s"] < min_silence_ms / 1000 and seg["type"] == "silence" and merged:
            # 静音太短 → 合并到前一段
            merged[-1]["end_s"] = seg["end_s"]
            merged[-1]["duration_s"] = round(merged[-1]["end_s"] - merged[-1]["start_s"], 3)
            continue
        if seg["duration_s"] < min_voice_ms / 1000 and seg["type"] == "voice" and merged:
            if merged[-1]["type"] == "silence":
                merged[-1]["end_s"] = seg["end_s"]
                merged[-1]["duration_s"] = round(merged[-1]["end_s"] - merged[-1]["start_s"], 3)
                continue
        merged.append(seg)

    return merged


def _extract_primitive(audio_segment: np.ndarray, sr: int) -> dict:
    """
    对一段音频提取原始基元：
    - 幅度统计 (max, min, rms, peak_to_peak)
    - 过零率
    - 频谱质心 (原始值)
    - 能量包络
    - FFT 幅值谱（缩放到固定 bin 数）
    """
    n = len(audio_segment)
    if n < 32:
        return {"samples": int(n), "duration_s": round(n / sr, 4),
                "amplitude": None, "zcr": None, "energy": None, "fft_bins": None}

    # 幅度
    amp = {
        "max": float(np.max(audio_segment)),
        "min": float(np.min(audio_segment)),
        "rms": float(np.sqrt(np.mean(audio_segment ** 2))),
        "peak_to_peak": float(np.ptp(audio_segment)),
    }

    # 过零率
    zcr = float(np.mean(np.abs(np.diff(np.sign(audio_segment))) > 0))

    # RMS 能量包络（按 10ms 帧）
    frame_len = int(sr * 0.01)  # 10ms
    if frame_len < 1:
        energy_envelope = [amp["rms"]]
    else:
        n_frames = max(1, n // frame_len)
        energy_envelope = [
            float(np.sqrt(np.mean(audio_segment[i * frame_len:(i + 1) * frame_len] ** 2)))
            for i in range(n_frames)
        ]

    # 单帧 FFT → 前 128 个 bins（fft 归一化幅值谱）
    fft_spectrum = np.abs(np.fft.rfft(audio_segment))
    fft_bins = fft_spectrum[:128].tolist()  # 低频为主

    # 频谱质心（原始计算，非 librosa）
    freqs = np.fft.rfftfreq(n, 1 / sr)[:128]
    if np.sum(fft_bins) > 1e-10:
        spectral_centroid = float(np.sum(freqs * fft_bins) / np.sum(fft_bins))
    else:
        spectral_centroid = 0.0

    return {
        "samples": int(n),
        "duration_s": round(n / sr, 4),
        "amplitude": amp,
        "zcr": round(zcr, 6),
        "energy_rms_envelope": [round(e, 6) for e in energy_envelope],
        "spectral_centroid_hz": round(spectral_centroid, 2),
        "fft_bins_128": [round(float(b), 6) for b in fft_bins],
    }


# ========== 对外接口 ==========

def extract_primitives(audio_path: str) -> dict:
    """
    单文件声音基元提取。

    返回 JSON 结构:
    {
        "file": "xxx.wav",
        "filename": "xxx.wav",
        "format": {"duration_s": ..., "sample_rate": 16000, "samples": ...,
                    "channels": 1, "bits_per_sample": 16},
        "silence_segments": [...],   // 纯能量分割
        "voice_segments": [...],     // 仅有声段
        "primitives": [              // 按有声段为单位的基元
            {"segment_index": 0, "start_s": ..., "end_s": ..., ...基元字段...},
            ...
        ],
        "global_stats": {            // 全局统计
            "max_amplitude": ..., "rms_amplitude": ...,
            "dynamic_range_db": ..., "voice_ratio": ...
        }
    }
    """
    audio_path = str(audio_path)
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"文件不存在: {audio_path}")

    t0 = time.time()

    # 加载
    audio = load_audio(audio_path)
    duration = len(audio) / SAMPLE_RATE

    # 能量分割
    segs = _silence_segments(audio, SAMPLE_RATE)
    voice_segs = [s for s in segs if s["type"] == "voice"]

    # 对每个有声段提取基元
    primitives = []
    for i, vs in enumerate(voice_segs):
        start_sample = int(vs["start_s"] * SAMPLE_RATE)
        end_sample = int(min(vs["end_s"] * SAMPLE_RATE, len(audio)))
        seg_audio = audio[start_sample:end_sample]
        prim = _extract_primitive(seg_audio, SAMPLE_RATE)
        prim["segment_index"] = i
        prim["start_s"] = vs["start_s"]
        prim["end_s"] = vs["end_s"]
        primitives.append(prim)

    # 全局统计
    rms_all = np.sqrt(np.mean(audio ** 2))
    db_max = 20 * np.log10(np.max(np.abs(audio)) + 1e-10)
    db_min = 20 * np.log10(rms_all + 1e-10)
    voice_dur = sum(v["duration_s"] for v in voice_segs)

    global_stats = {
        "max_amplitude": round(float(np.max(np.abs(audio))), 6),
        "rms_amplitude": round(float(rms_all), 6),
        "dynamic_range_db": round(float(db_max - db_min), 2),
        "voice_ratio": round(voice_dur / duration, 4) if duration > 0 else 0.0,
        "total_voice_segments": len(voice_segs),
        "total_silence_segments": len([s for s in segs if s["type"] == "silence"]),
    }

    result = {
        "file": os.path.abspath(audio_path),
        "filename": os.path.basename(audio_path),
        "format": {
            "duration_s": round(duration, 3),
            "sample_rate": SAMPLE_RATE,
            "samples": len(audio),
            "channels": 1,
            "bits_per_sample": 16,
        },
        "silence_segments": segs,
        "voice_segments": voice_segs,
        "primitives": primitives,
        "global_stats": global_stats,
        "extract_time_s": round(time.time() - t0, 3),
    }

    return result


def extract_single(audio_path: str, output_dir: str = None) -> dict:
    """单文件提取，可选写文件"""
    result = extract_primitives(audio_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(audio_path))[0]
        out = os.path.join(output_dir, f"{base}_声音基元.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"[已保存] {out}")
    return result


def extract_batch(file_paths: list, output_dir: str = None) -> list:
    """批量文件提取，返回结果列表"""
    all_results = []
    for fp in file_paths:
        try:
            print(f"[声音基元] {os.path.basename(fp)} ...", end=" ")
            r = extract_primitives(fp)
            all_results.append(r)
            print(f"OK {r['global_stats']['total_voice_segments']} 段有声")
        except Exception as e:
            print(f"FAIL 失败: {e}")
            all_results.append({"file": fp, "error": str(e)})

    # 批量汇总写文件
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        meta = {
            "batch_size": len(file_paths),
            "success": sum(1 for r in all_results if "error" not in r),
            "failed": sum(1 for r in all_results if "error" in r),
            "results": all_results,
        }
        ts = time.strftime("%Y%m%d_%H%M%S")
        out = os.path.join(output_dir, f"batch_声音基元_{ts}.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        print(f"[批量汇总已保存] {out}")

    return all_results


# ========== CLI ==========

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("用法:")
        print("  单文件: python -m core.sound_primitives <音频文件>")
        print("  批量:   python -m core.sound_primitives --batch <文件1> <文件2> ...")
        sys.exit(1)

    output_dir = os.path.join(os.path.dirname(__file__), "..", "output")

    if sys.argv[1] == "--batch":
        files = sys.argv[2:]
        if not files:
            print("请指定音频文件列表")
            sys.exit(1)
        results = extract_batch(files, output_dir=output_dir)
        print(f"\n批量完成: {len(results)} 个文件")
    else:
        audio = sys.argv[1]
        result = extract_single(audio, output_dir=output_dir)
        print(json.dumps(result, ensure_ascii=False, indent=2)[:2000])
