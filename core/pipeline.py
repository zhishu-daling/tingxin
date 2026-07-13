"""
听心 · 核心管线

编排 预处理 → ASR → 说话人分离 → 输出

用法:
    python -m core.pipeline 音频.mp3
"""
import json
import os
import sys
import time
from pathlib import Path

from .asr import transcribe
from .diarize import diarize
from .preprocess import load_audio, get_duration


def run(
    audio_path: str,
    output_dir: str = None,
    language: str = "zh",
    use_diarization: bool = True,
) -> dict:
    """
    执行核心管线

    参数:
        audio_path: 音频文件路径
        output_dir: 输出目录（None = 不写文件）
        language: 语言
        use_diarization: 是否做说话人分离

    返回:
        pipeline 输出 dict
    """
    start_time = time.time()
    audio_path = str(audio_path)

    # 验证文件存在
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"音频文件不存在: {audio_path}")

    duration = get_duration(audio_path)
    print(f"📂 音频时长: {duration:.1f}s")

    # === 1. ASR ===
    print(f"🎤 ASR 识别中...")
    t0 = time.time()
    segments = transcribe(
        audio_path,
        language=language,
        beam_size=5,
        word_timestamps=False,
    )
    t1 = time.time()
    print(f"  耗时: {t1-t0:.1f}s, 识别出 {len(segments)} 句")

    if not segments:
        print("⚠️ 未识别出任何内容")
        return {"file": audio_path, "duration": duration, "segments": []}

    # === 2. 说话人分离 ===
    if use_diarization and len(segments) > 1:
        print(f"🔊 说话人分离中...")
        segments = diarize(segments, audio_path)
        print(f"  完成: 共 {len(set(s['speaker'] for s in segments))} 人")
    else:
        for seg in segments:
            seg["speaker"] = "SPK1"

    t2 = time.time()
    print(f"⏱ 总耗时: {t2-start_time:.1f}s")

    # === 3. 组装结果 ===
    result = {
        "file": audio_path,
        "filename": os.path.basename(audio_path),
        "duration": round(duration, 2),
        "total_time": round(t2 - start_time, 2),
        "num_speakers": len(set(s["speaker"] for s in segments)),
        "segments": segments,
    }

    # === 4. 写文件 ===
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        basename = os.path.splitext(os.path.basename(audio_path))[0]
        out_path = os.path.join(output_dir, f"{basename}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"💾 已保存: {out_path}")

    return result


def print_result(result: dict):
    """控制台打印带说话人的逐句结果"""
    print("\n" + "=" * 60)
    print(f"文件: {result['filename']}")
    print(f"时长: {result['duration']:.1f}s | 说话人: {result['num_speakers']}")
    print("=" * 60)

    for seg in result["segments"]:
        speaker = seg["speaker"]
        ts = f"{seg['start']:7.1f}s - {seg['end']:5.1f}s"
        text = seg["text"]
        print(f"  {speaker}  [{ts}]  {text}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python -m core.pipeline <音频文件> [--no-diarize]")
        sys.exit(1)

    audio = sys.argv[1]
    use_diar = "--no-diarize" not in sys.argv

    output_dir = os.path.join(os.path.dirname(__file__), "..", "output")

    result = run(audio, output_dir=output_dir, use_diarization=use_diar)
    print_result(result)
