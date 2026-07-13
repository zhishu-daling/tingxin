"""
ASR 模块 · Whisper GPU (持久子进程)

启动一个 venv Python 进程，加载一次模型
通过 stdin/stdout 管道逐个处理音频块
"""
import os, json, subprocess, tempfile, time, threading
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENV_PY = os.path.join(BASE, "venv_py311", "Scripts", "python.exe")

# 持久子进程和锁
_proc = None
_proc_lock = threading.Lock()


def _start_proc():
    """启动持久 GPU whisper 子进程"""
    global _proc
    script = r'''# -*- coding: utf-8 -*-
import sys, json, os
sys.stdout.reconfigure(encoding="utf-8")
os.environ["WHISPER_CACHE_DIR"] = r"F:\认知副脑\cache\whisper"
os.environ["XDG_CACHE_HOME"] = r"F:\认知副脑\cache"

import whisper, soundfile as sf, warnings
warnings.filterwarnings("ignore")

model = whisper.load_model("small")

for line in sys.stdin:
    line = line.strip()
    if not line: continue
    if line == "__exit__":
        break
    try:
        audio, sr = sf.read(line, dtype="float32")
        result = model.transcribe(audio, language="zh", verbose=False)
        segs = []
        for s in result.get("segments", []):
            segs.append({"start": round(float(s["start"]),2), "end": round(float(s["end"]),2),
                         "text": s["text"].strip(), "probability": round(float(s.get("avg_logprob",0)),3)})
        print(json.dumps(segs, ensure_ascii=False), flush=True)
    except Exception as e:
        print(json.dumps({"error": str(e)[:200]}), flush=True)
'''
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    _proc = subprocess.Popen(
        [VENV_PY, "-c", script],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        encoding="utf-8", bufsize=1, env=env,
    )
    return _proc


def transcribe(audio_chunk, sample_rate=16000):
    """
    用 GPU whisper 转写一段音频（numpy数组）
    持久子进程，模型只加载一次
    """
    global _proc

    # 确保进程在运行
    with _proc_lock:
        if _proc is None or _proc.poll() is not None:
            _start_proc()

    # 写临时 WAV
    tmp = os.path.join(tempfile.gettempdir(), "tingxin_chunk.wav")
    import soundfile as sf
    sf.write(tmp, audio_chunk, sample_rate)

    # 发路径到子进程 stdin
    with _proc_lock:
        _proc.stdin.write(tmp + "\n")
        _proc.stdin.flush()
        line = _proc.stdout.readline()

    # 清理临时文件
    try: os.remove(tmp)
    except: pass

    if not line:
        return []

    try:
        data = json.loads(line.strip())
        if "error" in data:
            print(f"[ASR] GPU 错误: {data['error']}")
            return []
        return data
    except json.JSONDecodeError:
        return []


def stop():
    """关闭持久进程"""
    global _proc
    with _proc_lock:
        if _proc and _proc.poll() is None:
            _proc.stdin.write("__exit__\n")
            _proc.stdin.flush()
            _proc.wait(timeout=5)
        _proc = None
