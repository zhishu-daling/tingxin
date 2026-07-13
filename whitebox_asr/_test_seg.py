# -*- coding: utf-8 -*-
"""快速测试音节切分"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audio_loader import segment_audio

r = segment_audio("test_3000chars_16k.wav")
print("有声段:", len(r["voice_segments"]), "音节级:", len(r["syllables"]))

if r["syllables"]:
    s = r["syllables"][0]
    print("首音节:", {k: v for k, v in s.items() if k != "audio"})
    durs = [s["duration_s"] for s in r["syllables"]]
    print(f"时长: min={min(durs):.3f}s max={max(durs):.3f}s mean={sum(durs)/len(durs):.3f}s")
    print(f"预期音节数: ~744字, 实际切出: {len(r['syllables'])}段")
