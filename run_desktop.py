# -*- coding: utf-8 -*-
"""流式管线跑桌面录音"""
import sys, os, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from core.stream_pipeline import StreamPipeline

audio = os.path.expanduser("~/Desktop/录音.m4a")
print("=== 流式管线: 录音.m4a ===")

pipe = StreamPipeline(audio)
pipe._chunk_ms = 15000

def callback(r):
    pct = round(r["progress"] * 100)
    print(f"[CB] {pct}% | 块{r['chunk']}/{r['total_chunks']} | 已出{r['total_segments']}句")

result = pipe.run(callback=callback)

print()
print(f"完成! {len(result['segments'])}句, {result['num_speakers']}个说话人, 总耗时{result['total_time']:.0f}s")
for seg in result["segments"][:15]:
    ts = f"{seg['start']:7.1f}s - {seg['end']:5.1f}s"
    print(f"  {seg['speaker']} [{ts}]  {seg['text'][:80]}")
if len(result["segments"]) > 15:
    print(f"  ... ({len(result['segments'])}句总)")
