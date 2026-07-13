# -*- coding: utf-8 -*-
"""完整管线：转写 + 声学分析 + 报告生成"""
import sys, os, json, time
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.stream_pipeline import StreamPipeline

MP3_PATH = r"F:\认知副脑\SVID_20260619_203835_1.wav"
OUTPUT_DIR = r"F:\认知副脑\output"

def callback(r):
    if "progress" in r and not r.get("analyzing"):
        pct = round(r["progress"] * 100)
        chunk = r.get("chunk", "")
        segs = r.get("total_segments", 0)
        print(f"[进度] {pct}% | 已出{segs}句")
    elif r.get("analyzing"):
        ap = round(r.get("analysis_progress", 0) * 100)
        print(f"[分析] {ap}%")
    elif "refined" in r:
        print(f"[精校] {len(r['refined'])}段")

print("=" * 60)
print("听心 启动")
print("=" * 60)
print(f"音频: {MP3_PATH}")
print()

t0 = time.time()

pipe = StreamPipeline(MP3_PATH)
result = pipe.run(language="zh", use_diarization=True, callback=callback, output_dir=OUTPUT_DIR)

elapsed = time.time() - t0
print()
print("=" * 60)
print(f"完成! {elapsed:.0f}s")
print(f"  段数: {len(result['segments'])}")
print(f"  说话人: {result['num_speakers']}")
print(f"  时长: {result['duration']:.0f}s")

# 输出摘要
for seg in result["segments"][:10]:
    ts = f"{seg['start']:7.1f}s - {seg['end']:5.1f}s"
    label = seg.get("analysis", {}).get("label", "?")
    defense = seg.get("analysis", {}).get("defense_mechanism", "?")
    print(f"  {seg['speaker']} [{ts}] [{label}] [{defense}] {seg['text'][:60]}")
if len(result["segments"]) > 10:
    print(f"  ... (共{len(result['segments'])}句)")

# 说话人统计
print()
print("=" * 60)
print("说话人统计")
print("-" * 60)
from collections import Counter
spk_counts = Counter(s["speaker"] for s in result["segments"])
total_dur = result["duration"]
for spk, cnt in spk_counts.most_common():
    spk_segs = [s for s in result["segments"] if s["speaker"] == spk]
    dur = sum(s["end"] - s["start"] for s in spk_segs)
    labels = Counter(s.get("analysis", {}).get("label", "?") for s in spk_segs)
    defenses = Counter(s.get("analysis", {}).get("defense_mechanism", "?") for s in spk_segs)
    print(f"\n{spk}: {cnt}句 ({dur:.0f}s/{total_dur:.0f}s = {dur/total_dur*100:.0f}%)")
    print(f"  情绪分布: {dict(labels.most_common())}")
    print(f"  防御机制: {dict(defenses.most_common())}")
    avg_ar = sum(s.get("analysis",{}).get("arousal",0) for s in spk_segs)/cnt
    avg_va = sum(s.get("analysis",{}).get("valence",0) for s in spk_segs)/cnt
    avg_do = sum(s.get("analysis",{}).get("dominance",0) for s in spk_segs)/cnt
    print(f"  平均: 唤醒{avg_ar:.2f}·效价{avg_va:.2f}·控制{avg_do:.2f}")

print()
print(f"JSON报告已保存至: {OUTPUT_DIR}")

# ── 调用plain_report生成白话总结 ──
json_path = os.path.join(OUTPUT_DIR, os.path.splitext(os.path.basename(MP3_PATH))[0] + ".json")
try:
    from plain_report import generate_plain_report
    plain = generate_plain_report(json_path)
    print(plain)
except Exception as e:
    print(f"[提示] 白话总结生成跳过（{e}）")
