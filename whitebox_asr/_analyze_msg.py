# -*- coding: utf-8 -*-
"""分析用户发来的语音消息"""
import sys, json
sys.path.insert(0, r'F:\认知副脑\听心')

from core.preprocess import load_audio
from core.sound_primitives import _silence_segments, _extract_primitive
from core.acoustic_features import extract_segment_features
from core.analysis import compute_arousal, compute_valence, compute_dominance, emotion_label

src = r'C:\Users\Administrator\.openclaw\media\inbound\20260528_201826---61572750-533a-40a1-9c35-6b84164be3a9.mp4'
audio = load_audio(src)
sr = 16000

segs = _silence_segments(audio, sr)
voice = [s for s in segs if s['type'] == 'voice']
print(f"有声段: {len(voice)}段, 总长{len(audio)/sr:.1f}s")

for i, vs in enumerate(voice):
    ss = int(vs["start_s"] * sr)
    se = int(vs["end_s"] * sr)
    seg_audio = audio[ss:se]
    prim = _extract_primitive(seg_audio, sr)
    
    feat = extract_segment_features(seg_audio, sr)
    a, _ = compute_arousal(feat)
    v, _ = compute_valence(feat)
    d, _ = compute_dominance(feat)
    label, conf = emotion_label(a, v, d)
    
    dur = vs["duration_s"]
    energy = prim.get("amplitude", {}).get("rms", 0)
    centroid = prim.get("spectral_centroid_hz", 0)
    
    print(f"  [{i}] {vs['start_s']:.1f}s-{vs['end_s']:.1f}s ({dur:.1f}s) "
          f"E={energy:.4f} C={centroid:.0f}Hz "
          f"A={a:.2f} V={v:.2f} D={d:.2f} -> {label}")

# 全局统计
print(f"\n总体: 唤醒度均值={sum(compute_arousal(extract_segment_features(audio[int(v['start_s']*sr):int(v['end_s']*sr)]))[0] for v in voice if v['duration_s']>0.3)/max(len([v for v in voice if v['duration_s']>0.3]),1):.2f}")
