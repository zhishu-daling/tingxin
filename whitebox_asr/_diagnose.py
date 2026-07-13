# -*- coding: utf-8 -*-
"""诊断：为什么全是e"""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from audio_loader import segment_audio
from primitive_extractor import extract_frame_features, extract_segment_features
from initial_classifier import classify_initial
from final_classifier import classify_final
from tone_classifier import classify_tone
from syllable_fuser import fuse_syllable

r = segment_audio("test_3000chars_16k.wav")
syllables = r["syllables"]

# 看前20个音节的分分类结果
print("前20个音节分类诊断:")
print(f"{'idx':>3} {'dur':>6} {'energy':>8} {'pitch':>7} {'centroid':>9} {'zcr':>6} → {'initial':>8} {'final':>6} {'tone':>4} {'pinyin':>10}")
print("-" * 75)

for i in range(min(20, len(syllables))):
    s = syllables[i]
    frames = extract_frame_features(s["audio"], r["sr"])
    seg_feat = extract_segment_features(frames)
    
    init_r = classify_initial(seg_feat, frames)
    final_r = classify_final(seg_feat, frames)
    
    f0_contour = seg_feat.get("pitch_contour", np.array([]))
    tone_r = classify_tone(f0_contour, segment_features=seg_feat)
    
    syllable = fuse_syllable(init_r, final_r, tone_r)
    
    energy = seg_feat.get("energy_mean", 0)
    pitch = seg_feat.get("pitch_mean", 0)
    centroid = seg_feat.get("spectral_centroid_mean", 0)
    zcr = seg_feat.get("zcr_mean", 0)
    
    print(f"{i:3d} {s['duration_s']:6.3f} {energy:8.5f} {pitch:7.1f} {centroid:9.1f} {zcr:6.4f} → {init_r.get('initial','?'):>8} {final_r.get('final','?'):>6} {tone_r.get('tone',0):>4} {syllable.get('pinyin_numbered','?'):>10}")

# 看第1段的原始特征详情
print("\n\n第1段原始特征:")
s0 = syllables[0]
frames0 = extract_frame_features(s0["audio"], r["sr"])
print(f"  帧数: {frames0['n_frames']}")
print(f"  前5帧能量: {frames0['rms_energy'][:5]}")
print(f"  前5帧F0: {frames0['f0'][:5]}")
print(f"  前5帧质心: {frames0['spectral_centroid'][:5]}")
feat0 = extract_segment_features(frames0)
for k, v in sorted(feat0.items()):
    if isinstance(v, (int, float, np.floating)):
        print(f"  {k}: {v:.4f}")
