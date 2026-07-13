# -*- coding: utf-8 -*-
"""用达令的录音校准白盒ASR"""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from audio_loader import segment_audio
from primitive_extractor import extract_frame_features, extract_segment_features
from initial_classifier import classify_initial
from final_classifier import classify_final
from tone_classifier import classify_tone
from syllable_fuser import fuse_syllable
from engine import WhiteboxASR

src = r'C:\Users\Administrator\.openclaw\media\inbound\20260528_201826---61572750-533a-40a1-9c35-6b84164be3a9.mp4'

# 预期文本
expected = "动态成长系统静态封闭系统体验探索系统关系依存系统绩效最优系统"
expected_pinyin = "dong4 tai4 cheng2 zhang3 xi4 tong3 jing4 tai4 feng1 bi4 xi4 tong3 ti3 yan4 tan4 suo3 xi4 tong3 guan1 xi4 yi1 cun2 xi4 tong3 ji4 xiao4 zui4 you1 xi4 tong3"

print("=" * 60)
print("达令校准语音 — 白盒ASR测试")
print("=" * 60)

# 1. 音节切分
print("\n[1] 音节切分...")
seg = segment_audio(src)
print(f"  VAD有声段: {len(seg['voice_segments'])}")
print(f"  音节级段: {len(seg['syllables'])}")

# 2. 逐音节诊断
print("\n[2] 逐音节分类诊断:")
print(f"{'#':>3} {'dur':>6} {'energy':>8} {'pitch':>7} {'init':>8} {'final':>6} {'tone':>4} {'pinyin':>10}")
print("-" * 60)

for i, s in enumerate(seg["syllables"]):
    frames = extract_frame_features(s["audio"], seg["sr"])
    feat = extract_segment_features(frames)
    
    init_r = classify_initial(feat, frames)
    final_r = classify_final(feat, frames)
    f0 = feat.get("pitch_contour", np.array([]))
    tone_r = classify_tone(f0, segment_features=feat)
    syl = fuse_syllable(init_r, final_r, tone_r)
    
    energy = feat.get("energy_mean", 0)
    pitch = feat.get("pitch_mean", 0)
    
    print(f"{i:3d} {s['duration_s']:6.3f} {energy:8.4f} {pitch:7.1f} "
          f"{init_r.get('initial','?'):>8} {final_r.get('final','?'):>6} "
          f"{tone_r.get('tone',0):>4} {syl.get('pinyin_numbered','?'):>10}")

# 3. 整段跑引擎
print("\n[3] 整段转写:")
asr = WhiteboxASR()
result = asr.transcribe(src)
print(f"  识别文本: {result['text']}")
print(f"  拼音: {' '.join(result['pinyin_numbered'])}")
print(f"  耗时: {result['timing']['total']:.3f}s")

# 4. 对比
print(f"\n[4] 对比:")
print(f"  预期: {expected}")
print(f"  识别: {result['text']}")

# 5. 输出校准数据
print("\n[5] 真相输出（逐音节特征，供校准用）:")
calibration_data = []
for i, s in enumerate(seg["syllables"]):
    frames = extract_frame_features(s["audio"], seg["sr"])
    feat = extract_segment_features(frames)
    init_r = classify_initial(feat, frames)
    final_r = classify_final(feat, frames)
    f0 = feat.get("pitch_contour", np.array([]))
    tone_r = classify_tone(f0, segment_features=feat)
    syl = fuse_syllable(init_r, final_r, tone_r)
    
    cal = {
        "index": i,
        "duration": s["duration_s"],
        "energy_mean": feat.get("energy_mean", 0),
        "pitch_mean": feat.get("pitch_mean", 0),
        "centroid_mean": feat.get("spectral_centroid_mean", 0),
        "zcr_mean": feat.get("zcr_mean", 0),
        "high_freq_ratio": feat.get("high_freq_ratio", 0),
        "voiced_ratio": feat.get("voiced_ratio", 0),
        "energy_max_jump": feat.get("energy_max_jump", 0),
        "initial_detected": init_r.get("initial", ""),
        "final_detected": final_r.get("final", ""),
        "tone_detected": tone_r.get("tone", 0),
        "pinyin_detected": syl.get("pinyin_numbered", ""),
    }
    calibration_data.append(cal)
    
    # 打印特征丰富的行
    print(f"  [{i:2d}] dur={s['duration_s']:.3f} E={feat.get('energy_mean',0):.4f} "
          f"P={feat.get('pitch_mean',0):.0f} C={feat.get('spectral_centroid_mean',0):.0f} "
          f"Z={feat.get('zcr_mean',0):.4f} H={feat.get('high_freq_ratio',0):.2f} "
          f"J={feat.get('energy_max_jump',0):.4f} "
          f"→ init={init_r.get('initial','?'):>4} fin={final_r.get('final','?'):>5} "
          f"t={tone_r.get('tone',0)} py={syl.get('pinyin_numbered','?'):>6}")

# 保存校准数据
cal_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration_data.json")
with open(cal_path, "w", encoding="utf-8") as f:
    json.dump({"expected": expected, "syllables": calibration_data}, f, ensure_ascii=False, indent=2)
print(f"\n校准数据已保存: {cal_path}")
