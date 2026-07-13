# -*- coding: utf-8 -*-
"""就用听心声音基元 -> 给拼音"""
import sys, json, numpy as np
sys.path.insert(0, r'F:\认知副脑\听心')
from core.sound_primitives import extract_primitives

src = r'C:\Users\Administrator\.openclaw\media\inbound\20260528_201826---61572750-533a-40a1-9c35-6b84164be3a9.mp4'

result = extract_primitives(src)
print(f"文件: {result['filename']}")
print(f"时长: {result['format']['duration_s']:.1f}s")
print(f"有声段: {len(result['voice_segments'])}段")
print()

for i, p in enumerate(result['primitives']):
    amp = p.get('amplitude', {})
    centroid = p.get('spectral_centroid_hz', 0)
    zcr = p.get('zcr', 0)
    fft_top5 = p.get('fft_bins_128', [])[:5]
    energy = p.get('energy_rms_envelope', [])
    energy_avg = sum(energy)/len(energy) if energy else 0
    
    print(f"[{i}] {p['start_s']:.1f}s-{p['end_s']:.1f}s ({p['duration_s']:.2f}s)")
    print(f"    能量={energy_avg:.5f} 质心={centroid:.0f}Hz ZCR={zcr:.4f}")
    print(f"    FFT前5: {[round(x,1) for x in fft_top5]}")
    print(f"    幅度: max={amp.get('max',0):.3f} rms={amp.get('rms',0):.4f}")
    print()
