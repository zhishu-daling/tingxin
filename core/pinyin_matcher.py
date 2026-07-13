# -*- coding: utf-8 -*-
"""
拼音匹配引擎 (PinyinMatcher) - 修复版
使用和DB生成时一致的提取参数
"""
import os
import sqlite3
import struct
import numpy as np

# 自适应基元提取（与DB生成时一致）
def extract_primitives_adaptive(audio_path, silence_thresh_db=-50):
    """从文件直接提取基元，使用与DB生成时一致的低阈值"""
    import sys
    if __name__ == '__main__' or not __package__:
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        from core.sound_primitives import _silence_segments, _extract_primitive
        from core.preprocess import load_audio, SAMPLE_RATE
    else:
        from .sound_primitives import _silence_segments, _extract_primitive
        from .preprocess import load_audio, SAMPLE_RATE
    
    audio = load_audio(audio_path, SAMPLE_RATE)
    segs = _silence_segments(audio, SAMPLE_RATE, silence_thresh_db=silence_thresh_db)
    voice_segs = [s for s in segs if s['type'] == 'voice']
    
    primitives = []
    for i, vs in enumerate(voice_segs):
        start = int(vs['start_s'] * SAMPLE_RATE)
        end = int(min(vs['end_s'] * SAMPLE_RATE, len(audio)))
        seg_audio = audio[start:end]
        prim = _extract_primitive(seg_audio, SAMPLE_RATE)
        prim['segment_index'] = i
        prim['start_s'] = vs['start_s']
        prim['end_s'] = vs['end_s']
        primitives.append(prim)
    
    return primitives


class PinyinMatcher:
    """拼音匹配引擎"""
    
    def __init__(self, db_path=None):
        if db_path is None:
            db_path = os.path.join(os.path.dirname(__file__), 'pinyin.db')
        self.db_path = db_path
        self._load_reference()
    
    def _load_reference(self):
        """从数据库加载所有参考指纹"""
        self.conn = sqlite3.connect(self.db_path)
        cursor = self.conn.cursor()
        
        cursor.execute('''
            SELECT id, pinyin, shengmu, yunmu, shengdiao,
                   fft_profile, spectral_centroid, amplitude_rms,
                   duration_ms
            FROM pinyin_combinations 
            WHERE fft_profile IS NOT NULL
            ORDER BY id
        ''')
        
        self.refs = []
        for row in cursor.fetchall():
            ref = {
                'id': row[0],
                'pinyin': row[1],
                'shengmu': row[2],
                'yunmu': row[3],
                'shengdiao': row[4],
                'fft': self._unpack_fft(row[5]),
                'centroid': row[6],
                'rms': row[7],
                'duration_ms': row[8],
            }
            self.refs.append(ref)
        
        print(f"[PinyinMatcher] 已加载 {len(self.refs)} 个拼音参考指纹")
    
    def _unpack_fft(self, blob):
        if not blob:
            return None
        return np.array(struct.unpack('128f', blob), dtype=np.float32)
    
    def match_from_audio(self, audio_path, top_k=5):
        """从音频文件直接提取基元并匹配（推荐使用）"""
        primitives = extract_primitives_adaptive(audio_path, silence_thresh_db=-50)
        
        if not primitives:
            return []
        
        prim = primitives[0]
        query_fft = np.array(prim.get('fft_bins_128', []), dtype=np.float32)
        query_centroid = prim.get('spectral_centroid_hz', 0.0)
        query_rms = prim.get('amplitude', {}).get('rms', 0.0)
        
        return self._match(query_fft, query_centroid, query_rms, top_k=top_k)
    
    def match_from_primitives(self, primitives_dict, top_k=5):
        """从听心基元字典匹配拼音"""
        if not primitives_dict.get('primitives'):
            return []
        
        prim = primitives_dict['primitives'][0]
        query_fft = np.array(prim.get('fft_bins_128', []), dtype=np.float32)
        query_centroid = prim.get('spectral_centroid_hz', 0.0)
        query_rms = prim.get('amplitude', {}).get('rms', 0.0)
        
        return self._match(query_fft, query_centroid, query_rms, top_k=top_k)
    
    def _match(self, query_fft, query_centroid, query_rms, top_k=5):
        """核心匹配：基于FFT欧氏距离 + FFT余弦 + 质心 + RMS"""
        if len(query_fft) < 128:
            return []
        
        q_fft = np.array(query_fft, dtype=np.float32)
        q_norm = np.linalg.norm(q_fft)
        q_fft_n = q_fft / q_norm if q_norm > 0 else q_fft
        
        scores = []
        for ref in self.refs:
            r_fft = ref['fft']
            r_norm = np.linalg.norm(r_fft)
            r_fft_n = r_fft / r_norm if r_norm > 0 else r_fft
            
            # 1) FFT余弦相似度（归一化后）
            fft_cos = float(np.dot(q_fft_n, r_fft_n))
            fft_cos = max(-1.0, min(1.0, fft_cos))
            fft_dist = 1.0 - fft_cos
            
            # 2) FFT欧氏距离（不归一化，保留能量信息）
            fft_euclidean = float(np.sqrt(np.mean((q_fft - r_fft) ** 2)))
            fft_euclidean_normed = fft_euclidean / (np.sqrt(np.mean(q_fft ** 2)) + 1e-6)
            
            # 3) 质心差异
            c_diff = abs(query_centroid - ref['centroid']) / max(query_centroid, ref['centroid'], 1.0)
            
            # 4) RMS差异
            r_diff = abs(query_rms - ref['rms']) / max(query_rms, ref['rms'], 1e-6)
            
            # 综合得分：FFT余弦40% + FFT欧氏30% + 质心20% + RMS10%
            score = (0.40 * fft_dist + 
                     0.30 * min(fft_euclidean_normed, 1.0) +
                     0.20 * min(c_diff, 1.0) + 
                     0.10 * min(r_diff, 1.0))
            
            scores.append((ref['pinyin'], score))
        
        scores.sort(key=lambda x: x[1])
        top = scores[:top_k]
        
        results = []
        for p, sc in top:
            confidence = float(np.exp(-sc * 4))
            results.append((p, confidence, sc))
        
        return results
    
    def transcribe(self, audio_path):
        """转录音频文件为拼音列表"""
        primitives = extract_primitives_adaptive(audio_path, silence_thresh_db=-50)
        
        result = []
        for seg in primitives:
            q_fft = np.array(seg.get('fft_bins_128', []), dtype=np.float32)
            q_centroid = seg.get('spectral_centroid_hz', 0.0)
            q_rms = seg.get('amplitude', {}).get('rms', 0.0)
            
            matches = self._match(q_fft, q_centroid, q_rms, top_k=3)
            
            if matches:
                result.append({
                    'pinyin': matches[0][0],
                    'confidence': matches[0][1],
                    'start_s': seg.get('start_s', 0),
                    'end_s': seg.get('end_s', 0),
                    'candidates': [m[0] for m in matches],
                })
        
        return result
    
    def close(self):
        if self.conn:
            self.conn.close()


# ========== 测试 ==========
if __name__ == '__main__':
    import time
    
    db_path = r'F:\认知副脑\听心\core\pinyin.db'
    matcher = PinyinMatcher(db_path)
    
    # 测试：用match_from_audio直接匹配（与DB使用相同提取参数）
    print("\n=== 自匹配测试（自适应提取）===")
    test_pinyins = ['ba1', 'ma1', 'da4', 'yi1', 'wu3', 'shi4', 'ni3', 'ta1',
                    'zhe1', 'zhe3', 'zhe4', 'fa1', 'la4', 'pa1', 'bo1']
    wav_dir = r'F:\认知副脑\听心\temp\pinyin_wavs'
    
    correct = 0
    total = 0
    for p in test_pinyins:
        wav_path = os.path.join(wav_dir, f"{p}.wav")
        if not os.path.exists(wav_path):
            continue
        
        t0 = time.time()
        matches = matcher.match_from_audio(wav_path, top_k=5)
        elapsed = time.time() - t0
        
        total += 1
        if matches:
            is_correct = matches[0][0] == p
            if is_correct:
                correct += 1
            icon = '✅' if is_correct else '❌'
            top_pinyins = [m[0] for m in matches[:3]]
            print(f"  '{p}' -> '{matches[0][0]}' (conf={matches[0][1]:.3f}) {icon} | {top_pinyins} | [{elapsed*1000:.0f}ms]")
    
    print(f"\n  自匹配准确率: {correct}/{total} ({correct/total*100:.0f}%)")
    
    # 音近组
    print("\n=== 音近拼音区分 ===")
    groups = [
        ['ba1', 'pa1', 'ma1', 'fa1'],
        ['da4', 'ta4', 'na4', 'la4'],
        ['zhe4', 'zhe1', 'zhe3'],
        ['bo1', 'po1', 'mo1', 'fo1'],
    ]
    
    for group in groups:
        print(f"\n  组: {group}")
        g_correct = 0
        for p in group:
            wav_path = os.path.join(wav_dir, f"{p}.wav")
            if not os.path.exists(wav_path):
                continue
            matches = matcher.match_from_audio(wav_path, top_k=4)
            top_pinyins = [m[0] for m in matches]
            is_correct = matches and matches[0][0] == p
            if is_correct:
                g_correct += 1
            icon = '✅' if is_correct else '❌'
            print(f"    '{p}' -> {top_pinyins} {icon}")
        print(f"    组准确率: {g_correct}/{len(group)}")
    
    matcher.close()
