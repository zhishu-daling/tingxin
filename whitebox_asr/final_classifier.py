"""
L3.2 韵母分类公式（白盒）

基于共振峰(F1, F2) + F0的韵母分类器。

核心公式：
- 单元音: (F1, F2) → 元音三角形最近邻匹配
- 复韵母: F1/F2时变轨迹 → 滑动模板匹配
- 鼻韵母: 韵尾鼻音检测
- 卷舌韵母: F3下降检测

白盒式：每个韵母结果可回溯到共振峰坐标与模板的距离。
"""

import numpy as np
from typing import Dict, List, Optional, Tuple

# ========== 单元音模板 (F1, F2) ==========
# 基于标准汉语元音三角形的平均共振峰频率 (Hz)
# 来源：标准普通话发音，成年男性参考值
# 女性偏高约15-20%，但相对位置一致

VOWEL_TEMPLATES = {
    "a":  {"f1_mean": 800,  "f2_mean": 1300, "f1_range": (650, 950),  "f2_range": (1100, 1500)},
    "o":  {"f1_mean": 500,  "f2_mean": 900,  "f1_range": (400, 600),  "f2_range": (700, 1100)},
    "e":  {"f1_mean": 550,  "f2_mean": 1600, "f1_range": (400, 700),  "f2_range": (1300, 1900)},
    "i":  {"f1_mean": 300,  "f2_mean": 2300, "f1_range": (200, 400),  "f2_range": (2000, 2600)},
    "u":  {"f1_mean": 350,  "f2_mean": 700,  "f1_range": (250, 450),  "f2_range": (500, 900)},
    "ü":  {"f1_mean": 300,  "f2_mean": 1800, "f1_range": (200, 400),  "f2_range": (1600, 2000)},
    "er": {"f1_mean": 550,  "f2_mean": 1400, "f1_range": (400, 700),  "f2_range": (1100, 1700),
           "f3_drop": 300},  # er 的 F3 比正常低 >300Hz
}

# ========== 复韵母轨迹模板 ==========
# 每个复韵母用 F1/F2 起止坐标描述
# (f1_start, f2_start) → (f1_end, f2_end)

DIPHTHONG_TEMPLATES = {
    "ai": {"start": (800, 1300), "end": (350, 2000)},      # a→i
    "ei": {"start": (550, 1500), "end": (350, 2100)},      # e→i
    "ao": {"start": (800, 1300), "end": (500, 900)},       # a→o/u
    "ou": {"start": (500, 1000), "end": (380, 750)},       # o→u
    "ia": {"start": (350, 2200), "end": (750, 1400)},      # i→a
    "ie": {"start": (350, 2200), "end": (500, 1700)},      # i→e
    "iu": {"start": (350, 2200), "end": (400, 750)},       # i→o/u (iou)
    "ua": {"start": (380, 700),  "end": (750, 1400)},      # u→a
    "uo": {"start": (380, 700),  "end": (500, 950)},       # u→o
    "ui": {"start": (380, 700),  "end": (350, 2100)},      # u→e→i (uei)
    "üe": {"start": (320, 1850), "end": (500, 1700)},      # ü→e
}

# ========== 鼻韵母 ==========
# 鼻韵母 = 元音部分 + 鼻音韵尾 (n/ng)
# 检测韵尾的鼻音特征

NASAL_FINALS = [
    "an", "en", "in", "un", "ün",
    "ang", "eng", "ing", "ong",
    "ian", "uan", "üan",
    "iang", "uang", "iong",
    "ueng",
]

# ========== 阈值 ==========

THRESHOLDS = {
    "vowel_max_distance": 300,        # (F1,F2) 欧氏距离最大匹配阈值(Hz)
    "vowel_candidate_count": 3,       # 保留候选项数
    
    "diphthong_min_duration_ms": 80,  # 复韵母最小时长
    "diphthong_trajectory_length": 0.3,  # 轨迹变化最小距离
    
    "nasal_coda_min_ratio": 0.15,     # 韵尾鼻音段占比
    "nasal_f1_threshold": 400,        # 鼻音韵尾F1上限
    "nasal_f2_threshold": 1600,       # 鼻音韵尾F2上限（区分n/ng）
    
    "er_f3_drop_threshold": 250,      # er的F3下降阈值(Hz)
}


def _compute_formant_tracks(frame_features: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    从帧特征中提取F1、F2随时间变化的轨迹
    
    返回 (f1_track, f2_track, f3_track) 每帧的共振峰估计
    """
    # 从MFCC + 频谱信息估算共振峰轨迹
    # 简化方法：用spectral centroid + MFCC近似
    mfcc = frame_features.get("mfcc", np.zeros((13, 0)))
    sc = frame_features.get("spectral_centroid", np.array([]))
    n_frames = mfcc.shape[1]
    
    if n_frames == 0:
        return np.array([]), np.array([]), np.array([])
    
    # 从MFCC到(F1,F2)的映射：使用经验公式
    # MFCC0 ~ 总能量
    # MFCC1 ~ 频谱倾斜 → 间接反映F1
    # MFCC2 ~ 第一个共振峰位置
    # MFCC3 ~ 第二个共振峰位置
    
    f1_track = np.zeros(n_frames)
    f2_track = np.zeros(n_frames)
    f3_track = np.zeros(n_frames)
    
    for i in range(n_frames):
        # 简化共振峰跟踪：基于MFCC的启发式公式
        c0 = mfcc[0, i] if mfcc.shape[0] > 0 else 0
        c1 = mfcc[1, i] if mfcc.shape[0] > 1 else 0
        c2 = mfcc[2, i] if mfcc.shape[0] > 2 else 0
        c3 = mfcc[3, i] if mfcc.shape[0] > 3 else 0
        
        # 经验映射（基于统计，后续可优化）
        f1_track[i] = max(200, 500 - c1 * 30 + c2 * 20)
        f2_track[i] = max(500, 1200 + c1 * 20 - c2 * 40 + c3 * 15)
        f3_track[i] = max(1800, 2500 - c2 * 15 + c3 * 10)
    
    # 用实际共振峰做校准（如果有）
    # 这里是简化版本，更精确的需要帧级LPC
    
    return f1_track, f2_track, f3_track


def _match_monophthong(f1_mean: float, f2_mean: float,
                       f3: float = None) -> Dict:
    """
    单元音匹配：基于(F1,F2)到元音三角形的最近邻
    
    返回候选项列表（按距离排序）
    """
    candidates = []
    
    for vowel, tmpl in VOWEL_TEMPLATES.items():
        if vowel == "er":
            # er 需要额外检测 F3 下降
            if f3 is not None:
                # 假设正常第三共振峰约2500Hz
                f3_drop = 2500 - f3
                if f3_drop < THRESHOLDS["er_f3_drop_threshold"]:
                    continue
            else:
                continue
        
        # 欧氏距离
        dist = np.sqrt((f1_mean - tmpl["f1_mean"])**2 + (f2_mean - tmpl["f2_mean"])**2)
        
        # 检查是否在范围内
        in_range = (tmpl["f1_range"][0] <= f1_mean <= tmpl["f1_range"][1]
                    and tmpl["f2_range"][0] <= f2_mean <= tmpl["f2_range"][1])
        
        candidates.append({
            "vowel": vowel,
            "distance": dist,
            "in_range": in_range,
            "template": tmpl,
        })
    
    # 排序
    candidates.sort(key=lambda c: c["distance"])
    
    return {"candidates": candidates[:THRESHOLDS["vowel_candidate_count"]]}


def _match_diphthong(f1_track: np.ndarray, f2_track: np.ndarray,
                     frame_times: np.ndarray) -> Dict:
    """
    复韵母匹配：检测F1/F2的滑动轨迹
    """
    if len(f1_track) < 3 or len(f2_track) < 3:
        return {"detected": False, "final": None, "score": 0.0, "details": {}}
    
    # 取轨迹的起止和中间点
    n = len(f1_track)
    start_idx = n // 4
    end_idx = 3 * n // 4
    
    f1_start = np.mean(f1_track[:start_idx+1]) if start_idx > 0 else f1_track[0]
    f2_start = np.mean(f2_track[:start_idx+1]) if start_idx > 0 else f2_track[0]
    f1_end = np.mean(f1_track[end_idx:]) if end_idx < n else f1_track[-1]
    f2_end = np.mean(f2_track[end_idx:]) if end_idx < n else f2_track[-1]
    
    # 轨迹总变化量
    trajectory_length = np.sqrt((f1_end - f1_start)**2 + (f2_end - f2_start)**2)
    
    result = {
        "detected": False,
        "final": None,
        "score": 0.0,
        "details": {
            "f1_start": float(f1_start),
            "f2_start": float(f2_start),
            "f1_end": float(f1_end),
            "f2_end": float(f2_end),
            "trajectory_length": float(trajectory_length),
        }
    }
    
    if trajectory_length < THRESHOLDS["diphthong_trajectory_length"] * 200:
        return result  # 轨迹太短 → 单元音
    
    # 匹配滑动模板
    best_match = None
    best_score = 0.0
    
    for diphthong, tmpl in DIPHTHONG_TEMPLATES.items():
        start_dist = np.sqrt((f1_start - tmpl["start"][0])**2
                           + (f2_start - tmpl["start"][1])**2)
        end_dist = np.sqrt((f1_end - tmpl["end"][0])**2
                         + (f2_end - tmpl["end"][1])**2)
        
        # 起止点都要匹配得好
        total_dist = start_dist + end_dist
        
        # 检查方向一致性
        actual_dir = np.arctan2(f2_end - f2_start, f1_end - f1_start)
        expected_dir = np.arctan2(tmpl["end"][1] - tmpl["start"][1],
                                  tmpl["end"][0] - tmpl["start"][0])
        dir_diff = abs(actual_dir - expected_dir)
        
        score = 1.0 / (1.0 + total_dist / 200 + dir_diff)
        
        if score > best_score:
            best_score = score
            best_match = diphthong
    
    if best_match and best_score > 0.3:
        result["detected"] = True
        result["final"] = best_match
        result["score"] = best_score
        result["details"]["best_match"] = best_match
        result["details"]["match_score"] = float(best_score)
    
    return result


def _detect_nasal_coda(f1_track: np.ndarray, f2_track: np.ndarray,
                       f0_contour: np.ndarray, rms_energy: np.ndarray) -> Dict:
    """
    检测韵尾是否有鼻音特征
    """
    result = {
        "has_nasal_coda": False,
        "coda_type": None,  # "n" or "ng"
        "coda_ratio": 0.0,
        "details": {},
    }
    
    if len(f1_track) < 5:
        return result
    
    n = len(f1_track)
    # 检查后1/3段
    tail_start = 2 * n // 3
    tail_f1 = f1_track[tail_start:]
    tail_f2 = f2_track[tail_start:]
    
    # 鼻音特征：F1低(~350Hz), F2低(n ~1600Hz, ng ~1000Hz)
    f1_tail_mean = np.mean(tail_f1)
    f2_tail_mean = np.mean(tail_f2)
    
    result["details"]["f1_tail_mean"] = float(f1_tail_mean)
    result["details"]["f2_tail_mean"] = float(f2_tail_mean)
    
    # 能量衰减特征（鼻音韵尾能量下降）
    if len(rms_energy) > tail_start:
        tail_energy = rms_energy[tail_start:]
        energy_decay = np.mean(tail_energy) / (np.mean(rms_energy[:tail_start]) + 1e-10)
        result["details"]["energy_decay_ratio"] = float(energy_decay)
    else:
        energy_decay = 1.0
    
    # F0存在
    if len(f0_contour) > tail_start:
        tail_voiced = np.mean(f0_contour[tail_start:] > 0) > 0.3
    else:
        tail_voiced = False
    result["details"]["tail_voiced"] = bool(tail_voiced)
    
    is_nasal = (f1_tail_mean < THRESHOLDS["nasal_f1_threshold"]
                and f2_tail_mean < THRESHOLDS["nasal_f2_threshold"]
                and energy_decay < 0.8
                and tail_voiced)
    
    if is_nasal:
        result["has_nasal_coda"] = True
        # 区分 n vs ng
        if f2_tail_mean < 1200:
            result["coda_type"] = "ng"
        else:
            result["coda_type"] = "n"
        result["coda_ratio"] = 1.0 - float(energy_decay)
    
    return result


def classify_final(segment_features: dict, frame_features: dict,
                   templates: dict = None) -> dict:
    """
    韵母分类主入口

    流程:
    1. 提取共振峰轨迹 (F1, F2, F3)
    2. 检测鼻音韵尾
    3. 检测复韵母轨迹
    4. 检测卷舌韵母 (er)
    5. 匹配单元音

    返回:
    {
        "final": str,              # 韵母标签
        "final_type": str,         # 类型
        "confidence": float,
        "evidence": dict,          # 证据链
    }
    """
    formants = segment_features.get("formants", [0, 0, 0, 0])
    f1_mean_global = formants[0] if len(formants) > 0 else 0
    f2_mean_global = formants[1] if len(formants) > 1 else 0
    f3_global = formants[2] if len(formants) > 2 else 0
    
    f0 = segment_features.get("pitch_contour", np.array([]))
    rms = segment_features.get("rms_energy_contour", np.array([]))
    frame_times = frame_features.get("frame_times", np.array([]))
    
    # 1. 计算共振峰轨迹
    f1_track, f2_track, f3_track = _compute_formant_tracks(frame_features)
    
    evidence = {
        "f1_mean": float(f1_mean_global),
        "f2_mean": float(f2_mean_global),
        "f3": float(f3_global) if f3_global > 0 else None,
        "global_formants": formants,
    }
    
    # 2. 检测鼻音韵尾
    nasal = _detect_nasal_coda(f1_track, f2_track, f0, rms) if len(f1_track) > 0 else {"has_nasal_coda": False}
    evidence["nasal_coda"] = nasal
    
    # 3. 检测复韵母
    diphthong = _match_diphthong(f1_track, f2_track, frame_times) if len(f1_track) > 0 else {"detected": False}
    evidence["diphthong"] = diphthong
    
    # 4. 检测卷舌韵母
    is_er = False
    if f3_global > 0:
        f3_drop = 2500 - f3_global
        if f3_drop > THRESHOLDS["er_f3_drop_threshold"]:
            is_er = True
    evidence["is_er"] = is_er
    
    # 5. 匹配单元音
    vowel_match = _match_monophthong(f1_mean_global, f2_mean_global, f3_global)
    evidence["vowel_candidates"] = vowel_match["candidates"]
    
    # ===== 决策 =====
    
    # 优先检测：er > 鼻韵母 > 复韵母 > 单元音
    if is_er:
        final = "er"
        final_type = "retroflex"
        confidence = 0.7
        evidence["decision_path"] = "er_detected"
        
    elif nasal["has_nasal_coda"]:
        # 确定鼻韵母的具体元音部分
        best_vowel = vowel_match["candidates"][0] if vowel_match["candidates"] else None
        if best_vowel:
            vowel_base = best_vowel["vowel"]
            coda = nasal["coda_type"]  # "n" or "ng"
            # 组合
            if vowel_base in ["a", "e", "i", "o", "u", "ü"]:
                final = vowel_base + coda
            elif vowel_base == "ia":
                final = "ian" if coda == "n" else "iang"
            elif vowel_base == "ua":
                final = "uan" if coda == "n" else "uang"
            elif vowel_base == "ü":
                final = "ün" if coda == "n" else "iong"  # 近似
            else:
                final = vowel_base + coda
            final_type = "nasal_final"
            confidence = 0.6
            evidence["decision_path"] = "nasal_final"
            evidence["vowel_base"] = vowel_base
            evidence["coda"] = coda
        else:
            # 无法确定元音部分
            final = nasal["coda_type"]
            final_type = "nasal_final"
            confidence = 0.3
            evidence["decision_path"] = "nasal_coda_only"
    
    elif diphthong["detected"]:
        final = diphthong["final"]
        final_type = "diphthong"
        confidence = diphthong["score"]
        evidence["decision_path"] = "diphthong"
        
    elif vowel_match["candidates"]:
        best = vowel_match["candidates"][0]
        if best["in_range"] or best["distance"] < THRESHOLDS["vowel_max_distance"]:
            final = best["vowel"]
            final_type = "monophthong"
            confidence = max(0.0, 1.0 - best["distance"] / 500)
            evidence["decision_path"] = "monophthong"
            evidence["best_distance"] = best["distance"]
        else:
            # 匹配失败
            final = "e"  # 默认
            final_type = "monophthong"
            confidence = 0.2
            evidence["decision_path"] = "fallback"
    else:
        final = "e"
        final_type = "monophthong"
        confidence = 0.15
        evidence["decision_path"] = "fallback_empty"
    
    return {
        "final": final,
        "final_type": final_type,
        "confidence": round(confidence, 4),
        "evidence": evidence,
    }
