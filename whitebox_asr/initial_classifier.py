"""
L3.1 声母分类公式（白盒）

基于声学原理的声母分类器，无需训练数据。
- 爆破音: 能量包络存在"闭合→爆发"阶跃
- 擦音: 高频能量占比高，能量连续
- 塞擦音: 爆破+擦音特征先后出现
- 鼻音: 低频共振峰稳定，F0存在
- 边音/近音: 共振峰过渡特征

白盒式：每个判断可以回溯到哪个特征阈值触发了哪个分支。
"""

import numpy as np
from typing import Dict, List, Optional


# ========== 阈值参数（手工调参）==========

THRESHOLDS = {
    # 爆破音检测
    "plosive_energy_jump_ratio": 3.0,      # 能量阶跃比（爆发点前后能量比）
    "plosive_flatness_threshold": 0.6,      # 爆发点附近频谱平坦度
    "plosive_mfcc_std_threshold": 1.0,      # 爆破音MFCC标准差较低
    
    # 送气检测
    "aspirated_duration_ms": 60,            # 送气段最小长度(ms)
    "aspirated_high_freq_ratio": 0.25,      # 送气段高频能量占比阈值
    
    # 擦音检测
    "fricative_high_energy_ratio": 0.3,     # 高频能量占比 > 0.3
    "fricative_energy_continuity": 0.5,     # 能量连续性（低标准差）
    "fricative_min_duration_ms": 80,        # 最小擦音段长度
    
    # 擦音细分 - MFCC重心位置
    "fricative_mfcc_centroid_zh": 2.0,      # 腭龈擦音MFCC重心 > 此值
    "fricative_mfcc_centroid_z": 1.0,       # 齿龈擦音MFCC重心 > 此值
    
    # 鼻音检测
    "nasal_low_freq_energy_ratio": 0.6,     # 低频能量占比
    "nasal_f0_min": 80,                     # 鼻音必须有F0
    "nasal_energy_flatness": 0.3,           # 能量平坦度
    
    # 边音检测
    "lateral_f1_max": 500,                  # 边音F1上限(Hz)
    "lateral_f2_min": 800,                  # 边音F2下限(Hz)
    "lateral_f2_max": 2000,                 # 边音F2上限(Hz)
    
    # 近音 r
    "retroflex_f3_drop": 200,               # 卷舌F3下降量(Hz)
    
    # 零声母（韵母开头）
    "null_initial_energy_rise_time_ms": 30, # 能量上升时间 < 30ms 视为零声母
}

# ========== 声母列表 ==========

INITIALS_PLOSIVE_UNASPIRATED = ["b", "d", "g"]
INITIALS_PLOSIVE_ASPIRATED = ["p", "t", "k"]
INITIALS_FRICATIVE_DENTAL = ["z", "c", "s"]      # 齿龈擦音 (舌尖前)
INITIALS_FRICATIVE_PALATAL = ["zh", "ch", "sh"]  # 腭龈擦音 (舌尖后)
INITIALS_FRICATIVE_GUTTURAL = ["h"]              # 喉擦音
INITIALS_FRICATIVE_LABIODENTAL = ["f"]           # 唇齿擦音
INITIALS_AFFRICATE = ["j", "q", "x"]             # 腭塞擦音
INITIALS_NASAL = ["m", "n"]                      # 鼻音 (ng只做韵尾)
INITIALS_LIQUID = ["l", "r"]                     # 边音/近音

ALL_INITIALS = [
    "b", "p", "m", "f",
    "d", "t", "n", "l",
    "g", "k", "h",
    "j", "q", "x",
    "zh", "ch", "sh", "r",
    "z", "c", "s",
    "",  # 零声母
]


def _compute_energy_envelope_jump(rms_energy: np.ndarray) -> Dict:
    """
    检测能量包络的阶跃（爆破音特征）
    
    返回:
    {
        "has_jump": bool,
        "jump_position": int,       # 阶跃位置(帧索引)
        "jump_ratio": float,        # 阶跃比（爆发后/爆发前）
        "pre_jump_energy": float,
        "post_jump_energy": float,
    }
    """
    if len(rms_energy) < 3:
        return {"has_jump": False, "jump_position": -1, "jump_ratio": 0.0,
                "pre_jump_energy": 0.0, "post_jump_energy": 0.0}

    # 滑动窗口寻找最大能量变化率
    max_ratio = 0.0
    jump_pos = -1
    
    for i in range(1, len(rms_energy) - 1):
        pre = np.mean(rms_energy[max(0, i-2):i]) + 1e-10
        post = np.mean(rms_energy[i:min(len(rms_energy), i+3)]) + 1e-10
        ratio = post / pre
        if ratio > max_ratio:
            max_ratio = ratio
            jump_pos = i

    return {
        "has_jump": max_ratio >= THRESHOLDS["plosive_energy_jump_ratio"],
        "jump_position": jump_pos,
        "jump_ratio": max_ratio,
        "pre_jump_energy": float(np.mean(rms_energy[max(0, jump_pos-2):jump_pos])) if jump_pos > 0 else 0.0,
        "post_jump_energy": float(np.mean(rms_energy[jump_pos:min(len(rms_energy), jump_pos+3)])) if jump_pos >= 0 else 0.0,
    }


def _compute_spectral_flatness(mfcc: np.ndarray, frame_idx: int, window: int = 3) -> float:
    """
    计算频谱平坦度（用于检测爆破音的"噪声爆发"）
    平坦度高 ≈ 类似白噪声（爆破音特征）
    """
    if mfcc.shape[1] == 0:
        return 0.0
    start = max(0, frame_idx - window)
    end = min(mfcc.shape[1], frame_idx + window + 1)
    # MFCC方差小 ≈ 频谱平坦
    frame_mfcc = mfcc[:, start:end]
    if frame_mfcc.shape[1] < 2:
        return 0.0
    # 各MFCC系数的帧间标准差均值
    stds = np.std(frame_mfcc, axis=1)
    return 1.0 - np.mean(stds) / (np.mean(np.abs(np.mean(frame_mfcc, axis=1))) + 1e-10)


def _detect_plosive_type(segment_features: dict, frame_features: dict) -> Dict:
    """
    爆破音检测与细分
    返回 evidence dict
    """
    evidence = {"detected": False, "type": None, "score": 0.0, "details": {}}
    
    rms = frame_features.get("rms_energy", np.array([]))
    if len(rms) < 3:
        return evidence

    # 1. 检测能量阶跃
    jump = _compute_energy_envelope_jump(rms)
    evidence["details"]["energy_jump"] = jump

    if not jump["has_jump"]:
        return evidence

    # 2. 检测频谱平坦度
    flatness = _compute_spectral_flatness(
        frame_features["mfcc"], jump["jump_position"]
    )
    evidence["details"]["spectral_flatness"] = flatness

    if flatness < THRESHOLDS["plosive_flatness_threshold"]:
        # 不够平坦 → 可能是其他音
        evidence["details"]["flatness_rejected"] = True
        return evidence

    # 3. 判断送气与否
    # 送气爆破音(p/t/k)在爆发后有较长的高频噪声段
    # 不送气爆破音(b/d/g)爆发后快速过渡到韵母
    jump_pos = jump["jump_position"]
    if jump_pos < len(rms) - 2:
        # 爆发后段
        post_segment = rms[jump_pos + 1:]
        # 能量持续长度
        high_energy_frames = np.sum(post_segment > np.mean(rms) * 0.5)
        post_duration_ms = high_energy_frames * 10  # 10ms/帧
        
        # 高频能量（用spectral_centroid近似）
        sc = frame_features.get("spectral_centroid", np.array([]))
        post_sc = sc[jump_pos + 1:] if len(sc) > jump_pos + 1 else np.array([])
        high_freq_ratio = np.mean(post_sc > 3000) if len(post_sc) > 0 else 0.0
        
        evidence["details"]["post_duration_ms"] = post_duration_ms
        evidence["details"]["post_high_freq_ratio"] = high_freq_ratio
        
        is_aspirated = (post_duration_ms > THRESHOLDS["aspirated_duration_ms"]
                        and high_freq_ratio > THRESHOLDS["aspirated_high_freq_ratio"])
        
        # 4. 按发音部位细分
        # 用MFCC区分 b/p(唇), d/t(齿), g/k(腭)
        mfcc = frame_features["mfcc"]
        mfcc_mean = np.mean(mfcc, axis=1)
        
        # 基于 MFCC 的粗略发音部位估计
        # b/p 的 MFCC 前几维有特定模式
        mfcc_1_2_ratio = abs(mfcc_mean[1] / (mfcc_mean[2] + 1e-10))
        mfcc_0 = mfcc_mean[0]
        
        if mfcc_0 > 0:
            # 高MFFC0 → 唇音(b/p)
            place = "labial"
            initials = INITIALS_PLOSIVE_ASPIRATED if is_aspirated else INITIALS_PLOSIVE_UNASPIRATED
            initial = initials[0]  # b or p
        elif mfcc_1_2_ratio > 1.5:
            place = "alveolar"
            initials = ["t"] if is_aspirated else ["d"]
            initial = initials[0]
        else:
            place = "velar"
            initials = ["k"] if is_aspirated else ["g"]
            initial = initials[0]
        
        evidence["detected"] = True
        evidence["type"] = "plosive"
        evidence["initial"] = initial
        evidence["score"] = min(1.0, jump["jump_ratio"] / 10.0)
        evidence["details"]["place"] = place
        evidence["details"]["aspirated"] = is_aspirated

    return evidence


def _detect_fricative_type(segment_features: dict, frame_features: dict) -> Dict:
    """
    擦音检测与细分
    """
    evidence = {"detected": False, "type": None, "score": 0.0, "details": {}}

    sc = frame_features.get("spectral_centroid", np.array([]))
    rms = frame_features.get("rms_energy", np.array([]))
    mfcc = frame_features.get("mfcc", np.array([]))

    if len(sc) < 2 or len(rms) < 2:
        return evidence

    # 1. 高频能量占比
    high_freq_ratio = np.mean(sc > 4000) if np.max(sc) > 0 else 0.0
    evidence["details"]["high_freq_ratio"] = high_freq_ratio

    if high_freq_ratio < THRESHOLDS["fricative_high_energy_ratio"]:
        return evidence

    # 2. 能量连续性（擦音能量稳定，不像爆破音有阶跃）
    rms_std = np.std(rms) / (np.mean(rms) + 1e-10)
    evidence["details"]["rms_cv"] = rms_std

    # 检查是否有爆破特征（有则可能是塞擦音）
    jump = _compute_energy_envelope_jump(rms)
    has_plosive = jump["has_jump"]

    if has_plosive:
        # 有爆破特征 → 可能是塞擦音，让affricate检测处理
        evidence["details"]["has_plosive_precursor"] = True
        return evidence

    # 3. 细分擦音类型
    # 用MFCC区分齿龈擦音(z/c/s)、腭龈擦音(zh/ch/sh)、喉擦音(h)、唇齿擦音(f)
    mfcc_mean = np.mean(mfcc, axis=1) if mfcc.shape[0] > 0 else np.zeros(13)
    mfcc_centroid = np.mean(mfcc_mean[:5])  # 前5维MFCC均值

    # 粗略识别
    # f: 唇齿——MFCC模式特殊，低频能量较高
    # h: 喉——MFCC第1维较高
    # zh/ch/sh: 腭龈——MFCC重心居中
    # z/c/s: 齿龈——MFCC重心较低
    
    if mfcc_centroid > THRESHOLDS["fricative_mfcc_centroid_zh"]:
        # 高频丰富的擦音 → zh/ch/sh 或 h
        # 检查spectral centroid最高值
        if np.max(sc) > 6000:
            initial = "h"
            place = "guttural"
        else:
            initial = "sh"  # 默认腭龈，后续细化
            place = "palatal"
    elif mfcc_centroid > THRESHOLDS["fricative_mfcc_centroid_z"]:
        # 中频擦音 → z/c/s 或 f
        if mfcc_mean[0] < -200:
            initial = "f"
            place = "labiodental"
        else:
            initial = "s"
            place = "dental"
    else:
        initial = "s"
        place = "dental"

    evidence["detected"] = True
    evidence["type"] = "fricative"
    evidence["initial"] = initial
    evidence["score"] = min(1.0, high_freq_ratio * 2.0)
    evidence["details"]["place"] = place
    evidence["details"]["mfcc_centroid"] = float(mfcc_centroid)

    return evidence


def _detect_nasal_type(segment_features: dict, frame_features: dict) -> Dict:
    """
    鼻音检测 (m/n/ng)
    """
    evidence = {"detected": False, "type": None, "score": 0.0, "details": {}}

    formants = segment_features.get("formants", [0, 0, 0, 0])
    f0 = frame_features.get("f0", np.array([]))
    rms = frame_features.get("rms_energy", np.array([]))

    if len(f0) < 2 and len(rms) < 2:
        return evidence

    # 1. 鼻音的F0存在且稳定
    voiced = frame_features.get("voiced", np.array([], dtype=bool))
    f0_vals = f0[voiced] if len(f0) > 0 and len(voiced) > 0 else np.array([])
    
    if len(f0_vals) > 0:
        f0_mean = np.mean(f0_vals)
        f0_present = f0_mean > THRESHOLDS["nasal_f0_min"]
    else:
        f0_mean = 0.0
        f0_present = False
    evidence["details"]["f0_present"] = f0_present
    evidence["details"]["f0_mean"] = float(f0_mean)

    # 2. 能量平坦
    if len(rms) > 2:
        energy_cv = np.std(rms) / (np.mean(rms) + 1e-10)
        energy_flat = energy_cv < THRESHOLDS["nasal_energy_flatness"]
    else:
        energy_flat = False
    evidence["details"]["energy_flat"] = energy_flat

    # 3. 共振峰特征：鼻音F1低(~300Hz)，F2低(~1200Hz)
    f1 = formants[0] if len(formants) > 0 else 0
    evidence["details"]["f1"] = f1
    evidence["details"]["f2"] = formants[1] if len(formants) > 1 else 0

    if not (f0_present and energy_flat):
        return evidence

    # 4. 区分 m/n/ng
    # m: F1~300, F2~1200
    # n: F1~350, F2~1600
    # ng: F1~400, F2~2000（ng只做韵尾，韵头不出现）
    f2 = formants[1] if len(formants) > 1 else 0

    if f2 < 1400:
        initial = "m"
    else:
        initial = "n"

    evidence["detected"] = True
    evidence["type"] = "nasal"
    evidence["initial"] = initial
    evidence["score"] = 0.7

    return evidence


def _detect_liquid_type(segment_features: dict, frame_features: dict) -> Dict:
    """
    边音/近音检测 (l/r)
    """
    evidence = {"detected": False, "type": None, "score": 0.0, "details": {}}

    formants = segment_features.get("formants", [0, 0, 0, 0])
    f0 = frame_features.get("f0", np.array([]))

    if len(formants) < 3:
        return evidence

    # 边音 l: F1~400, F2~1500, F3~2500
    # 近音 r: F1~350, F2~1400, F3明显下降~2000
    f1, f2, f3 = formants[0], formants[1], formants[2] if len(formants) > 2 else 0

    evidence["details"]["f1"] = f1
    evidence["details"]["f2"] = f2
    evidence["details"]["f3"] = f3

    # 1. 边音特征：F2在800-2000范围内，F3正常
    is_lateral = (THRESHOLDS["lateral_f1_max"] >= f1 >= 200
                  and THRESHOLDS["lateral_f2_min"] <= f2 <= THRESHOLDS["lateral_f2_max"]
                  and f3 > 2200)
    
    # 2. 卷舌近音 r：F3明显下降（与标准元音相比低200Hz以上）
    is_retroflex = (f3 > 0 and f3 < 2200)

    if is_lateral:
        evidence["detected"] = True
        evidence["type"] = "lateral"
        evidence["initial"] = "l"
        evidence["score"] = 0.65
        evidence["details"]["subtype"] = "lateral"
    elif is_retroflex:
        evidence["detected"] = True
        evidence["type"] = "retroflex"
        evidence["initial"] = "r"
        evidence["score"] = 0.6
        evidence["details"]["subtype"] = "retroflex"

    return evidence


def _check_zero_initial(segment_features: dict, frame_features: dict) -> Dict:
    """
    零声母检测：如果韵母开头直接是元音，没有声母
    """
    rms = frame_features.get("rms_energy", np.array([]))
    f0 = frame_features.get("f0", np.array([]))
    
    evidence = {"detected": False, "type": "null", "score": 0.0, "initial": "", "details": {}}
    
    if len(rms) < 2:
        return evidence
    
    # 零声母特征：能量从第一帧开始就平稳上升（没有爆破/擦音特征）
    # 第一帧RMS相对较高，且没有能量阶跃
    first_rms = rms[0] if rms[0] > 0 else rms[1] if len(rms) > 1 else 0
    evidence["details"]["first_rms"] = float(first_rms)
    
    # 检查是否有爆破特征
    jump = _compute_energy_envelope_jump(rms)
    evidence["details"]["has_plosive_jump"] = jump["has_jump"]
    
    # 检查是否有擦音特征
    sc = frame_features.get("spectral_centroid", np.array([]))
    high_ratio = np.mean(sc > 4000) if len(sc) > 0 else 0.0
    evidence["details"]["high_freq_ratio"] = float(high_ratio)
    
    # F0是否在段首就存在（元音有F0，清声母没有）
    f0_start_voiced = False
    if len(f0) > 3:
        f0_start_voiced = np.mean(f0[:3] > 0) > 0.5
    evidence["details"]["f0_present_at_start"] = f0_start_voiced
    
    # 综合判断：
    # 没有爆破阶跃 + 噪音少 + F0段首存在 = 零声母
    if not jump["has_jump"] and high_ratio < 0.2 and f0_start_voiced:
        evidence["detected"] = True
        evidence["score"] = 0.5
    
    return evidence


def classify_initial(segment_features: dict, frame_features: dict) -> dict:
    """
    声母分类主入口

    执行顺序:
    1. 爆破音检测
    2. 擦音检测
    3. 鼻音检测
    4. 边音/近音检测
    5. 零声母检测

    返回:
    {
        "initial": str,          # 声母标签
        "initial_type": str,     # 类型
        "confidence": float,     # 0~1
        "evidence": dict,        # 证据链（白盒追溯）
    }
    """
    # 按检测优先级依次尝试
    detectors = [
        ("plosive", _detect_plosive_type),
        ("fricative", _detect_fricative_type),
        ("nasal", _detect_nasal_type),
        ("liquid", _detect_liquid_type),
        ("null", _check_zero_initial),
    ]

    results = []
    for type_name, detector in detectors:
        result = detector(segment_features, frame_features)
        if result["detected"]:
            results.append(result)

    # 如果没有检测到任何声母，默认零声母
    if not results:
        return {
            "initial": "",
            "initial_type": "null",
            "confidence": 0.3,
            "evidence": {"no_detector_fired": True},
        }

    # 选择置信度最高的
    best = max(results, key=lambda r: r["score"])

    return {
        "initial": best.get("initial", ""),
        "initial_type": best["type"],
        "confidence": round(best["score"], 4),
        "evidence": best.get("details", {}),
    }
