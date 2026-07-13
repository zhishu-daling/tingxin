"""
L3.3 声调分类公式（白盒）

基于 F0 曲线的声调决策树。
- 第一声(55): 高平
- 第二声(35): 上升
- 第三声(214): 降升
- 第四声(51): 下降
- 轻声: 短+低+平坦

白盒式：每个声调判断可回溯到F0曲线的统计量。
"""

import numpy as np
from typing import Dict, Optional, List


def _clean_f0_contour(f0: np.ndarray, voiced: Optional[np.ndarray] = None) -> np.ndarray:
    """
    F0序列清理：去野点+插值
    
    如果是无声段，返回空数组。
    """
    if len(f0) == 0:
        return np.array([])
    
    if voiced is not None:
        f0_clean = f0.copy()
        f0_clean[~voiced] = 0.0
    else:
        f0_clean = f0.copy()
    
    # 剔除异常值 (<50Hz 或 >600Hz)
    f0_clean[f0_clean < 50] = 0.0
    f0_clean[f0_clean > 600] = 0.0
    
    # 只保留有声段
    voiced_frames = f0_clean > 0
    if np.sum(voiced_frames) < 3:
        return np.array([])
    
    # 进一步去野点（3-sigma）
    f0_vals = f0_clean[voiced_frames]
    mean_f0 = np.mean(f0_vals)
    std_f0 = np.std(f0_vals)
    if std_f0 > 0:
        mask = np.abs(f0_vals - mean_f0) < 3 * std_f0
        f0_vals = f0_vals[mask]
    
    return f0_vals


def _compute_tone_features(f0_contour: np.ndarray) -> Dict:
    """
    计算声调特征向量
    
    返回:
    {
        "f0_start": float,      # 起点F0
        "f0_end": float,        # 终点F0
        "f0_max": float,        # 最大F0
        "f0_min": float,        # 最小F0
        "f0_mean": float,       # 均值
        "f0_std": float,        # 标准差
        "f0_range": float,      # 范围 (max-min)
        "slope": float,         # 线性回归斜率 (Hz/帧)
        "is_rising": bool,      # 是否上升趋势
        "is_falling": bool,     # 是否下降趋势
        "has_valley": bool,     # 是否有谷点（先降后升）
        "valley_position": float, # 谷点位置比例 (0~1)
        "valley_depth": float,  # 谷点深度 (均值 - 谷点)
        "duration_frames": int,
    }
    """
    if len(f0_contour) < 3:
        return {
            "f0_start": 0, "f0_end": 0, "f0_max": 0, "f0_min": 0,
            "f0_mean": 0, "f0_std": 0, "f0_range": 0,
            "slope": 0, "is_rising": False, "is_falling": False,
            "has_valley": False, "valley_position": 0, "valley_depth": 0,
            "duration_frames": 0,
        }
    
    start = f0_contour[0]
    end = f0_contour[-1]
    max_val = np.max(f0_contour)
    min_val = np.min(f0_contour)
    mean_val = np.mean(f0_contour)
    std_val = np.std(f0_contour)
    
    # 线性回归
    x = np.arange(len(f0_contour))
    slope, _ = np.polyfit(x, f0_contour, 1)
    
    # 趋势
    is_rising = (end - start) > 15  # Hz
    is_falling = (start - end) > 15  # Hz
    
    # 谷点检测（用于第三声 214）
    has_valley = False
    valley_pos = 0.0
    valley_depth = 0.0
    
    if len(f0_contour) >= 5:
        # 找最小值位置
        min_idx = np.argmin(f0_contour)
        # 谷点不能太靠边（不能在前20%或后20%）
        min_ratio = min_idx / len(f0_contour)
        if 0.2 < min_ratio < 0.8:
            # 检查是否真的是"先降后升"
            before_min = f0_contour[:min_idx]
            after_min = f0_contour[min_idx:]
            if len(before_min) > 1 and len(after_min) > 1:
                # 下降段：起点到谷点，应下降 > 15 Hz
                drop = before_min[0] - min_val
                # 上升段：谷点到终点，应上升 > 15 Hz
                rise = after_min[-1] - min_val
                if drop > 15 and rise > 15:
                    has_valley = True
                    valley_pos = min_ratio
                    valley_depth = mean_val - min_val
    
    return {
        "f0_start": float(start),
        "f0_end": float(end),
        "f0_max": float(max_val),
        "f0_min": float(min_val),
        "f0_mean": float(mean_val),
        "f0_std": float(std_val),
        "f0_range": float(max_val - min_val),
        "slope": float(slope),
        "is_rising": bool(is_rising),
        "is_falling": bool(is_falling),
        "has_valley": bool(has_valley),
        "valley_position": float(valley_pos),
        "valley_depth": float(valley_depth),
        "duration_frames": len(f0_contour),
    }


def classify_tone(f0_contour: np.ndarray,
                  segment_features: dict = None,
                  global_f0_median: float = None) -> dict:
    """
    声调分类决策树

    参数:
        f0_contour: 清理后的F0序列 (仅有声段)
        segment_features: 片段特征（可选，用于轻声音量判断）
        global_f0_median: 全局F0中位数（用于归一化）

    返回:
    {
        "tone": int,              # 1/2/3/4/5(轻声)/0(无法判断)
        "tone_label": str,        # "55","35","214","51","轻声"
        "confidence": float,
        "features": {...},        # F0统计特征
        "evidence": {...},        # 决策证据
    }
    """
    # 默认返回
    default_result = {
        "tone": 0,
        "tone_label": "unknown",
        "confidence": 0.0,
        "features": {},
        "evidence": {"reason": "insufficient_f0"},
    }
    
    if len(f0_contour) < 3:
        return default_result
    
    # 计算特征
    feat = _compute_tone_features(f0_contour)
    if feat["duration_frames"] == 0:
        return default_result
    
    # 全局中位数（用于"高平"判断）
    if global_f0_median is None:
        global_f0_median = feat["f0_mean"]
    
    evidence = {}
    
    # ===== 决策树 =====
    
    # 轻声检测：短 + 低能量 + 平坦
    is_light = False
    if segment_features:
        duration = segment_features.get("duration_s", 1.0)
        energy_mean = segment_features.get("energy_mean", 0.5)
        if duration < 0.15 and energy_mean < 0.1:
            is_light = True
            evidence["light_tone"] = {
                "duration_s": duration,
                "energy_mean": energy_mean,
                "reason": "short_and_low_energy"
            }
    
    if is_light:
        return {
            "tone": 5,
            "tone_label": "轻声",
            "confidence": 0.6,
            "features": feat,
            "evidence": evidence,
        }
    
    # 第一声 (55) - 高平
    # 条件：均值高 + 标准差小
    is_tone1 = (feat["f0_mean"] > global_f0_median * 0.9
                and feat["f0_std"] < 0.2 * feat["f0_mean"]
                and not feat["is_rising"]
                and not feat["is_falling"])
    
    if is_tone1:
        evidence["tone1"] = {
            "f0_mean": feat["f0_mean"],
            "f0_std": feat["f0_std"],
            "global_median": global_f0_median,
            "reason": "high_and_flat"
        }
        return {
            "tone": 1,
            "tone_label": "55",
            "confidence": min(1.0, (0.2 * feat["f0_mean"]) / (feat["f0_std"] + 1) * 0.5),
            "features": feat,
            "evidence": evidence,
        }
    
    # 第三声 (214) - 降升（优先于纯升降，因为有谷点特征更明确）
    if feat["has_valley"]:
        evidence["tone3"] = {
            "has_valley": True,
            "valley_position": feat["valley_position"],
            "valley_depth": feat["valley_depth"],
            "reason": "fall_then_rise_with_valley"
        }
        return {
            "tone": 3,
            "tone_label": "214",
            "confidence": min(1.0, feat["valley_depth"] / 30 * 0.8),
            "features": feat,
            "evidence": evidence,
        }
    
    # 第四声 (51) - 下降
    if feat["is_falling"] and feat["slope"] < -1.0:
        evidence["tone4"] = {
            "f0_start": feat["f0_start"],
            "f0_end": feat["f0_end"],
            "slope": feat["slope"],
            "reason": "sharp_fall"
        }
        return {
            "tone": 4,
            "tone_label": "51",
            "confidence": min(1.0, (feat["f0_start"] - feat["f0_end"]) / 40),
            "features": feat,
            "evidence": evidence,
        }
    
    # 第二声 (35) - 上升
    if feat["is_rising"] and feat["slope"] > 1.0:
        evidence["tone2"] = {
            "f0_start": feat["f0_start"],
            "f0_end": feat["f0_end"],
            "slope": feat["slope"],
            "reason": "steady_rise"
        }
        return {
            "tone": 2,
            "tone_label": "35",
            "confidence": min(1.0, (feat["f0_end"] - feat["f0_start"]) / 40),
            "features": feat,
            "evidence": evidence,
        }
    
    # Fallback: 最接近的声调
    # 计算到各个声调的距离
    scores = {
        1: max(0, 1.0 - feat["f0_std"] / (feat["f0_mean"] * 0.3)),
        2: max(0, (feat["f0_end"] - feat["f0_start"]) / 50),
        4: max(0, (feat["f0_start"] - feat["f0_end"]) / 50),
    }
    
    # 如果有谷点特征但不够强，可能是第三声
    if np.argmin(f0_contour) > 1 and np.argmin(f0_contour) < len(f0_contour) - 2:
        scores[3] = 0.3
    
    best_tone = max(scores, key=scores.get)
    best_score = scores[best_tone]
    
    evidence["fallback"] = {
        "scores": scores,
        "best_tone": best_tone,
        "reason": "no_clear_match"
    }
    
    return {
        "tone": best_tone,
        "tone_label": ["", "55", "35", "214", "51"][best_tone],
        "confidence": max(0.15, best_score * 0.5),
        "features": feat,
        "evidence": evidence,
    }
