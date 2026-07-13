"""
L5 白盒声学拼音识别引擎 — 总装

将 L1~L4 串联为完整流水线：
音频 → 预处理 → 特征提取 → 声母/韵母/声调分类 → 拼音融合 → → 汉字
"""

import time
import numpy as np
from typing import Dict, List, Optional

# 支持直接运行和模块导入两种模式
import importlib
_pkg = __name__.rpartition('.')[0] or 'whitebox_asr'

try:
    from .audio_loader import load_and_vad, load_and_vad_from_array
    from .primitive_extractor import extract_frame_features, extract_segment_features
    from .initial_classifier import classify_initial
    from .final_classifier import classify_final
    from .tone_classifier import classify_tone
    from .syllable_fuser import fuse_syllable
    from .pinyin_to_hanzi import PinyinToHanzi
except ImportError:
    from audio_loader import load_and_vad, load_and_vad_from_array
    from primitive_extractor import extract_frame_features, extract_segment_features
    from initial_classifier import classify_initial
    from final_classifier import classify_final
    from tone_classifier import classify_tone
    from syllable_fuser import fuse_syllable
    from pinyin_to_hanzi import PinyinToHanzi


class WhiteboxASR:
    """
    白盒声学拼音识别引擎

    无需GPU、无需神经网络、完全规则驱动。
    每个结果可回溯到声学特征和决策路径。

    用法:
        asr = WhiteboxASR()
        result = asr.transcribe("audio.wav")
        print(result["text"])
    """

    def __init__(self, user_dict: dict = None, templates: dict = None,
                 debug: bool = False):
        """
        参数:
            user_dict: 用户自定义词库 { "pinyin": ["汉字1", ...] }
            templates: 自定义模板（用于韵母匹配）
            debug: 默认是否保留调试信息
        """
        self._hanzi = PinyinToHanzi(user_dict=user_dict)
        self._templates = templates or {}
        self._debug = debug

    def transcribe(self, audio_path: str, debug: bool = None) -> dict:
        """
        全流水线：音频 → 拼音 → 汉字

        参数:
            audio_path: 音频文件路径
            debug: 是否保留白盒调试信息（覆盖默认）

        返回:
        {
            "text": str,             # 最终汉字文本
            "pinyin_raw": [str],     # 逐音节拼音
            "pinyin_segments": [     # 逐段详细结果
                {
                    "segment_index": int,
                    "start_s": float,
                    "end_s": float,
                    "syllables": [
                        {
                            "pinyin": str,
                            "pinyin_numbered": str,
                            "initial": str,
                            "final": str,
                            "tone": int,
                            "confidence": float,
                            "evidence": dict,
                        }
                    ],
                    "text": str,
                }
            ],
            "timing": { ... },       # 各阶段耗时
            "debug": dict or None,   # 白盒追溯
        }
        """
        if debug is None:
            debug = self._debug

        t0 = time.time()

        # ===== L1: 音频加载 + VAD =====
        t1_l1 = time.time()
        vad_result = load_and_vad(audio_path)
        segments = vad_result["segments"]
        t_l1 = time.time() - t1_l1

        if not segments:
            return {
                "text": "",
                "pinyin_raw": [],
                "pinyin_segments": [],
                "timing": {"l1_vad": t_l1, "total": time.time() - t0},
                "debug": {"no_voice_segments": True} if debug else None,
            }

        # ===== L2+L3: 逐段处理 =====
        t1_l2 = time.time()
        all_syllables = []       # 扁平拼音列表
        segment_results = []     # 逐段结果
        all_f0_vals = []         # 全局F0（用于声调归一化）

        for seg in segments:
            audio = seg["audio"]
            sr = vad_result["sr"]

            # L2: 帧特征提取
            frames = extract_frame_features(audio, sr)

            # L2: 片段特征聚合
            seg_feat = extract_segment_features(frames)

            # 保存全局F0统计
            voiced_f0 = frames["f0"][frames["voiced"]] if frames["n_frames"] > 0 else np.array([])
            if len(voiced_f0) > 0:
                all_f0_vals.extend(voiced_f0.tolist())

            # 空段跳过
            if frames["n_frames"] == 0:
                segment_results.append({
                    "segment_index": seg["index"],
                    "start_s": seg["start_s"],
                    "end_s": seg["end_s"],
                    "syllables": [],
                    "text": "",
                    "skip_reason": "empty",
                })
                continue

            # L3.1: 声母分类
            initial_result = classify_initial(seg_feat, frames)

            # L3.2: 韵母分类
            final_result = classify_final(seg_feat, frames, templates=self._templates)

            # L3.3: 声调分类
            f0_contour = seg_feat.get("pitch_contour", np.array([]))
            tone_result = classify_tone(
                f0_contour, segment_features=seg_feat
            )

            # L3.4: 音节融合
            syllable = fuse_syllable(
                initial_result, final_result, tone_result,
                debug=debug
            )

            all_syllables.append(syllable)

            # 构建段结果
            seg_syllables = [syllable]
            seg_text = ""
            if syllable["pinyin_numbered"]:
                pinyin_decode = self._hanzi.decode([syllable["pinyin_numbered"]])
                seg_text = pinyin_decode["text"]

            segment_results.append({
                "segment_index": seg["index"],
                "start_s": seg["start_s"],
                "end_s": seg["end_s"],
                "syllables": seg_syllables,
                "text": seg_text,
            })

        t_l2_l3 = time.time() - t1_l2

        # 计算全局F0中位数
        global_f0_median = float(np.median(all_f0_vals)) if all_f0_vals else 150.0

        # 对每个音节用全局F0重新修正声调（可选）
        # 当前简化实现

        # ===== L4: 拼音→汉字 =====
        t1_l4 = time.time()
        pinyin_list = [s["pinyin_numbered"] for s in all_syllables 
                      if s.get("pinyin_numbered") and s["pinyin_numbered"].strip()]
        if not pinyin_list:
            hanzi_result = {"text": "", "pinyin": [], "segments": []}
        else:
            try:
                hanzi_result = self._hanzi.decode(pinyin_list)
            except Exception:
                # 解码失败时降级：直接拼接拼音
                hanzi_result = {"text": " ".join(pinyin_list), "pinyin": pinyin_list, "segments": []}
        t_l4 = time.time() - t1_l4

        total_time = time.time() - t0

        return {
            "text": hanzi_result["text"],
            "pinyin_raw": [s["pinyin"] for s in all_syllables],
            "pinyin_numbered": pinyin_list,
            "pinyin_segments": segment_results,
            "hanzi_detail": hanzi_result,
            "timing": {
                "l1_vad": round(t_l1, 3),
                "l2_l3_acoustic_pinyin": round(t_l2_l3, 3),
                "l4_hanzi": round(t_l4, 3),
                "total": round(total_time, 3),
            },
            "debug": {
                "vad": vad_result if debug else None,
                "segments": len(segments),
                "syllables": len(all_syllables),
                "global_f0_median": global_f0_median,
            } if debug else None,
        }

    def transcribe_to_pinyin(self, audio_path: str) -> List[str]:
        """
        仅输出拼音串（调试用）

        返回: ["ni3", "hao3", ...]
        """
        result = self.transcribe(audio_path, debug=False)
        return result.get("pinyin_numbered", [])

    def transcribe_stream(self, audio_chunks: List[np.ndarray],
                          sr: int = 16000) -> List[str]:
        """
        逐段转写（流式场景）

        参数:
            audio_chunks: 音频段列表
            sr: 采样率

        返回: 拼音列表
        """
        all_pinyin = []

        for chunk in audio_chunks:
            vad_result = load_and_vad_from_array(chunk, sr=sr)
            for seg in vad_result["segments"]:
                audio = seg["audio"]
                frames = extract_frame_features(audio, sr)
                seg_feat = extract_segment_features(frames)

                if frames["n_frames"] == 0:
                    continue

                init_r = classify_initial(seg_feat, frames)
                final_r = classify_final(seg_feat, frames)
                tone_r = classify_tone(
                    seg_feat.get("pitch_contour", np.array([])),
                    segment_features=seg_feat
                )
                syllable = fuse_syllable(init_r, final_r, tone_r)

                if syllable["pinyin_numbered"]:
                    all_pinyin.append(syllable["pinyin_numbered"])

        return all_pinyin
