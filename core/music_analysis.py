"""
听心 · 音乐分析引擎
曲式结构 + 声线分析 + 高潮点 + 为什么会火
纯本地信号处理，无联网调用
"""
import numpy as np
import librosa
import re
from collections import Counter


# ============================================================
# 音乐结构检测
# ============================================================
class StructureDetector:
    """基于能量 + 自相似性的曲式结构切分"""

    def __init__(self):
        self.hop = 512
        self.HISTOGRAM_BINS = 8  # 能量量化等级

    def detect(self, audio: np.ndarray, sr: int, segments: list = None) -> list[dict]:
        """
        检测曲式结构
        返回: [{"label":"前奏","start":0,"end":8},{"label":"主歌","start":8,"end":35},...]
        """
        # 1. 计算能量包络
        rms = librosa.feature.rms(y=audio, hop_length=self.hop)[0]
        times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=self.hop)
        duration = len(audio) / sr

        # 2. 用能量变化 + 光谱变化检测边界
        # 光谱通量
        spec = np.abs(librosa.stft(audio, hop_length=self.hop))
        onset_env = librosa.onset.onset_strength(S=librosa.amplitude_to_db(spec), sr=sr, hop_length=self.hop)

        # 归一化
        rms_norm = (rms - rms.min()) / max(float(np.ptp(rms)), 1e-8)
        onset_norm = (onset_env - onset_env.min()) / max(float(np.ptp(onset_env)), 1e-8)

        # 合成检测信号
        combined = rms_norm[:len(onset_norm)] * 0.6 + onset_norm * 0.4

        # 3. 用滑动窗口检测突变边界
        boundaries = self._find_boundaries(combined, sr)

        # 4. 每个片段的特征统计 → 标签
        sections = []
        prev_b = 0
        for b in boundaries + [len(combined)-1]:
            start_t = prev_b * self.hop / sr
            end_t = min(b * self.hop / sr, duration)
            if end_t <= start_t + 1:  # 跳过太短的段
                prev_b = b
                continue

            s_idx = max(0, prev_b)
            e_idx = min(b+1, len(rms))
            seg_rms = rms_norm[s_idx:e_idx]
            seg_spec = combined[s_idx:e_idx]

            label = self._classify_section(seg_rms, seg_spec, start_t, duration, len(sections))

            sections.append({
                "label": label,
                "start": round(start_t, 1),
                "end": round(end_t, 1),
                "duration": round(end_t - start_t, 1),
                "energy_mean": round(float(rms[s_idx:e_idx].mean()), 4),
            })
            prev_b = b

        # 如果没检测到结构或只有1段，按简单规则分
        if len(sections) <= 2:
            sections = self._fallback_segments(audio, sr, duration)

        # 合并相邻同类标签
        sections = self._merge_adjacent(sections)

        return sections

    def _find_boundaries(self, signal: np.ndarray, sr: int) -> list[int]:
        """找突变边界"""
        # 滑动窗口方差变化
        win = max(10, len(signal) // 50)
        diff = np.zeros(len(signal))
        for i in range(win, len(signal)-win):
            before = signal[i-win:i].var()
            after = signal[i:i+win].var()
            diff[i] = abs(before - after)

        diff = diff / max(diff.max(), 1e-8)
        # 找峰值
        peaks = []
        window = win // 2
        for i in range(window, len(diff)-window):
            if diff[i] > 0.15 and diff[i] == diff[i-window:i+window+1].max():
                peaks.append(i)

        # 降采样：间隔 > 最小段长(5s)
        min_gap = int(5 * sr / self.hop)
        filtered = [peaks[0]] if peaks else []
        for p in peaks[1:]:
            if p - filtered[-1] > min_gap:
                filtered.append(p)

        # 至少3段
        if len(filtered) < 2:
            # 等分
            n_parts = max(3, int(len(signal) * self.hop / sr / 15))
            step = len(signal) // n_parts
            filtered = [i * step for i in range(1, n_parts)]

        return filtered

    def _classify_section(self, seg_rms, seg_spec, start, duration, index) -> str:
        """根据能量和位置判断段落类型"""
        energy_profile = seg_rms.mean()
        variance = seg_rms.std()
        pos = start / max(duration, 1)

        # 开头5% → 前奏
        if pos < 0.08 and energy_profile < 0.4:
            return "前奏"

        # 结尾15% → 尾奏
        if pos > 0.85:
            if energy_profile < 0.3:
                return "尾奏"
            return "副歌"

        # 高能量 + 高变化 → 副歌
        if energy_profile > 0.6 and variance > 0.2:
            return "副歌"

        # 中低能量 → 主歌/间奏
        if energy_profile < 0.35:
            # 靠近开头但能量低 → 间奏
            if 0.08 <= pos < 0.25:
                return "主歌"
            return "间奏"

        # 中能量 → 主歌
        if energy_profile < 0.55:
            return "主歌"

        # 默认
        return f"第{index+1}段"

    def _fallback_segments(self, audio: np.ndarray, sr: int, duration: float) -> list[dict]:
        """兜底：按固定比例分"""
        parts = [
            ("前奏", 0, min(15, duration*0.1)),
            ("主歌", None, None),
            ("副歌", None, None),
            ("主歌", None, None),
            ("副歌", None, None),
            ("尾奏", None, None),
        ]
        sections = []
        total = duration
        # 按比例分配
        ratios = [0.08, 0.25, 0.15, 0.22, 0.18, 0.12]
        start_t = 0
        for label, _, _ in parts:
            dur = total * ratios[len(sections)]
            end_t = start_t + dur
            s_idx = int(start_t * sr)
            e_idx = int(end_t * sr)
            rms_val = float(librosa.feature.rms(y=audio[s_idx:e_idx])[0].mean()) if e_idx > s_idx else 0
            sections.append({
                "label": label,
                "start": round(start_t, 1),
                "end": round(end_t, 1),
                "duration": round(dur, 1),
                "energy_mean": rms_val,
            })
            start_t = end_t
        return sections

    def _merge_adjacent(self, sections: list[dict]) -> list[dict]:
        """合并相邻同标签段落"""
        if not sections: return sections
        merged = [sections[0]]
        for s in sections[1:]:
            if s["label"] == merged[-1]["label"]:
                merged[-1]["end"] = s["end"]
                merged[-1]["duration"] = round(merged[-1]["end"] - merged[-1]["start"], 1)
                merged[-1]["energy_mean"] = round(
                    (merged[-1]["energy_mean"] + s["energy_mean"]) / 2, 4
                )
            else:
                merged.append(s)
        return merged


# ============================================================
# 声线分析
# ============================================================
class VocalAnalyzer:
    """歌手声线/音色特征分析"""

    def analyze(self, audio: np.ndarray, sr: int) -> dict:
        """返回声线特征字典"""
        duration = len(audio) / sr
        # 音高追踪
        f0, voiced, _ = librosa.pyin(y=audio, fmin=65, fmax=1200, sr=sr, hop_length=512, fill_na=0.0)

        # 只取有声段
        f0_vals = f0[voiced & (f0 > 0)]
        if len(f0_vals) == 0:
            return {"error": "未检测到人声", "pitch_range": [0, 0], "pitch_mean": 0}

        pitch_min = float(f0_vals.min())
        pitch_max = float(f0_vals.max())
        pitch_mean = float(f0_vals.mean())
        pitch_std = float(f0_vals.std())

        # 音色特征
        spectral_centroid = float(librosa.feature.spectral_centroid(y=audio, sr=sr).mean())
        spectral_rolloff = float(librosa.feature.spectral_rolloff(y=audio, sr=sr).mean())
        spectral_bandwidth = float(librosa.feature.spectral_bandwidth(y=audio, sr=sr).mean())
        zcr = float(librosa.feature.zero_crossing_rate(y=audio).mean())

        # 音色分类
        timbre = self._classify_timbre(pitch_mean, spectral_centroid, zcr)
        # 性别估计
        if pitch_mean < 160:
            gender = "男声"
        elif pitch_mean < 220:
            gender = "中音"
        else:
            gender = "女声"

        # 声区范围命名
        pitch_range_cents = 1200 * np.log2(max(pitch_max, 1) / max(pitch_min, 1))
        if pitch_range_cents < 600:
            range_label = "窄声区"
        elif pitch_range_cents < 1200:
            range_label = "中声区"
        else:
            range_label = "宽声区"

        return {
            "pitch_range": [round(pitch_min, 1), round(pitch_max, 1)],
            "pitch_range_label": range_label,
            "pitch_mean": round(pitch_mean, 1),
            "pitch_std": round(pitch_std, 1),
            "fundamental_hz": round(pitch_mean, 1),
            "gender": gender,
            "timbre": timbre,
            "spectral_centroid": round(spectral_centroid, 1),
            "spectral_rolloff": round(spectral_rolloff, 1),
            "spectral_bandwidth": round(spectral_bandwidth, 1),
            "zcr": round(zcr, 4),
            "max_f0": round(pitch_max, 1),
            "min_f0": round(pitch_min, 1),
        }

    def _classify_timbre(self, pitch_mean: float, centroid: float, zcr: float) -> str:
        """基于声学特征判断音色类型"""
        # 亮/暗
        bright = centroid > 2500 and zcr > 0.1
        # 厚/薄
        thick = pitch_mean < 180 and centroid < 2000

        if bright and not thick: return "明亮型"
        if thick and not bright: return "浑厚型"
        if bright and thick: return "饱满型"
        if centroid < 1500: return "低沉型"
        return "通透型"


# ============================================================
# 高潮点检测
# ============================================================
class ClimaxDetector:
    """精确标注高潮能量峰值"""

    def detect(self, audio: np.ndarray, sr: int, top_n: int = 3) -> list[dict]:
        """返回能量峰值点列表"""
        hop = 512
        rms = librosa.feature.rms(y=audio, hop_length=hop)[0]
        times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop)

        # 平滑
        kernel = np.ones(5) / 5
        rms_smooth = np.convolve(rms, kernel, mode="same")

        # 找局部峰值
        peaks = []
        window = max(10, len(rms) // 100)
        for i in range(window, len(rms_smooth) - window):
            if rms_smooth[i] == rms_smooth[i-window:i+window+1].max() and rms_smooth[i] > rms_smooth.mean() * 1.5:
                peaks.append((times[i], rms_smooth[i]))

        # 去重（间隔 > 5秒）
        peaks = sorted(peaks, key=lambda x: -x[1])
        result = []
        forbidden = set()
        for t, e in peaks:
            overlap = False
            for ft in forbidden:
                if abs(t - ft) < 5:
                    overlap = True
                    break
            if not overlap:
                result.append({"time": round(t, 1), "energy": round(float(e), 4), "rank": len(result)+1})
                forbidden.add(t)
                if len(result) >= top_n:
                    break

        return result


# ============================================================
# "为什么会火"分析
# ============================================================
class PopularityAnalyzer:
    """基于信号特征猜测歌曲走红原因"""

    def analyze(self, audio: np.ndarray, sr: int, structure: list[dict],
                vocal: dict, climax: list[dict]) -> list[dict]:
        findings = []

        duration = len(audio) / sr

        # 1. 副歌重复次数
        chorus_count = sum(1 for s in structure if "副歌" in s["label"])
        if chorus_count >= 3:
            findings.append({
                "factor": "副歌重复",
                "detail": f"副歌出现{chorus_count}次，强记忆重复驱动传播力",
                "confidence": "高",
            })
        elif chorus_count >= 2:
            findings.append({
                "factor": "副歌重复",
                "detail": f"副歌出现{chorus_count}次，适度重复",
                "confidence": "中",
            })

        # 2. 人声频段
        if vocal and "pitch_mean" in vocal and vocal["pitch_mean"] > 0:
            v_mean = vocal.get("pitch_mean", 200)
            centroid = vocal.get("spectral_centroid", 2000)
            # 人声主频在1-4kHz为"耳虫"频段
            if 1500 <= centroid <= 4000:
                findings.append({
                    "factor": "人声频段优势",
                    "detail": f"人声频谱重心{centroid}Hz落在耳敏感区(1-4kHz)，辨识度高",
                    "confidence": "高",
                })
            elif centroid < 1500:
                findings.append({
                    "factor": "低频质感",
                    "detail": f"人声重心{centroid}Hz偏低频，营造沉浸感",
                    "confidence": "中",
                })

        # 3. 动态范围
        if len(audio) > sr:
            rms_total = librosa.feature.rms(y=audio, hop_length=512)[0]
            dynamic_range = librosa.amplitude_to_db(rms_total.max() / max(rms_total.min(), 1e-8))
            if dynamic_range > 30:
                findings.append({
                    "factor": "强动态对比",
                    "detail": f"动态范围{dynamic_range:.0f}dB，主歌副歌层次分明",
                    "confidence": "高",
                })
            elif dynamic_range > 20:
                findings.append({
                    "factor": "动态适度",
                    "detail": f"动态范围{dynamic_range:.0f}dB，层次清晰",
                    "confidence": "中",
                })

        # 4. 时长分析（3-4分钟流行歌标准）
        if 180 <= duration <= 260:
            findings.append({
                "factor": "黄金时长",
                "detail": f"歌曲长度{duration:.0f}s(约{duration/60:.0f}分)，符合流媒体黄金时长",
                "confidence": "高",
            })
        elif 150 <= duration <= 180:
            findings.append({
                "factor": "短曲风",
                "detail": f"歌曲长度{duration:.0f}s，适合短视频传播",
                "confidence": "中",
            })

        # 5. 高潮强度
        if climax and climax[0]["energy"] > 0.3:
            findings.append({
                "factor": "高潮爆发力",
                "detail": f"峰值能量{climax[0]['energy']:.3f}，高潮点显著突出",
                "confidence": "高",
            })

        # 6. 节奏密度
        onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
        if onset_env.max() > 0:
            density = len(librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)) / duration
            if density > 2:
                findings.append({
                    "factor": "节奏密集",
                    "detail": f"每秒{density:.1f}个节拍点，节奏驱动性强",
                    "confidence": "中",
                })

        return findings


# ============================================================
# 主入口
# ============================================================
class MusicAnalyzer:
    """音乐分析全流程"""

    def __init__(self):
        self.structure = StructureDetector()
        self.vocal = VocalAnalyzer()
        self.climax = ClimaxDetector()
        self.popularity = PopularityAnalyzer()

    def analyze(self, filepath: str, audio: np.ndarray = None, sr: int = 16000) -> dict:
        """分析一首歌"""
        if audio is None:
            from .preprocess import load_audio
            audio = load_audio(filepath, sr=sr)
        sr = sr  # 16kHz already

        duration = len(audio) / sr

        # 重采样到16kHz（如果原文件是44.1kHz的会丢失高频，但不影响结构分析）
        # 音乐信号建议检测用原始采样率，但为了统一用16kHz

        # 各部检测
        structure = self.structure.detect(audio, sr)
        vocal = self.vocal.analyze(audio, sr)
        climax_points = self.climax.detect(audio, sr)
        popularity = self.popularity.analyze(audio, sr, structure, vocal, climax_points)

        # 副歌统计
        chorus_sections = [s for s in structure if "副歌" in s["label"]]
        total_chorus_time = sum(s["duration"] for s in chorus_sections)
        chorus_pct = round(total_chorus_time / max(duration, 1) * 100, 1)

        return {
            "file": filepath,
            "duration_sec": round(duration, 1),
            "duration_label": f"{int(duration//60)}分{int(duration%60)}秒",
            "structure": structure,
            "structure_summary": self._summarize_structure(structure),
            "vocal": vocal,
            "climax": climax_points,
            "chorus_count": len(chorus_sections),
            "chorus_time_pct": chorus_pct,
            "popularity_factors": popularity,
            "popularity_summary": self._summarize_popularity(popularity),
        }

    def _summarize_structure(self, structure: list[dict]) -> str:
        """生成结构摘要"""
        parts = [f"{s['label']} {s['duration']:.0f}s" for s in structure]
        return " → ".join(parts)

    def _summarize_popularity(self, factors: list[dict]) -> str:
        if not factors:
            return "无明显特征"
        return "; ".join(f["detail"] for f in factors[:3])

    def to_markdown(self, data: dict) -> str:
        """生成可读报告"""
        lines = []
        fname = data.get("file", "")
        lines.append(f"# 音乐分析报告")
        lines.append(f"## {fname}")
        lines.append(f"- 时长: {data.get('duration_label', '?')}")
        lines.append("")

        # 曲式结构
        lines.append("## 曲式结构")
        lines.append(f"{data.get('structure_summary', '?')}")
        lines.append("")
        lines.append("| 段落 | 开始 | 结束 | 时长 | 能量 |")
        lines.append("|------|------|------|------|------|")
        for s in data.get("structure", []):
            lines.append(f"| {s['label']} | {s['start']}s | {s['end']}s | {s['duration']}s | {s.get('energy_mean','')} |")
        lines.append(f"\n副歌出现 **{data.get('chorus_count','?')}** 次，占比 **{data.get('chorus_time_pct','?')}%**")
        lines.append("")

        # 声线分析
        lines.append("## 歌手声线分析")
        v = data.get("vocal", {})
        if v.get("error"):
            lines.append(f"- {v['error']}")
        else:
            lines.append(f"- 性别倾向: **{v.get('gender','?')}**")
            lines.append(f"- 音色: **{v.get('timbre','?')}**")
            lines.append(f"- 音高范围: {v.get('min_f0','?')}Hz ~ {v.get('max_f0','?')}Hz ({v.get('pitch_range_label','?')})")
            lines.append(f"- 基频均值: {v.get('fundamental_hz','?')}Hz")
            lines.append(f"- 频谱重心: {v.get('spectral_centroid','?')}Hz")
            lines.append(f"- 频谱带宽: {v.get('spectral_bandwidth','?')}Hz")
        lines.append("")

        # 高潮点
        lines.append("## 高潮点")
        for c in data.get("climax", []):
            lines.append(f"- **{c['time']}s** (能量{c['energy']:.3f})")
        lines.append("")

        # 为什么会火
        lines.append("## 为什么会火")
        for f in data.get("popularity_factors", []):
            lines.append(f"### {f['factor']} [{f['confidence']}]")
            lines.append(f"> {f['detail']}")
            lines.append("")

        lines.append("---")
        lines.append("*报告由 听心·音乐分析引擎 生成*")
        return "\n".join(lines)
