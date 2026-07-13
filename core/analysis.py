"""
听心 · MEITR 五维分析引擎（完整版）

基于你的文档中所有公式体系实现：
1. 情绪三维向量 (A/V/D)
2. 跨模态矛盾检测
3. 双人交互同步性
4. 状态预测 (爆发检测)
5. 对话质量评估
6. 防御机制识别
"""
import numpy as np
from collections import Counter, defaultdict
from .acoustic_features import extract_segment_features


# ========== 1. 单人情绪三维向量（完整证据链） ==========

def _clamp(v, lo=-1, hi=1): return max(lo, min(hi, v))

def compute_arousal(feat):
    """唤醒度 Arousal: tanh(0.4*E + 0.35*P + 0.25*T - 0.1)"""
    if not feat: return 0.0, {}
    E = _clamp(feat.get("energy_mean", 0) * 5, 0, 1)
    P = _clamp(feat.get("pitch_mean", 0) / 300, 0, 1)
    T = _clamp(abs(feat.get("zcr_mean", 0)) * 50, 0, 1)
    val = float(np.tanh(0.4*E + 0.35*P + 0.25*T - 0.1))
    return val, {"energy": round(E,3), "pitch": round(P,3), "tempo": round(T,3)}

def compute_valence(feat):
    """效价 Valence: tanh(0.4*(1-2SF) + 0.35*(1-SC) + 0.25*HNR)"""
    if not feat: return 0.0, {}
    sf = _clamp(np.mean(feat.get("mfcc_std", [0.5]*13)) * 2, 0, 1)
    sc = _clamp(feat.get("spectral_centroid_mean", 2000) / 4000, 0, 1)
    hnr = _clamp(feat.get("pitch_std", 0) / 80, 0, 1)
    val = float(np.tanh(0.4*(1-2*sf) + 0.35*(1-sc) + 0.25*hnr))
    return val, {"flatness": round(sf,3), "centroid": round(sc,3), "hnr": round(hnr,3)}

def compute_dominance(feat):
    """控制度 Dominance: tanh(0.45*(1-PV) + 0.35*(1-EA) + 0.2*SR)"""
    if not feat: return 0.0, {}
    pv = _clamp(feat.get("pitch_std", 0) / 100, 0, 1)
    ea = _clamp(feat.get("energy_std", 0) * 10, 0, 1)
    sr = 1 - min(pv, 1.0)
    val = float(np.tanh(0.45*(1-pv) + 0.35*(1-ea) + 0.2*sr))
    return val, {"pitch_var": round(pv,3), "energy_accel": round(ea,3), "stability": round(sr,3)}

def emotion_label(a, v, d):
    """三维→可读标签+置信度"""
    if a>0.3 and v>0.3: return "兴奋", round(a*v,3)
    if a>0.3 and v<-0.3 and d<-0.3: return "愤怒", round(a*abs(v),3)
    if a>0.3 and v<-0.3 and d>0.3: return "冷蔑视", round(a*abs(v)*d,3)
    if a<-0.3 and v<-0.3: return "低落", round(abs(a)*abs(v),3)
    if a<-0.3 and v>0.3: return "平静", round(abs(a)*v,3)
    if a>0.3 and abs(v)<0.2 and d<-0.3: return "焦虑", round(a*abs(d),3)
    if v<-0.3 and abs(d)<0.3: return "压抑", round(abs(v)*(1-abs(d)),3)
    if a>0.5: return "激动", round(a,3)
    return "中性", round(max(0,1-abs(a)-abs(v)),3)


# ========== 2. 防御机制识别（37种配置） ==========

DEFENSE_MECHS = {
    "压抑": {"A": "H", "V": "L", "D": "H", "conflict": "H"},
    "否认": {"A": "M", "V": "H", "D": "H", "conflict": "VH"},
    "投射": {"A": "H", "V": "L", "D": "H", "keyword": ["你才", "明明是你", "你自己"]},
    "合理化": {"A": "L", "V": "M", "D": "M", "keyword": ["因为", "所以", "毕竟", "没办法"]},
    "理智化": {"A": "L", "V": "M", "D": "VH", "keyword": ["从理论上", "客观来说", "数据"]},
    "幽默化": {"A": "H", "V": "H", "D": "M", "keyword": ["哈哈", "开玩笑"]},
    "回避": {"A": "L", "V": "M", "D": "M", "keyword": ["算了", "不说", "换个话题"]},
    "攻击转向": {"A": "VH", "V": "VH", "D": "L", "keyword": ["都怪你", "全是你的错"]},
    "顺从": {"A": "L", "V": "L", "D": "L", "keyword": ["好吧", "行吧", "那就这样"]},
    "反向形成": {"A": "H", "V": "H", "D": "M", "conflict": "H"},
}

def detect_defense(analysis, text):
    """检测当前片段是否使用了防御机制"""
    a = analysis.get("arousal", 0)
    v = analysis.get("valence", 0)
    d = analysis.get("dominance", 0)
    conflict = analysis.get("conflict_score", 0)

    def level(val):
        if abs(val) > 0.6: return "VH"
        if abs(val) > 0.3: return "H"
        if abs(val) > 0.1: return "M"
        return "L"

    al, vl, dl = level(a), level(v), level(d)
    cl = "VH" if conflict > 0.5 else "H" if conflict > 0.3 else "L"

    best, best_score = None, 0
    for name, cfg in DEFENSE_MECHS.items():
        score = 0
        if cfg.get("A") == al: score += 2
        if cfg.get("V") == vl: score += 2
        if cfg.get("D") == dl: score += 2
        if cfg.get("conflict") == cl: score += 3
        for kw in cfg.get("keyword", []):
            if kw in text: score += 3
        if score > best_score:
            best_score = score
            best = name

    if best_score >= 4:
        return best, round(best_score / 10, 2)
    return None, 0


# ========== 3. 跨模态矛盾检测（完整公式） ==========

def text_sentiment(text):
    """简单文本情感极性 [-1, 1]"""
    pos = sum(1 for w in ["好","是","可以","对","谢谢","开心","喜欢","厉害","棒","同意","行","要"] if w in text)
    neg = sum(1 for w in ["不","没","烦","讨厌","气","错","差","不行","算了","别","不要","滚"] if w in text)
    return (pos - neg) / max(pos + neg, 1) if (pos+neg) > 0 else 0.0

def detect_conflict(text, va, do, aro):
    """冲突分数: |S_text - V_acoustic| * (1-|D|) * |A|"""
    st = text_sentiment(text)
    score = float(abs(st - va) * (1 - abs(do)) * abs(aro))
    # 解释
    if score > 0.4:
        if st > 0 and va < -0.3: exp = f"嘴上说好但声音难受 (矛盾度={score:.2f})"
        elif st < 0 and va > 0.3: exp = f"嘴上说不好但情绪积极 (矛盾度={score:.2f})"
        elif abs(st-va) > 0.5: exp = f"声文显著不一致 (矛盾度={score:.2f})"
        else: exp = None
    else: exp = None
    return round(score, 3), exp, round(st, 3)


# ========== 4. 双人交互同步性 ==========

class InteractionAnalyzer:
    """双人对话交互分析器"""
    def __init__(self):
        self.history = defaultdict(list)  # speaker -> [{time, energy, pitch}]

    def add_segment(self, seg, feat):
        spk = seg.get("speaker", "UNKNOWN")
        self.history[spk].append({
            "time": seg["start"],
            "energy": feat.get("energy_mean", 0) if feat else 0,
            "pitch": feat.get("pitch_mean", 0) if feat else 0,
            "arousal": 0,
        })

    def energy_coupling(self):
        """能量耦合度: Pearson(E_A, E_B)"""
        spks = list(self.history.keys())
        if len(spks) < 2: return 0
        a, b = spks[0], spks[1]
        ea = [s["energy"] for s in self.history[a]]
        eb = [s["energy"] for s in self.history[b]]
        n = min(len(ea), len(eb))
        if n < 3: return 0
        from scipy.stats import pearsonr
        r, _ = pearsonr(ea[:n], eb[:n])
        return round(r, 3) if not np.isnan(r) else 0

    def pitch_following(self):
        """音高追随度"""
        spks = list(self.history.keys())
        if len(spks) < 2: return 0, 0
        a, b = spks[0], spks[1]
        pa = [s["pitch"] for s in self.history[a]]
        pb = [s["pitch"] for s in self.history[b]]
        n = min(len(pa), len(pb))
        if n < 3: return 0, 0
        from scipy.stats import pearsonr
        r, _ = pearsonr(pa[:n], pb[:n])
        return round(r, 3) if not np.isnan(r) else 0, n

    def relation_label(self, coupling):
        """耦合度 → 关系描述"""
        if coupling > 0.5: return "高同步, 对话紧密"
        if coupling > 0.2: return "弱同步, 基本协调"
        if coupling > -0.2: return "无关, 各自独立"
        if coupling > -0.5: return "弱对抗, 有张力"
        return "强对抗, 权力博弈"


# ========== 5. 状态预测（爆发检测） ==========

class StatePredictor:
    """基于 P_suppress 的爆发预测"""
    def __init__(self, alpha=0.95, theta_burst=0.6, theta_dom=0.3):
        self.alpha = alpha
        self.theta_burst = theta_burst
        self.theta_dom = theta_dom
        self.cum_strain = 0
        self.prev_valence = 0

    def update(self, arousal, valence, dominance, conflict):
        dV = abs(valence - self.prev_valence)
        self.cum_strain = self.alpha * self.cum_strain + dV * (1 - abs(dominance))
        self.prev_valence = valence

        p_suppress = abs(arousal) * conflict * (1 + self.cum_strain)

        result = {"p_suppress": round(p_suppress, 3), "cum_strain": round(self.cum_strain, 3)}
        if p_suppress > self.theta_burst and abs(dominance) < self.theta_dom:
            result["alert"] = "爆发风险高 ⚠️"
        elif p_suppress > self.theta_burst and abs(dominance) >= self.theta_dom:
            result["alert"] = "冷暴力风险 ⚠️"
        else:
            result["alert"] = None
        return result


# ========== 6. 对话质量评估 ==========

# ========== 7. 共谋伪装检测（文假声也假） ==========

class SpeakerBaseline:
    """
    每个说话人的个人声纹基线。
    用于"跟TA自己比"而不是"跟全人类比"。

    """
    def __init__(self):
        self.pitch_variance = []    # 累积基频方差
        self.decay_times = []       # 累积衰减时间
        self.micro_pauses = []      # 累积微停顿有无
        self.total_speech_s = 0     # 累计说话时长（秒）

    def update(self, feat: dict, duration_s: float):
        if not feat:
            return
        if feat.get("pitch_std", 0) > 0:
            self.pitch_variance.append(feat["pitch_std"]**2)
        if feat.get("decay_time_s", 0) > 0:
            self.decay_times.append(feat["decay_time_s"])
        self.micro_pauses.append(1 if feat.get("has_micro_pause") else 0)
        self.total_speech_s += duration_s

    @property
    def pitch_variance_baseline(self) -> float:
        if not self.pitch_variance:
            return 0.05  # 默认fallback
        return float(np.median(self.pitch_variance))

    @property
    def decay_time_baseline(self) -> float:
        if not self.decay_times:
            return 0.15  # 默认fallback（秒）
        return float(np.median(self.decay_times))

    @property
    def micro_pause_rate(self) -> float:
        if not self.micro_pauses:
            return 0.5
        return float(np.mean(self.micro_pauses))

    @property
    def confidence(self) -> float:
        """基线置信度：说话越久越可信"""
        return min(1.0, self.total_speech_s / 120)  # 120秒达到全置信


def compute_context_contradiction(idx: int, segments: list) -> float:
    """
    语境矛盾分数。
    看当前句和上下文的文本情感差异。
    前面说"我妈住院了"，现在说"我挺好的" → 矛盾。

    返回: [0, 2]，越高越矛盾
    """
    current = segments[idx]
    current_text = current.get("text", "")
    current_sent = text_sentiment(current_text)

    # 前后各取3句
    context_texts = []
    for offset in range(-3, 4):
        if offset == 0:
            continue
        j = idx + offset
        if 0 <= j < len(segments):
            # 只看同说话人
            if segments[j].get("speaker") == current.get("speaker"):
                context_texts.append(segments[j].get("text", ""))

    if not context_texts:
        return 0.0

    # 如果有分析结果，优先用已有valence；否则用文本情感
    sentiments = []
    for ct in context_texts:
        sentiments.append(text_sentiment(ct))

    context_sent = float(np.mean(sentiments))
    contradiction = abs(current_sent - context_sent)

    return round(contradiction, 3)


def compute_fabrication_score(
    feat: dict,
    baseline: SpeakerBaseline,
    context_contradiction: float,
) -> tuple:
    """
    共谋伪装分数计算。
    """
    if not feat:
        return 0.0, 0.0, {}

    # === 2.1 音高过度稳定 ===
    pitch_var = feat.get("pitch_std", 0) ** 2
    pitch_var_baseline = baseline.pitch_variance_baseline
    pitch_var_norm = min(1.0, pitch_var / (pitch_var_baseline + 1e-15))
    pitch_stability_score = max(0.0, 1 - pitch_var_norm)
    # 正常情况下pitch_var_norm ≈ 1（跟平时差不多），分数=0
    # 如果pitch_var_norm接近0（比平时稳太多），分数接近1

    # === 2.2 能量衰减非自然性 ===
    decay = feat.get("decay_time_s", 0)
    decay_baseline = baseline.decay_time_baseline
    decay_norm = decay / (decay_baseline + 1e-15)
    decay_score = max(0.0, min(1.0, 1 - decay_norm))
    # 收得太快（decay_norm << 1）= 分数高

    # === 2.3 微停顿缺失 ===
    has_pause = feat.get("has_micro_pause", False)
    pause_rate = baseline.micro_pause_rate
    # 如果这个人平时50%会说停，现在没停 → 异常
    # 如果这个人平时就10%会说停 → 也算异常但惩罚小
    pause_absence_score = (0 if has_pause else 1) * pause_rate

    # === Overcontrol ===
    w1, w2, w3 = 0.4, 0.35, 0.25
    overcontrol = (
        pitch_stability_score * w1
        + decay_score * w2
        + pause_absence_score * w3
    )

    # === Fabrication_Score ===
    fabrication = context_contradiction * overcontrol
    fabrication_norm = min(1.0, fabrication / 2.0)

    # 基线置信度打折（冷启动保护）
    confidence = baseline.confidence
    fabrication_norm *= confidence

    evidence = {
        "pitch_var_current": round(pitch_var, 6),
        "pitch_var_baseline": round(pitch_var_baseline, 6),
        "pitch_stability_score": round(pitch_stability_score, 3),
        "decay_current_s": round(decay, 4),
        "decay_baseline_s": round(decay_baseline, 4),
        "decay_abnormality": round(decay_score, 3),
        "micro_pause_absent_risk": round(pause_absence_score, 3),
        "baseline_confidence": round(confidence, 3),
    }

    return round(fabrication_norm, 4), round(overcontrol, 4), evidence


# ========== 8. 联合判定 ==========

def assess_dialogue(segments_with_analysis, interaction):
    """对话全局质量评估"""
    arousals = [s.get("analysis", {}).get("arousal", 0) for s in segments_with_analysis]
    valences = [s.get("analysis", {}).get("valence", 0) for s in segments_with_analysis]
    conflicts = [s.get("analysis", {}).get("conflict_score", 0) for s in segments_with_analysis]
    fabrications = [s.get("analysis", {}).get("fabrication_score", 0) for s in segments_with_analysis]
    emotions = [s.get("analysis", {}).get("label", "?") for s in segments_with_analysis]

    耦合度 = interaction.energy_coupling() if interaction else 0

    return {
        "卷入度": round(np.mean(arousals) * (1 + abs(耦合度)), 3),
        "心理安全感": round(np.mean(valences) * (max(0, 1 - abs(耦合度))) if 耦合度 < -0.3 else 1, 3),
        "真诚度": round(1 - min(np.mean(conflicts + fabrications), 1), 3),
        "情绪分布": {e: emotions.count(e) for e in sorted(set(emotions))},
        "关系模式": interaction.relation_label(耦合度) if interaction else "单人",
    }
