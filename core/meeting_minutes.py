"""
听心 · 会议纪要核心模块
话题切分 + 结论提取 + 待办事项
纯本地 NLP，无联网调用
"""
import re, jieba
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

# ============================================================
# 待办事项正则库
# ============================================================
ACTION_PATTERNS = [
    (re.compile(r"(我|我们?)(来|负责|安排|跟进|处理|做|整理|修改|联系|搞定|尝试)"), "承诺"),
    (re.compile(r"(交给|交给)我"), "承诺"),
    (re.compile(r"(会|要)(去|想办法|试着|尽量|安排|处理)"), "承诺"),
    (re.compile(r"(我|我们)答应"), "承诺"),
    (re.compile(r"(要不|那|可以)(我|我们)"), "提议"),
    (re.compile(r"(我|我们)(需要|必须|应该|得)"), "主动"),
    (re.compile(r"(我|我们)(可以|能)(先)"), "主动"),
    (re.compile(r"(你|您|他|她)(要|需要|负责|去)"), "指派"),
]

# 结论关键词
SUMMARY_MARKERS = [
    "所以", "总之", "结论", "决定", "最终", "总结", "整体",
    "我的意思是", "我建议", "建议", "我的看法",
    "关键", "核心", "主要问题", "首先要",
]

# 话题标记词
TOPIC_MARKERS = [
    "接下来", "第二个", "另一个", "说到", "关于", "然后",
    "我们来看", "讨论", "转到", "换一个",
]


# ============================================================
# 中文文本预处理
# ============================================================
def preprocess(text: str) -> str:
    """分词 + 过滤停用词"""
    stops = {"的","了","在","是","我","有","和","就","不","人","都","一","一个",
             "上","也","很","到","说","要","去","你","会","着","没有","看","好",
             "自己","这","他","她","它","那","里","吗","啊","呢","吧","嗯","哦",
             "这个","那个","什么","怎么","因为","所以","但是","如果","然后","我们"}
    words = jieba.lcut(text)
    return " ".join(w for w in words if len(w)>1 and w not in stops)


# ============================================================
# 话题切分器
# ============================================================
class TopicSegmenter:
    """基于 TF-IDF + 滑动窗口相似度的话题切分"""

    def __init__(self, threshold: float = 0.08, window: int = 2):
        self.threshold = threshold  # 相似度低于此值 → 话题切换
        self.window = window        # 滑动窗口（段数）

    def segment(self, segments: list[dict]) -> list[dict]:
        """输入：segment 列表（每段含 text）
           输出：每段加上 topic_id
        """
        texts = [s["text"] for s in segments]
        cleaned = [preprocess(t) for t in texts]

        # 建 TF-IDF
        vec = TfidfVectorizer(max_features=200, sublinear_tf=True)
        try:
            X = vec.fit_transform(cleaned).toarray()
        except ValueError:
            # 所有文本都一样或为空
            for s in segments:
                s["topic_id"] = 0
            return segments

        # 滑动窗口相似度
        topic_ids = [0]
        for i in range(1, len(X)):
            # 与前一段比较
            sim_adj = cosine_similarity(X[i:i+1], X[i-1:i])[0][0]
            # 与前 window 段平均比较
            start = max(0, i - self.window)
            sim_win = cosine_similarity(X[i:i+1], X[start:i].mean(axis=0, keepdims=True))[0][0] if i > start else 1.0
            sim = max(sim_adj, sim_win)

            if sim < self.threshold:
                topic_ids.append(topic_ids[-1] + 1)
            else:
                topic_ids.append(topic_ids[-1])

        for s, tid in zip(segments, topic_ids):
            s["topic_id"] = tid

        return segments

    def label_topics(self, segments: list[dict]) -> dict[int, str]:
        """为每个话题块生成标签"""
        if not segments or "topic_id" not in segments[0]:
            return {}

        topics = {}
        for s in segments:
            tid = s["topic_id"]
            if tid not in topics:
                topics[tid] = []
            topics[tid].append(s["text"])

        labels = {}
        vec = TfidfVectorizer(max_features=200, sublinear_tf=True, stop_words=["的","了","在","是","我"])
        for tid, texts in topics.items():
            combined = " ".join(texts)
            cleaned = preprocess(combined)
            if not cleaned.strip():
                labels[tid] = f"话题{tid+1}"
                continue

            # 用 TF-IDF 提取代表性词
            try:
                X = vec.fit_transform([cleaned])
                idxs = np.argsort(X.toarray()[0])[::-1][:3]
                terms = [vec.get_feature_names_out()[i] for i in idxs if X[0,i] > 0]
                label = " · ".join(terms) if terms else f"话题{tid+1}"
            except:
                label = f"话题{tid+1}"
            labels[tid] = label

        return labels


# ============================================================
# 关键结论提取器
# ============================================================
class ConclusionExtractor:
    """基于规则的句子重要性打分"""

    def __init__(self, top_n_per_topic: int = 3):
        self.top_n = top_n_per_topic

    def extract(self, segments: list[dict]) -> list[dict]:
        """提取关键结论"""
        conclusions = []
        # 按话题分组
        topics = {}
        for s in segments:
            tid = s.get("topic_id", 0)
            if tid not in topics:
                topics[tid] = []
            topics[tid].append(s)

        for tid, segs in topics.items():
            scored = []
            # 先按句拆分
            all_sents = []
            for i, s in enumerate(segs):
                sents = self._split_sentences(s["text"])
                for j, sent in enumerate(sents):
                    all_sents.append((s, i, j, sent))

            for s, si, sj, sent in all_sents:
                score = self._score_sentence(sent, si, len(segs), segs)
                if score > 0.3:
                    scored.append({
                        "text": sent,
                        "score": round(score, 2),
                        "topic_id": tid,
                        "speaker": s.get("speaker", "?"),
                        "time": s.get("start", 0),
                    })

            scored.sort(key=lambda x: x["score"], reverse=True)
            conclusions.extend(scored[:self.top_n])

        return sorted(conclusions, key=lambda x: (x["topic_id"], -x["score"]))

    def _split_sentences(self, text: str) -> list[str]:
        """简单中文分句"""
        text = re.sub(r"[，。！？；、\n]+", "|", text)
        return [s.strip() for s in text.split("|") if len(s.strip()) > 3]

    def _score_sentence(self, sent: str, idx: int, total: int, segs: list) -> float:
        """句子重要性打分"""
        score = 0.1  # 基础分

        # 含总结关键词
        for kw in SUMMARY_MARKERS:
            if kw in sent:
                score += 0.4
                break

        # 位置加分（前20%或后20%）
        pos = idx / max(total, 1)
        if pos < 0.2 or pos > 0.8:
            score += 0.2

        # 长度加分（15-60 字最佳）
        if 15 <= len(sent) <= 60:
            score += 0.15
        elif len(sent) > 60:
            score += 0.05

        # 含数字（时间、数量）
        if re.search(r"\d+", sent):
            score += 0.1

        # 含"是"类判断句式
        if re.search(r"是[^.。]*的", sent) or re.search(r"就是[^.。]*", sent):
            score += 0.15

        # 含回答标记（含"好"、"行"、"可以"）
        if re.search(r"\b(好|行|可以|对)\b", sent):
            score -= 0.05

        return min(score, 1.0)


# ============================================================
# 待办事项提取器
# ============================================================
class ActionItemExtractor:
    """基于正则的承诺/指派句式提取"""

    def extract(self, segments: list[dict]) -> list[dict]:
        items = []
        for s in segments:
            speaker = s.get("speaker", "?")
            text = s["text"]
            time = s.get("start", 0)

            matched = set()
        for pattern, atype in ACTION_PATTERNS:
            for m in pattern.finditer(text):
                start_pos = m.start()
                # 避免同一位置重复
                if start_pos in matched:
                    continue
                matched.add(start_pos)

                full = m.group(0)
                end_pos = m.end()
                action_phrase = text[end_pos:end_pos+30].strip()
                action_phrase = re.split(r"[，。！？；、\n]", action_phrase)[0]
                if len(action_phrase) > 20:
                    action_phrase = action_phrase[:20] + "..."
                # 去掉"那""的"等前缀
                full = re.sub(r"^(那|的|就)", "", full)
                desc = f"{full}{action_phrase}" if action_phrase else full

                items.append({
                    "speaker": speaker,
                    "action": desc,
                    "type": atype,
                    "time": round(time, 1),
                    "context": text[:80] + ("..." if len(text)>80 else ""),
                })
        return items


# ============================================================
# 综合报告生成器
# ============================================================
class MeetingReport:
    """会议纪要全流程"""

    def __init__(self):
        self.segmenter = TopicSegmenter()
        self.conclusioner = ConclusionExtractor()
        self.actioner = ActionItemExtractor()

    def generate(self, segments: list[dict], speakers: dict = None) -> dict:
        """生成完整会议纪要数据"""
        if not segments:
            return {"error": "无有效段落"}

        # 1. 话题切分
        segs = self.segmenter.segment(segments)
        topic_labels = self.segmenter.label_topics(segs)

        # 2. 话题流转时间线
        topic_timeline = []
        seen = set()
        for s in segs:
            tid = s["topic_id"]
            if tid not in seen:
                seen.add(tid)
                topic_timeline.append({
                    "id": tid,
                    "label": topic_labels.get(tid, f"话题{tid+1}"),
                    "start": round(s["start"], 1),
                })

        # 3. 说话人统计
        if speakers is None:
            spk_stats = {}
            for s in segs:
                spk = s.get("speaker", "?")
                if spk not in spk_stats:
                    spk_stats[spk] = {"sentences": 0, "time": 0.0}
                spk_stats[spk]["sentences"] += 1
                spk_stats[spk]["time"] += s["end"] - s["start"]
        else:
            spk_stats = speakers

        # 4. 结论
        conclusions = self.conclusioner.extract(segs)

        # 5. 待办
        actions = self.actioner.extract(segs)

        return {
            "topic_timeline": topic_timeline,
            "topic_labels": topic_labels,
            "speaker_stats": spk_stats,
            "conclusions": conclusions,
            "action_items": actions,
            "segments_with_topics": segs,
        }

    def to_markdown(self, data: dict, filename: str = "") -> str:
        """把会议纪要数据转成 Markdown"""
        lines = []
        title = os.path.splitext(os.path.basename(filename))[0] if filename else "会议纪要"
        lines.append(f"# {title}")
        lines.append("")

        # 基本信息
        segs = data.get("segments_with_topics", [])
        lines.append("## 基本信息")
        lines.append(f"- 总发言段数: {len(segs)}")
        lines.append(f"- 说话人数: {len(data.get('speaker_stats', {}))}")
        total_time = max(sum(v["time"] for v in data.get("speaker_stats", {}).values()), 1)
        lines.append(f"- 总时长: {int(total_time)}秒")
        lines.append("")

        # 说话人统计
        lines.append("## 发言统计")
        for spk, st in sorted(data.get("speaker_stats", {}).items(), key=lambda x: -x[1]["time"]):
            pct = st["time"] / total_time * 100
            lines.append(f"- **{spk}**: {st['sentences']}句, {pct:.0f}%时间")
        lines.append("")

        # 话题流转
        lines.append("## 话题流转")
        tl = data.get("topic_timeline", [])
        for t in tl:
            lines.append(f"- **{t['start']}s** → {t['label']}")
        lines.append("")

        # 逐段内容（含话题标签）
        lines.append("## 完整对话")
        for s in segs:
            tid = s.get("topic_id", 0)
            label = data.get("topic_labels", {}).get(tid, "")
            tag = f"[{label}]" if label else ""
            lines.append(f"**{s['start']:.0f}s {s.get('speaker','?')}** {tag}")
            lines.append(f"> {s['text']}")
            lines.append("")

        # 关键结论
        if data.get("conclusions"):
            lines.append("## 关键结论")
            for i, c in enumerate(data["conclusions"], 1):
                lines.append(f"{i}. **{c['speaker']}** ({c['time']:.0f}s): {c['text']}")
            lines.append("")

        # 待办事项
        if data.get("action_items"):
            lines.append("## 待办事项")
            for a in data["action_items"]:
                lines.append(f"- [{a['type']}] **{a['speaker']}**: {a['action']}")
            lines.append("")

        return "\n".join(lines)


# 避免循环引用
import os
