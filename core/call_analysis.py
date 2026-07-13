"""
听心 · 电话客服分析引擎
每通摘要 + 全部统计 + 高频关键词 + 问题电话标红
"""
import re, os, json, jieba
from collections import Counter


# ============================================================
# 客服短语料库
# ============================================================
CUSTOMER_REFUSAL = [
    "不需要", "不用了", "不用", "不要", "没兴趣", "考虑一下",
    "考虑考虑", "我再看看", "再看看", "有需要再联系",
    "暂时不需要", "现在不需要", "再说吧", "以后再说",
    "太贵了", "买不起", "价格太高", "贵了", "没预算",
    "我没钱", "没钱", "没时间", "不方便", "没空", "忙",
    "挂了", "别打了", "不要再打了", "烦不烦", "烦死了",
]

CUSTOMER_AGREEMENT = [
    "好的", "可以", "行", "好", "行吧", "好吧", "嗯好",
    "可以啊", "没问题", "没问题啊", "好的好的", "好呀",
    "那行", "我考虑下", "到时候看",
]

COMPLAINT_KEYWORDS = [
    "投诉", "骗子", "骗人", "诈骗", "垃圾", "差评",
    "退款", "退货", "取消", "投诉你们", "客服", "态度差",
    "怎么办", "什么时候", "到底", "凭什么", "为什么",
]

AGENT_KEYWORDS = [
    "为您好", "方便", "优惠", "活动", "套餐", "名额",
    "截止", "限时", "送", "免费", "赠送", "增值",
    "了解一下", "耽误您", "打扰您", "您好这边",
]

REJECTION_CLUSTER = ["考虑一下", "太贵了", "不需要", "再看看", "没兴趣", "忙", "没时间"]


# ============================================================
# 单通电话分析
# ============================================================
class CallAnalyzer:
    """单通电话摘要 + 情绪 + 问题标记"""

    def analyze(self, segments: list[dict]) -> dict:
        """分析单通电话"""
        if not segments:
            return {"error": "无数据","flags":["无有效语音段"]}

        # 分离说话人
        customers = [s for s in segments if s.get("speaker") != "SPK1"]
        agents = [s for s in segments if s.get("speaker") == "SPK1"]

        if not customers and not agents:
            # 不区分说话人时，按顺序推测
            customers = segments[1::2] if len(segments) > 1 else segments[:len(segments)//2]
            agents = segments[0::2]

        # 客户情绪轨迹
        emotion_trail = []
        for s in (customers or segments):
            a = s.get("analysis", {})
            if a.get("label"):
                emotion_trail.append({
                    "time": round(s["start"], 1),
                    "emotion": a["label"],
                    "arousal": a.get("arousal", 0),
                    "valence": a.get("valence", 0),
                })

        # 客户摘要
        cust_texts = [s["text"] for s in (customers or segments)]
        cust_summary = self._summarize_texts(cust_texts, 4)
        agent_texts = [s["text"] for s in (agents or [])]
        agent_summary = self._summarize_texts(agent_texts, 3)

        # 拒绝检测
        refusals = self._detect_refusals(cust_texts)

        # 投诉检测
        complaints = self._detect_complaints(cust_texts)

        # 情绪判定
        neg_segments = len([s for s in emotion_trail if s["valence"] < -0.2])
        pos_segments = len([s for s in emotion_trail if s["valence"] > 0.1])
        total_emo = len(emotion_trail)
        final_sentiment = "不满意" if neg_segments > pos_segments and neg_segments > total_emo * 0.3 else "满意"

        # 问题标记
        flags = self._flag_problems(emotion_trail, refusals, complaints, segments)

        # 客户情绪变化总结
        if len(emotion_trail) >= 2:
            first_emo = emotion_trail[0]["emotion"]
            last_emo = emotion_trail[-1]["emotion"]
            if first_emo != last_emo:
                emotion_change = f"从{first_emo}转为{last_emo}"
            else:
                emotion_change = f"全程{first_emo}"
        else:
            emotion_change = "—"

        return {
            "customer_summary": cust_summary,
            "agent_summary": agent_summary,
            "emotion_trail": emotion_trail,
            "emotion_change": emotion_change,
            "final_sentiment": final_sentiment,
            "refusals": refusals,
            "complaints": complaints,
            "flags": flags,
            "is_problem": bool(flags),
            "total_segments": len(segments),
            "customer_segments": len(cust_texts),
            "agent_segments": len(agent_texts),
        }

    def _summarize_texts(self, texts: list[str], n: int = 4) -> str:
        """从文本中提取关键内容"""
        if not texts: return "—"
        # 分词统计高频实词
        all_words = []
        for t in texts:
            all_words.extend(jieba.lcut(t))
        stops = {"的","了","在","是","我","有","和","就","不","人","都","一","一个",
                 "上","也","很","到","说","要","去","你","会","着","没有","看","好",
                 "自己","这","他","她","它","那","里","吗","啊","呢","吧","嗯","哦",
                 "这个","那个","什么","怎么","因为","所以","但是","如果","然后","我们",
                 "您","请","一下","一个","行","中","吧","啊","个","就","能","被",
                 "把","让","向","从","对","与","同","跟","及","或","等","于"}
        words = [w for w in all_words if len(w)>1 and w not in stops]
        top_words = [w for w, _ in Counter(words).most_common(n)]
        if top_words:
            return "·".join(top_words)
        # 兜底
        full = " ".join(texts)
        return full[:80] + ("..." if len(full) > 80 else "")

    def _detect_refusals(self, texts: list[str]) -> list[dict]:
        """检测拒绝语句"""
        refusals = []
        for t in texts:
            for kw in CUSTOMER_REFUSAL:
                if kw in t:
                    refusals.append({"keyword": kw, "text": t[:60]})
                    break
        return refusals

    def _detect_complaints(self, texts: list[str]) -> list[dict]:
        """检测投诉意向"""
        complaints = []
        for t in texts:
            for kw in COMPLAINT_KEYWORDS:
                if kw in t:
                    complaints.append({"keyword": kw, "text": t[:60]})
                    break
        return complaints

    def _flag_problems(self, emotion_trail, refusals, complaints, segments) -> list[str]:
        """标记问题电话"""
        flags = []

        # 持续消极情绪
        neg_count = sum(1 for e in emotion_trail if e["valence"] < -0.2)
        if len(emotion_trail) > 0 and neg_count / max(len(emotion_trail), 1) > 0.5:
            flags.append("情绪异常: 消极情绪占比超50%")

        # 多次拒绝
        if len(refusals) >= 3:
            flags.append(f"客户多次拒绝({len(refusals)}次)")

        # 投诉
        if len(complaints) >= 2:
            flags.append(f"客户投诉意向({len(complaints)}次)")

        # 高冲突段
        high_conf = [s for s in segments if s.get("analysis", {}).get("conflict_score", 0) > 0.3]
        if len(high_conf) >= 2:
            flags.append(f"声文矛盾({len(high_conf)}段)")

        # 客户占时极短（不满不聊）
        if len(emotion_trail) <= 2 and len(segments) > 5:
            flags.append("客户对话极短，高挂断风险")

        return flags


# ============================================================
# 批量统计
# ============================================================
class BatchCallAnalyzer:
    """全部通话的统计分析"""

    def analyze(self, reports: list[dict], segments_by_file: dict = None) -> dict:
        """
        输入：report 列表 (JSON 报告字典)
        输出：综合统计
        """
        if not reports:
            return {"error": "无通话数据"}

        # 逐通分析
        call_results = []
        all_texts = []
        all_emotions = []
        all_flags = []
        all_conflicts = []

        for i, rpt in enumerate(reports):
            segments = rpt.get("timeline", [])
            if not segments:
                continue

            # MEITR 标签从 emotion 字段提取
            for s in segments:
                emo = s.get("emotion", {})
                if emo.get("label"):
                    all_emotions.append(emo["label"])
                all_texts.append(s.get("text", ""))

                conf = s.get("conflict", {})
                if conf.get("score", 0) > 0.3:
                    all_conflicts.append(s)

            # 单通分析
            ca = CallAnalyzer()

            # 重建 segments 结构（时间线是扁平字典）
            rebuilt = []
            for s in segments:
                rebuilt.append(s)

            result = ca.analyze(rebuilt)
            all_flags.extend(result.get("flags", []))
            call_results.append(result)

        # 情绪分布
        emotion_dist = dict(Counter(all_emotions))
        total_emo = max(len(all_emotions), 1)

        # 高频关键词
        stops = {"的","了","在","是","我","有","和","就","不","人","都","一","一个",
                 "上","也","很","到","说","要","去","你","会","着","没有","看","好",
                 "自己","这","他","她","它","那","里","吗","啊","呢","吧","嗯","哦",
                 "这个","那个","什么","怎么","因为","所以","但是","如果","然后","我们",
                 "您","请","一下","吧","啊","个","就","能","被","把","让","向","从",
                 "对","与","同","跟","及","或","等","于","嗯","哦","噢","呀","哈"}

        all_words = []
        for t in all_texts:
            all_words.extend(jieba.lcut(t))
        keywords = [w for w in all_words if len(w) > 1 and w not in stops]
        top_keywords = Counter(keywords).most_common(30)

        # 拒绝词聚类统计
        rejection_counts = {}
        for kw in REJECTION_CLUSTER:
            count = sum(1 for t in all_texts if kw in t)
            if count > 0:
                rejection_counts[kw] = count

        # 问题电话统计
        problem_count = sum(1 for r in call_results if r.get("is_problem"))
        problem_details = [r for r in call_results if r.get("is_problem")]

        return {
            "total_calls": len(reports),
            "total_segments": sum(r.get("total_segments",0) for r in call_results),
            "emotion_distribution": emotion_dist,
            "emotion_summary": self._emotion_summary(emotion_dist, total_emo),
            "top_keywords": [{"keyword": w, "count": c} for w, c in top_keywords[:20]],
            "rejection_keywords": rejection_counts,
            "conflict_segments": len(all_conflicts),
            "problem_calls": problem_count,
            "problem_details": [{
                "call_index": i+1,
                "flags": result.get("flags", []),
                "customer_summary": result.get("customer_summary",""),
                "final_sentiment": result.get("final_sentiment",""),
            } for i, result in enumerate(call_results) if result.get("is_problem")],
            "call_results": call_results,
            "all_emotions": all_emotions,
        }

    def _emotion_summary(self, dist: dict, total: int) -> str:
        """生成情绪一句话摘要"""
        pos = sum(v for k, v in dist.items() if k in ("中性", "平和", "正常"))
        neg = sum(v for k, v in dist.items() if k in ("冷蔑视", "敌意", "悲伤", "恐惧", "烦躁"))
        if total == 0: return "无情绪数据"
        pos_pct = round(pos / total * 100)
        neg_pct = round(neg / total * 100)
        if pos_pct > 80:
            return f"{pos_pct}%客户满意，{neg_pct}%客户不满"
        elif pos_pct > 60:
            return f"{pos_pct}%客户满意，{neg_pct}%客户不满，需关注"
        else:
            return f"{pos_pct}%客户满意，{neg_pct}%客户不满，投诉风险高"


# ============================================================
# 报告生成
# ============================================================
class CallReport:
    """呼叫分析报告格式化"""

    def to_markdown(self, single: dict = None, batch: dict = None, filename: str = "") -> str:
        """生成 Markdown 报告"""

        if single:
            return self._single_md(single, filename)
        if batch:
            return self._batch_md(batch)
        return ""

    def _single_md(self, data: dict, filename: str) -> str:
        lines = []
        fname = os.path.basename(filename) if filename else "通话"
        lines.append(f"# 通话分析报告")
        lines.append(f"## {fname}")
        lines.append("")

        # 问题标记
        if data.get("is_problem"):
            lines.append("> ⚠️ **问题电话**")
            for f in data.get("flags", []):
                lines.append(f"> - {f}")
            lines.append("")

        # 摘要
        lines.append("## 通话摘要")
        lines.append(f"- **客户关注**: {data.get('customer_summary','—')}")
        lines.append(f"- **客服内容**: {data.get('agent_summary','—')}")
        lines.append(f"- **客户情绪变化**: {data.get('emotion_change','—')}")
        lines.append(f"- **最终情绪**: {data.get('final_sentiment','—')}")
        lines.append("")

        # 拒绝
        if data.get("refusals"):
            lines.append("## 客户拒绝")
            for r in data["refusals"]:
                lines.append(f"- \"{r['keyword']}\" → {r['text'][:40]}...")
            lines.append("")

        # 投诉
        if data.get("complaints"):
            lines.append("## 投诉/质疑")
            for c in data["complaints"]:
                lines.append(f"- \"{c['keyword']}\" → {c['text'][:40]}...")
            lines.append("")

        # 情绪轨迹表
        if data.get("emotion_trail"):
            lines.append("## 情绪时间线")
            lines.append("| 时间 | 情绪 | 唤醒 | 效价 |")
            lines.append("|------|------|------|------|")
            for e in data["emotion_trail"]:
                lines.append(f"| {e['time']}s | {e['emotion']} | {e['arousal']:.2f} | {e['valence']:.2f} |")
            lines.append("")

        lines.append("---")
        lines.append("*报告由 听心·话务分析引擎 生成*")
        return "\n".join(lines)

    def _batch_md(self, data: dict) -> str:
        lines = []
        lines.append("# 话务批量分析报告")
        lines.append("")

        lines.append("## 概览")
        lines.append(f"- 通话总数: **{data.get('total_calls',0)}**")
        lines.append(f"- 总语音段数: {data.get('total_segments',0)}")
        lines.append(f"- 情绪摘要: {data.get('emotion_summary','')}")
        lines.append(f"- 问题电话: **{data.get('problem_calls',0)}** 通")
        lines.append(f"- 声文矛盾段: {data.get('conflict_segments',0)}")
        lines.append("")

        # 情绪分布
        ed = data.get("emotion_distribution", {})
        if ed:
            lines.append("## 情绪分布")
            for k, v in sorted(ed.items(), key=lambda x: -x[1]):
                pct = round(v / max(sum(ed.values()), 1) * 100)
                bar = "█" * (pct // 5)
                lines.append(f"- **{k}**: {v}次 ({pct}%) {bar}")
            lines.append("")

        # 高频关键词
        if data.get("top_keywords"):
            lines.append("## 高频关键词")
            lines.append("| 关键词 | 出现次数 |")
            lines.append("|--------|---------|")
            for kw in data["top_keywords"][:15]:
                lines.append(f"| {kw['keyword']} | {kw['count']} |")
            lines.append("")

        # 拒绝词
        rk = data.get("rejection_keywords", {})
        if rk:
            lines.append("## 拒绝词统计")
            for kw, cnt in sorted(rk.items(), key=lambda x: -x[1]):
                lines.append(f"- \"{kw}\": {cnt}次")
            lines.append("")

        # 问题电话详情
        if data.get("problem_details"):
            lines.append("## 🚨 问题电话明细")
            for pd in data["problem_details"]:
                flag_str = "; ".join(pd.get("flags", []))
                lines.append(f"### 第{pd['call_index']}通")
                lines.append(f"- 标记: {flag_str}")
                lines.append(f"- 客户关注: {pd.get('customer_summary','')}")
                lines.append(f"- 最终情绪: {pd.get('final_sentiment','')}")
                lines.append("")

        lines.append("---")
        lines.append("*报告由 听心·话务分析引擎 生成*")
        return "\n".join(lines)
