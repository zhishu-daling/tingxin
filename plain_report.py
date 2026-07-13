# -*- coding: utf-8 -*-
"""
听心 · 普通人版报告生成器

用法（CLI）：
    python plain_report.py <json文件路径>

从Python调用：
    from plain_report import generate_plain_report
    report = generate_plain_report("path/to/result.json")
    print(report)

说明：
    读取听心输出的JSON分析文件，输出原有技术报告（格式参见 _report_summary.py），
    末尾追加一段「普通人版总结」，将技术指标翻译成白话。
"""

import json
import sys
import os
from collections import Counter

# 修复 Windows 控制台 GBK 编码无法打印 UTF-8 字符的问题
if sys.stdout.encoding and sys.stdout.encoding.upper() in ('GBK', 'GB2312', 'CP936'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        # Python < 3.7 不支持 reconfigure，用 buffer 绕开
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')


# ============================================================
# 第一部分：技术报告生成（格式与 _report_summary.py 保持一致）
# 来源映射：_report_summary.py → 本文件的 generate_technical_report()
# ============================================================

def generate_technical_report(data):
    """生成与 _report_summary.py 格式一致的技术报告。"""
    segments = data.get('segments', [])
    duration = data.get('duration', 0)
    num_speakers = data.get('num_speakers', 0)
    total_time = data.get('total_time', 0)
    filename = data.get('filename', '未知文件')

    lines = []
    lines.append('=' * 60)
    lines.append('听心 · 万物系统法报告')
    lines.append('=' * 60)
    lines.append(f'文件: {filename}')
    lines.append(f'时长: {duration:.0f}s ({duration/60:.0f}分 {duration%60:.0f}s)')
    lines.append(f'段数: {len(segments)}')
    lines.append(f'说话人数: {num_speakers}')
    lines.append(f'分析耗时: {total_time:.0f}s')
    lines.append('')

    # 按说话人统计
    speakers = sorted(set(s['speaker'] for s in segments))
    for spk in speakers:
        segs = [s for s in segments if s['speaker'] == spk]
        if not segs:
            continue
        dur = sum(s['end'] - s['start'] for s in segs)
        pct = dur / duration * 100 if duration > 0 else 0
        lines.append(f'── {spk} ({len(segs)}句, {dur:.0f}s/{duration:.0f}s = {pct:.0f}%)')

        avg_a = sum(s['analysis']['arousal'] for s in segs) / len(segs)
        avg_v = sum(s['analysis']['valence'] for s in segs) / len(segs)
        avg_d = sum(s['analysis']['dominance'] for s in segs) / len(segs)
        lines.append(f'   五维均值: 唤醒{avg_a:.2f} · 效价{avg_v:.2f} · 控制{avg_d:.2f}')

        labels = Counter(s['analysis']['label'] for s in segs)
        lines.append(f'   情绪: {dict(labels.most_common())}')

        defs = Counter(s['analysis'].get('defense_mechanism') or '无' for s in segs)
        lines.append(f'   防御: {dict(defs.most_common())}')

        fab_scores = [s['analysis'].get('fabrication_score', 0) for s in segs]
        lines.append(f'   伪装分数: 平均{sum(fab_scores)/len(fab_scores):.3f} · 最高{max(fab_scores):.3f}')

        ovc = [s['analysis'].get('overcontrol', 0) for s in segs]
        lines.append(f'   过度控制: 平均{sum(ovc)/len(ovc):.3f} · 最高{max(ovc):.3f}')

    lines.append('')

    # 高冲突段 (top 5)
    lines.append('── 最高冲突段 (top 5)')
    conflicts = sorted(segments, key=lambda s: -s['analysis'].get('conflict_score', 0))[:5]
    for s in conflicts:
        lines.append(f'  {s["speaker"]} [{s["start"]:.0f}s-{s["end"]:.0f}s] '
                      f'冲突{s["analysis"].get("conflict_score", 0):.2f} · '
                      f'{s["analysis"].get("label", "?")} · '
                      f'{s["analysis"].get("defense_mechanism") or "-"}')
        lines.append(f'    "{s["text"][:80]}"')
    lines.append('')

    # 最高伪装段 (top 5)
    lines.append('── 最高伪装段 (top 5)')
    fabs = sorted(segments, key=lambda s: -s['analysis'].get('fabrication_score', 0))[:5]
    for s in fabs:
        lines.append(f'  {s["speaker"]} [{s["start"]:.0f}s-{s["end"]:.0f}s] '
                      f'伪装{s["analysis"].get("fabrication_score", 0):.3f} · '
                      f'过度控制{s["analysis"].get("overcontrol", 0):.3f}')
        lines.append(f'    "{s["text"][:80]}"')
    lines.append('')

    # 全部对话转录
    lines.append('── 全部对话转录')
    for s in segments:
        ts = f'{s["start"]:6.0f}s - {s["end"]:5.0f}s'
        label = s['analysis'].get('label', '?')
        defn = s['analysis'].get('defense_mechanism') or '-'
        lines.append(f'  {s["speaker"]} [{ts}] {label:4s} {defn:4s} "{s["text"]}"')

    return '\n'.join(lines)


# ============================================================
# 第二部分：普通人版总结生成（规则模板，无AI）
# ============================================================

def _translate_arousal(val):
    """唤醒度翻译"""
    if val > 0.5:
        return "情绪投入度高", "声音明显激动/紧张"
    elif val >= 0.2:
        return "情绪平稳，有一定投入", "有一定情绪波动"
    else:
        return "情绪平淡", "声音缺乏能量"


def _translate_valence(val):
    """效价翻译"""
    if val > 0.3:
        return "情绪积极", "听起来开心/放松"
    elif val >= -0.3:
        return "情绪中性", "无明显倾向"
    else:
        return "情绪偏负面", "听起来低落/压抑"


def _translate_dominance(val):
    """控制度翻译"""
    if val > 0.4:
        return "说话有底气", "自信/掌控感强"
    elif val >= 0.2:
        return "基本自如", "处于不卑不亢状态"
    else:
        return "说话底气不足", "处于被动/顺从位置"


def _translate_conflict(val):
    """冲突分数翻译"""
    if val > 0.4:
        return "嘴上说的和真实情绪明显不一致"
    elif val >= 0.25:
        return "表达内容和声音信号略有出入"
    else:
        return "言行一致"


def _translate_defense(mech):
    """防御机制翻译"""
    mapping = {
        "压抑": "在努力控制情绪，不让真实感受流露",
        "否认": "嘴上说没事但声音暴露了焦虑",
        "合理化": "在用逻辑解释掩盖真实感受",
        "理智化": "用分析/客观的角度回避情感表达",
        "投射": "把自己的问题说成是对方的",
        "回避": "不想继续这个话题",
        "顺从": "被动接受，但内心未必认同",
        "幽默化": "用开玩笑来缓解紧张",
        "反向形成": "用相反的情绪掩盖真实感受",
        "无": None,
        None: None,
    }
    return mapping.get(mech)


def _translate_fabrication(val):
    """伪装分数翻译"""
    if val > 0.3:
        return "说话修饰痕迹重，可能在刻意表现"
    elif val >= 0.15:
        return "说话有一定修饰"
    else:
        return "说话自然，无明显修饰"


def _translate_alert(alert_val):
    """爆发预警翻译"""
    if alert_val is None:
        return None
    text = str(alert_val).strip()
    if "爆发" in text or "高危" in text:
        return "情绪压力积累快到达阈值，随时可能爆发"
    elif "冷暴力" in text:
        return "情绪压力高但控制力也强，可能转为冷暴力"
    else:
        return f"预警状态: {alert_val}"


def _describe_overall_emotion(segments):
    """判断整体情绪倾向"""
    valence_vals = [s['analysis'].get('valence', 0) for s in segments]
    arousal_vals = [s['analysis'].get('arousal', 0) for s in segments]
    avg_v = sum(valence_vals) / len(valence_vals) if valence_vals else 0
    avg_a = sum(arousal_vals) / len(arousal_vals) if arousal_vals else 0

    if avg_v > 0.2 and avg_a < 0.3:
        return "整体偏积极，气氛轻松"
    elif avg_v > 0.2 and avg_a >= 0.3:
        return "积极中带激动"
    elif avg_v < -0.2 and avg_a > 0.4:
        return "情绪紧张，偏负面状态"
    elif avg_v < -0.2 and avg_a < 0.3:
        return "整体低落、压抑"
    elif avg_v < -0.2:
        return "情绪偏负面"
    elif avg_a > 0.4:
        return "整体激动，情绪投入度高"
    else:
        return "情绪中性，无明显起伏"


def _dominant_defense_description(segs):
    """最常用的防御机制及其说明"""
    def_mechs = [s['analysis'].get('defense_mechanism') for s in segs if s['analysis'].get('defense_mechanism')]
    if not def_mechs:
        return None, None
    most_common = Counter(def_mechs).most_common(1)[0]
    mech, count = most_common
    desc = _translate_defense(mech)
    return mech, desc


def _has_notable_high_conflict(segments, threshold=0.4):
    """是否有明显的高冲突段"""
    return [s for s in segments if s['analysis'].get('conflict_score', 0) > threshold]


def _has_notable_fabrication(segments, threshold=0.3):
    """是否有明显的伪装段"""
    return [s for s in segments if s['analysis'].get('fabrication_score', 0) > threshold]


def _get_top_alert_segments(segments):
    """获取有预警的段"""
    return [s for s in segments if s['analysis'].get('prediction', {}).get('alert') is not None]


def _speaker_summary(segs):
    """生成单个说话人的白话总结信息"""
    if not segs:
        return None

    avg_a = sum(s['analysis'].get('arousal', 0) for s in segs) / len(segs)
    avg_v = sum(s['analysis'].get('valence', 0) for s in segs) / len(segs)
    avg_d = sum(s['analysis'].get('dominance', 0) for s in segs) / len(segs)
    avg_fab = sum(s['analysis'].get('fabrication_score', 0) for s in segs) / len(segs)
    avg_ovc = sum(s['analysis'].get('overcontrol', 0) for s in segs) / len(segs)

    # 情绪描述
    a_summary, a_detail = _translate_arousal(avg_a)
    v_summary, v_detail = _translate_valence(avg_v)
    d_summary, d_detail = _translate_dominance(avg_d)

    # 一致性
    high_conflict_segs = [s for s in segs if s['analysis'].get('conflict_score', 0) > 0.25]
    if high_conflict_segs:
        consistency = "不太一致"
        conflict_detail = f"，有{len(high_conflict_segs)}处明显矛盾"
    else:
        consistency = "基本一致"
        conflict_detail = ""

    # 防御机制
    mech, mech_desc = _dominant_defense_description(segs)
    if mech:
        psych_state = f"使用了「{mech}」防御机制，{mech_desc}"
    else:
        psych_state = "无明显防御机制，表达较为直接"

    # 压力水平
    strain_vals = [s['analysis'].get('prediction', {}).get('cum_strain', 0) for s in segs]
    max_strain = max(strain_vals) if strain_vals else 0
    alerts = [s for s in segs if s['analysis'].get('prediction', {}).get('alert') is not None]
    if alerts:
        pressure = "接近爆发阈值"
    elif max_strain > 0.7:
        pressure = "压力有累积，值得关注"
    elif max_strain > 0.4:
        pressure = "有一定压力累积"
    else:
        pressure = "相对稳定"

    # 说话风格
    fab_desc = _translate_fabrication(avg_fab)
    if avg_ovc > 0.3:
        style = f"{fab_desc}，且说话有过度控制痕迹"
    else:
        style = fab_desc

    return {
        'emotion': f"TA整体{v_summary}，说话时{a_detail}；{d_summary}（{d_detail}）",
        'consistency': f"TA说的话和声音传达的情绪{consistency}{conflict_detail}",
        'psych_state': psych_state,
        'pressure': pressure,
        'style': style,
    }


def _interaction_summary(segments):
    """双人互动总结"""
    speakers = sorted(set(s['speaker'] for s in segments))
    if len(speakers) < 2:
        return None

    spk_stats = {}
    for spk in speakers:
        segs = [s for s in segments if s['speaker'] == spk]
        dur = sum(s['end'] - s['start'] for s in segs)
        avg_d = sum(s['analysis'].get('dominance', 0) for s in segs) / len(segs)
        avg_v = sum(s['analysis'].get('valence', 0) for s in segs) / len(segs)
        avg_a = sum(s['analysis'].get('arousal', 0) for s in segs) / len(segs)
        spk_stats[spk] = {'duration': dur, 'avg_d': avg_d, 'avg_v': avg_v, 'avg_a': avg_a}

    # 谁主导
    sorted_by_dur = sorted(spk_stats.items(), key=lambda x: -x[1]['duration'])
    leader, follower = sorted_by_dur[0], sorted_by_dur[1]
    leader_spk = leader[0]
    follower_spk = follower[0]

    if leader[1]['avg_d'] > follower[1]['avg_d']:
        rhythm = f"{leader_spk}主导对话，{follower_spk}跟随"
    else:
        rhythm = f"{leader_spk}说话时间更长，但{follower_spk}在情绪上更有主导性"

    # 关系模式
    v_diff = leader[1]['avg_v'] - follower[1]['avg_v']
    a_diff = abs(leader[1]['avg_a'] - follower[1]['avg_a'])

    if v_diff > 0.3 and a_diff < 0.3:
        pattern = "一方积极一方中性，可能存在沟通温差"
    elif v_diff < -0.3 and a_diff > 0.3:
        pattern = "一方负面激动一方平静，情绪对抗明显"
    elif v_diff > -0.2 and a_diff < 0.2:
        pattern = "双方情绪同步，关系协调"
    elif follower[1]['avg_d'] < 0.2:
        pattern = f"{follower_spk}处于被动/顺从位置，{leader_spk}主导关系"
    else:
        pattern = "双方互动均衡，无明显对抗"

    return rhythm, pattern


def generate_plain_summary(data):
    """生成普通人版总结。"""
    segments = data.get('segments', [])
    duration = data.get('duration', 0)
    num_speakers = data.get('num_speakers', 0)
    filename = data.get('filename', '未知文件')

    if not segments:
        return "\n（无对话数据可总结）"

    minutes = int(duration // 60)
    seconds = int(duration % 60)

    lines = []
    lines.append('')
    lines.append('═' * 55)
    lines.append('📋 普通人版总结')
    lines.append('═' * 55)
    lines.append('')

    # 整体情况
    overall_emotion = _describe_overall_emotion(segments)
    lines.append('📌 整体情况')
    lines.append(f'一句话概括：这次对话总长{minutes}分{seconds}秒，有{num_speakers}个人参与。{overall_emotion}。')
    lines.append('')

    # 每人状态
    speakers = sorted(set(s['speaker'] for s in segments))
    for spk in speakers:
        segs = [s for s in segments if s['speaker'] == spk]
        if not segs:
            continue
        summary = _speaker_summary(segs)
        if summary is None:
            continue
        lines.append(f'🗣️ {spk} 的状态')
        lines.append(f'- 情绪：{summary["emotion"]}')
        lines.append(f'- 一致性：{summary["consistency"]}')
        lines.append(f'- 心理状态：{summary["psych_state"]}')
        lines.append(f'- 压力水平：{summary["pressure"]}')
        lines.append(f'- 说话风格：{summary["style"]}')
        lines.append('')

    # 双方互动（2人以上）
    if num_speakers >= 2:
        interaction = _interaction_summary(segments)
        if interaction:
            rhythm, pattern = interaction
            lines.append('🔄 双方互动')
            lines.append(f'- 对话节奏：{rhythm}')
            lines.append(f'- 关系模式：{pattern}')
            lines.append('')

    # 重点关注
    lines.append('⚠️ 重点关注（如有）')
    found_issue = False

    # 1) 高冲突段
    high_conf = _has_notable_high_conflict(segments)
    if high_conf:
        found_issue = True
        for s in high_conf[:3]:  # 最多显示3个
            start, end = int(s['start']), int(s['end'])
            text = s['text'][:60]
            conflict_desc = _translate_conflict(s['analysis'].get('conflict_score', 0))
            lines.append(f'- 第{start}s-{end}s段（{s["speaker"]}）：')
            lines.append(f'  "{text}"')
            lines.append(f'  这里{conflict_desc}')

    # 2) 伪装段
    high_fab = _has_notable_fabrication(segments)
    if high_fab:
        found_issue = True
        for s in high_fab[:2]:
            start, end = int(s['start']), int(s['end'])
            text = s['text'][:60]
            fab_desc = _translate_fabrication(s['analysis'].get('fabrication_score', 0))
            lines.append(f'- 第{start}s-{end}s段（{s["speaker"]}）：')
            lines.append(f'  "{text}"')
            lines.append(f'  这里{fab_desc}')

    # 3) 预警
    alert_segs = _get_top_alert_segments(segments)
    if alert_segs:
        found_issue = True
        for s in alert_segs[:2]:
            start, end = int(s['start']), int(s['end'])
            text = s['text'][:60]
            alert_val = s['analysis'].get('prediction', {}).get('alert')
            alert_desc = _translate_alert(alert_val)
            if alert_desc:
                lines.append(f'- 第{start}s-{end}s段（{s["speaker"]}）：')
                lines.append(f'  "{text}"')
                lines.append(f'  ⛔ {alert_desc}')

    if not found_issue:
        lines.append('- 无明显异常需要关注')
    lines.append('')

    # 一句话总结
    # 综合判断
    top_spk = speakers[0]
    spk_segs = [s for s in segments if s['speaker'] == top_spk]
    avg_d = sum(s['analysis'].get('dominance', 0) for s in spk_segs) / len(spk_segs) if spk_segs else 0
    avg_v = sum(s['analysis'].get('valence', 0) for s in segments) / len(segments)

    if num_speakers >= 2:
        spk2 = speakers[1]
        spk2_segs = [s for s in segments if s['speaker'] == spk2]
        avg_v2 = sum(s['analysis'].get('valence', 0) for s in spk2_segs) / len(spk2_segs) if spk2_segs else 0
        v_gap = abs(avg_v - avg_v2)

        if v_gap > 0.3 and (avg_v < 0 or avg_v2 < 0):
            conclusion = f"双方情绪存在明显差异，沟通中存在未被消解的情绪张力"
        elif high_conf:
            conclusion = f"对话中存在表里不一的时刻，有一方在隐藏真实感受"
        elif avg_v < -0.1:
            conclusion = f"整体气氛偏沉重，对话中潜藏着不满或无力感"
        else:
            conclusion = f"双方进行了有来有回的交流，整体气氛尚可"
    else:
        if avg_v < -0.1:
            conclusion = f"整体情绪偏负面，说话人内心有未表达的情绪"
        else:
            conclusion = f"单人表达，情绪状态平稳"

    lines.append('📊 一句话总结')
    lines.append(f'{conclusion}')
    lines.append('')

    return '\n'.join(lines)


# ============================================================
# 主入口
# ============================================================

def generate_plain_report(json_path):
    """
    读取听心输出的JSON分析文件，返回原有技术报告 + 普通人版总结。

    参数：
        json_path (str): JSON文件路径

    返回：
        str: 完整报告文本

    抛出：
        FileNotFoundError: 文件不存在
        json.JSONDecodeError: JSON格式错误
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"文件不存在: {json_path}")

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    technical = generate_technical_report(data)
    plain = generate_plain_summary(data)

    return technical + plain


def main():
    if len(sys.argv) < 2:
        print("用法: python plain_report.py <json文件路径>", file=sys.stderr)
        sys.exit(1)

    json_path = sys.argv[1]

    if not os.path.exists(json_path):
        print(f"❌ 错误：文件不存在 → {json_path}", file=sys.stderr)
        sys.exit(1)

    try:
        report = generate_plain_report(json_path)
        print(report)
    except json.JSONDecodeError as e:
        print(f"❌ 错误：JSON格式不正确 → {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"❌ 错误：{e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
