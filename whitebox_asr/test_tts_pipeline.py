# -*- coding: utf-8 -*-
"""
白盒ASR引擎测试脚本
1. 用Windows SAPI TTS合成~3000字中文语音
2. 跑白盒ASR引擎
3. 对比输入文本 vs 识别结果
"""
import sys, os, json, time, wave, struct, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from numpy import bool_ as np_bool

# 导入白盒ASR（因为当前目录就是whitebox_asr包）
from engine import WhiteboxASR

# ── 测试文本（~3000字，覆盖各声韵调）──
TEST_TEXT = """
白盒语音识别引擎是一个不需要神经网络、不需要GPU的纯公式驱动系统。
它基于听心声学基元提取层，将音频分解为MFCC、基频、共振峰、光谱质心等特征。
然后通过白盒公式分类声母韵母和声调，最后用输入法词库转换成汉字。

声母分类的核心公式包括爆破音检测、擦音检测、塞擦音检测、鼻音检测和边音检测。
爆破音通过能量包络的阶跃比来识别，擦音通过高频能量占比来判断。
韵母分类基于共振峰F1和F2映射到元音三角形，复韵母通过共振峰轨迹匹配。
声调分类基于F0曲线的统计特征，通过决策树区分四声和轻声。

这个系统的最大优势是完全白盒可追溯，每个识别结果都能回溯到具体的声学特征。
用户可以手工调整阈值参数来优化识别精度，不需要训练数据。
全部计算在CPU上实时完成，延迟在毫秒级别。

中国的汉字文化博大精深，从甲骨文到现代简体字，经历了三千多年的演变。
普通话有大约四百个音节，加上四个声调，总共约一千六百个发音单元。
这些发音单元对应着五千多个常用汉字，形成了汉语的完整体系。

语音识别技术的发展经历了从高斯混合模型到深度神经网络的转变。
但深度学习模型需要大量标注数据和GPU算力，而且推理过程不可解释。
白盒方案反其道而行之，用可解释的声学公式替代黑盒神经网络。
虽然精度可能不如深度学习模型，但在可控性和可追溯性上有独特优势。

未来的发展方向包括优化共振峰提取精度、增加音节切分模块、完善上下文消歧。
还可以将领域知识注入到词库中，比如历史人物名字、专业术语等。
这些优化都不需要重新训练模型，只需要调整公式阈值或更新词库即可。

测试用例到此结束，感谢您的关注。
白盒语音识别系统将声学特征与汉字文化相结合，实现了从声音到文字的高效转换。
这个系统不需要大量的训练数据，也不需要昂贵的硬件设备。
它完全基于声学原理和语言学知识，每个判断都有清晰的公式依据。
从声母的爆破音检测到韵母的共振峰匹配，从声调的F0曲线分析到汉字的维特比解码。
每一步都是可解释的、可调整的、可优化的。
这正是白盒方案的最大价值所在。
"""

# ═══════════════════════════════════════════════════
# 1. Windows SAPI TTS 合成
# ═══════════════════════════════════════════════════

def synthesize_with_sapi(text, output_wav, rate=-1, volume=100):
    """用Windows SAPI合成中文语音"""
    import win32com.client
    
    # 统计汉字数
    hanzi_count = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    print(f"[TTS] 合成中... (总字数:{len(text)} 汉字数:{hanzi_count})")
    t0 = time.time()
    
    speaker = win32com.client.Dispatch("SAPI.SpVoice")
    
    # 选择中文语音
    voices = speaker.GetVoices()
    chosen_voice = None
    for v in voices:
        desc = v.GetDescription().lower()
        if "hui" in desc or "xiaoxiao" in desc or "zh-cn" in desc:
            chosen_voice = v
            break
    if chosen_voice is None and voices.Count > 0:
        chosen_voice = voices[0]
    
    if chosen_voice:
        speaker.Voice = chosen_voice
        print(f"  语音: {chosen_voice.GetDescription()}")
    
    speaker.Rate = rate
    speaker.Volume = volume
    
    stream = win32com.client.Dispatch("SAPI.SpFileStream")
    stream.Open(output_wav, 3)
    speaker.AudioOutputStream = stream
    speaker.Speak(text)
    stream.Close()
    
    elapsed = time.time() - t0
    file_size = os.path.getsize(output_wav)
    duration_sec = file_size / 44100 / 2  # 16bit mono at 44100Hz
    print(f"  WAV: {output_wav} ({file_size/1024:.0f}KB, ~{duration_sec:.0f}s, {elapsed:.1f}s合成)")
    return output_wav


# ═══════════════════════════════════════════════════
# 2. 运行白盒ASR引擎
# ═══════════════════════════════════════════════════

def run_asr(audio_path):
    """运行白盒ASR引擎"""
    print(f"\n[ASR] 识别中... {audio_path}")
    t0 = time.time()
    
    asr = WhiteboxASR(debug=True)
    result = asr.transcribe(audio_path, debug=True)
    
    elapsed = time.time() - t0
    
    print(f"  耗时: {elapsed:.2f}s")
    print(f"  各阶段耗时: {result.get('timing', {})}")
    print(f"  识别文本({len(result['text'])}字): {result['text'][:200]}...")
    print(f"  拼音({len(result['pinyin_numbered'])}个): {' '.join(result['pinyin_numbered'][:30])}...")
    
    return result


# ═══════════════════════════════════════════════════
# 3. 对比评估
# ═══════════════════════════════════════════════════

def evaluate(input_text, result):
    """简单评估：按字准确率"""
    import re
    
    def clean(s):
        return re.sub(r'[^\u4e00-\u9fff]', '', s)
    
    ref = clean(input_text)
    hyp = clean(result.get("text", ""))
    
    min_len = min(len(ref), len(hyp))
    correct = sum(1 for i in range(min_len) if ref[i] == hyp[i])
    accuracy = correct / len(ref) if len(ref) > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"评估结果:")
    print(f"  参考汉字数: {len(ref)}")
    print(f"  识别汉字数: {len(hyp)}")
    print(f"  正确匹配: {correct}")
    print(f"  字准确率: {accuracy*100:.2f}%")
    
    # 显示前30处差异
    mismatch_positions = []
    for i in range(min_len):
        if ref[i] != hyp[i]:
            mismatch_positions.append(i)
            if len(mismatch_positions) <= 30:
                context = ref[max(0,i-2):i+3]
                print(f"  [{i}] 参考='{ref[i]}' 识别='{hyp[i]}' 上下文='{context}'")
    
    print(f"\n  总差异: {len(mismatch_positions)}/{min_len}")
    
    return {
        "ref_len": len(ref),
        "hyp_len": len(hyp),
        "correct": correct,
        "accuracy": accuracy,
        "mismatches": mismatch_positions[:100],
    }


# ═══════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

if __name__ == "__main__":
    output_dir = os.path.dirname(os.path.abspath(__file__))
    wav_path = os.path.join(output_dir, "test_3000chars.wav")
    
    print("=" * 60)
    print("白盒声学拼音识别引擎 — 测试")
    print("=" * 60)
    
    # 1. TTS合成
    synthesize_with_sapi(TEST_TEXT, wav_path, rate=-1, volume=100)
    
    # 2. 跑ASR
    result = run_asr(wav_path)
    
    # 3. 评估
    eval_result = evaluate(TEST_TEXT, result)
    
    # 4. 清理不可序列化的字段，然后保存报告
    import copy
    def strip_audio(d):
        if isinstance(d, dict):
            return {k: strip_audio(v) for k, v in d.items() if k != 'audio'}
        if isinstance(d, list):
            return [strip_audio(v) for v in d]
        return d
    
    result_clean = strip_audio(result)
    report = {
        "input": {"text": TEST_TEXT, "chars": len(TEST_TEXT)},
        "asr_result": result_clean,
        "evaluation": eval_result,
        "timing": result.get("timing", {}),
        "pinyin_count": len(result.get("pinyin_numbered", [])),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    report_path = os.path.join(output_dir, "test_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
    print(f"\n报告已保存: {report_path}")
    print("=" * 60)
