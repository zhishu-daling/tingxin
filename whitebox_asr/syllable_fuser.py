"""
L3.4 音节融合

声母 + 韵母 + 声调 → 带调拼音串
对零声母音节做 y/w 转换等正字法处理。
"""

import re
from typing import Dict, Optional


# ========== 拼音正字法规则 ==========

# 零声母时 i → y / u → w / ü → yu 的规则
_I_TO_Y = {
    "i": "yi", "ia": "ya", "ie": "ye", "iao": "yao",
    "iu": "you", "ian": "yan", "in": "yin", "iang": "yang",
    "ing": "ying", "iong": "yong",
}

_U_TO_W = {
    "u": "wu", "ua": "wa", "uo": "wo", "uai": "wai",
    "ui": "wei", "uan": "wan", "un": "wen", "uang": "wang",
    "ueng": "weng",
}

_U_DOT_REMOVE = {
    "ü": "yu", "üe": "yue", "üan": "yuan", "ün": "yun",
}

# 声母韵母组合合法性表（简化版）
# 某些韵母不能跟某些声母搭配
_VALID_COMBOS = {
    # 唇音(b,p,m,f) 不能跟 u 以外的合口呼搭配
    "b": {"i", "ia", "ie", "iao", "iu", "ian", "in", "ing",
          "a", "o", "e", "ai", "ei", "ao", "ou",
          "an", "en", "ang", "eng", "ong",
          "u"},
    "p": {"i", "ia", "ie", "iao", "iu", "ian", "in", "ing",
          "a", "o", "e", "ai", "ei", "ao", "ou",
          "an", "en", "ang", "eng", "ong",
          "u"},
    "m": {"i", "ie", "iao", "iu", "ian", "in", "ing",
          "a", "o", "e", "ai", "ei", "ao", "ou",
          "an", "en", "ang", "eng", "ong",
          "u"},
    "f": {"a", "o", "u", "ei", "ou", "an", "en", "ang", "eng", "ong"},
    
    # d,t 不能跟 i 或 ü 组
    "d": {"i", "ia", "ie", "iao", "iu", "ian", "in", "ing",
          "a", "o", "e", "ai", "ei", "ao", "ou",
          "an", "en", "ang", "eng", "ong",
          "u", "ua", "uo", "uai", "ui", "uan", "un", "uang",
          "ü", "üe", "üan"},
    "t": {"i", "ia", "ie", "iao", "iu", "ian", "in", "ing",
          "a", "o", "e", "ai", "ei", "ao", "ou",
          "an", "en", "ang", "eng", "ong",
          "u", "ua", "uo", "uai", "ui", "uan", "un", "uang",
          "ü", "üe", "üan"},
    
    # n,l 可以跟几乎所有韵母
    "n": "__all__",
    "l": "__all__",
    
    # g,k,h 不能跟 i 或 ü 组（腭化）
    "g": {"a", "o", "e", "ai", "ei", "ao", "ou",
          "an", "en", "ang", "eng", "ong",
          "u", "ua", "uo", "uai", "ui", "uan", "un", "uang", "ueng"},
    "k": {"a", "o", "e", "ai", "ei", "ao", "ou",
          "an", "en", "ang", "eng", "ong",
          "u", "ua", "uo", "uai", "ui", "uan", "un", "uang", "ueng"},
    "h": {"a", "o", "e", "ai", "ei", "ao", "ou",
          "an", "en", "ang", "eng", "ong",
          "u", "ua", "uo", "uai", "ui", "uan", "un", "uang", "ueng"},
    
    # j,q,x 只能跟 i 或 ü 组（腭化）
    "j": {"i", "ia", "ie", "iao", "iu", "ian", "in", "iang", "ing", "iong",
          "ü", "üe", "üan", "ün"},
    "q": {"i", "ia", "ie", "iao", "iu", "ian", "in", "iang", "ing", "iong",
          "ü", "üe", "üan", "ün"},
    "x": {"i", "ia", "ie", "iao", "iu", "ian", "in", "iang", "ing", "iong",
          "ü", "üe", "üan", "ün"},
    
    # zh,ch,sh,r 不能跟 i 或 ü 组
    "zh": "__all__",
    "ch": "__all__",
    "sh": "__all__",
    "r":  "__all__",
    
    # z,c,s
    "z": "__all__",
    "c": "__all__",
    "s": "__all__",
}


_TONE_MARKS = {
    "a": {1: "ā", 2: "á", 3: "ǎ", 4: "à", 5: "a"},
    "o": {1: "ō", 2: "ó", 3: "ǒ", 4: "ò", 5: "o"},
    "e": {1: "ē", 2: "é", 3: "ě", 4: "è", 5: "e"},
    "i": {1: "ī", 2: "í", 3: "ǐ", 4: "ì", 5: "i"},
    "u": {1: "ū", 2: "ú", 3: "ǔ", 4: "ù", 5: "u"},
    "ü": {1: "ǖ", 2: "ǘ", 3: "ǚ", 4: "ǜ", 5: "ü"},
}

# 声调标记优先级：按 a>o>e>i>u>ü 顺序
_TONE_MARK_PRIORITY = ["a", "o", "e", "i", "u", "ü"]

# 韵母中的主元音（用于声调标记位置）
_MAIN_VOWELS = ["a", "o", "e", "i", "u", "ü"]


def _apply_tone_mark(pinyin: str, tone: int) -> str:
    """
    给拼音串加上声调标记
    
    例如: "bei" + 3 → "běi"
    """
    if tone == 0 or tone == 5:
        return pinyin  # 轻声不加标记，或者未知声调
    
    # 找到要加声调标记的元音
    # 规则：a 或 o 一定标；否则标最后一个 e；否则标最后一个出现的元音
    target_vowel = None
    for v in _TONE_MARK_PRIORITY:
        if v in pinyin:
            target_vowel = v
            break
    
    if target_vowel is None:
        # 没有匹配的元音 → 直接加数字
        return pinyin + str(tone)
    
    # 替换目标元音
    tone_char = _TONE_MARKS[target_vowel][tone]
    # 只替换第一个出现的
    result = pinyin.replace(target_vowel, tone_char, 1)
    return result


def _zero_initial_transform(final: str) -> str:
    """
    零声母时做 y/w 转换
    
    i → yi, u → wu, ü → yu, 等等
    """
    # 检查是否是 i 开头的韵母
    if final.startswith("i"):
        if final in _I_TO_Y:
            return _I_TO_Y[final]
        # i 后面有别的元音（如 ian, iang）
        return "y" + final
    
    # 检查是否是 u 开头的韵母
    if final.startswith("u"):
        if final in _U_TO_W:
            return _U_TO_W[final]
        return "w" + final
    
    # 检查是否是 ü 开头的韵母
    if final.startswith("ü"):
        if final in _U_DOT_REMOVE:
            return _U_DOT_REMOVE[final]
        return "y" + final[1:]  # ü → yu
    
    return final


def _check_valid_combo(initial: str, final: str) -> bool:
    """检查声母韵母组合是否合法"""
    if initial == "":
        return True  # 零声母总是合法
    
    if initial in _VALID_COMBOS:
        valid_set = _VALID_COMBOS[initial]
        if valid_set == "__all__":
            return True
        return final in valid_set
    
    return True  # 未知声母组合默认通过


def fuse_syllable(initial_result: dict, final_result: dict,
                  tone_result: dict, debug: bool = False) -> dict:
    """
    音节融合主入口
    
    组合声母 + 韵母 + 声调 → 带调拼音
    
    参数:
        initial_result: classify_initial 的返回
        final_result: classify_final 的返回
        tone_result: classify_tone 的返回
        debug: 是否保留调试信息
        
    返回:
    {
        "pinyin": str,              # 带调拼音 "wǒ"
        "pinyin_numbered": str,      # 数字标调 "wo3"
        "initial": str,             # 声母
        "final": str,               # 韵母
        "tone": int,                # 声调 1-5
        "confidence": float,        # 综合置信度
        "debug_info": dict,         # 白盒追溯（debug=True时）
    }
    """
    initial = initial_result.get("initial", "")
    final = final_result.get("final", "")
    tone = tone_result.get("tone", 0)
    
    confidence = (initial_result.get("confidence", 0.5)
                  + final_result.get("confidence", 0.5)
                  + tone_result.get("confidence", 0.5)) / 3.0
    
    debug_info = {} if debug else None
    
    if debug_info is not None:
        debug_info.update({
            "initial": {"initial": initial_result.get("initial",""), "confidence": initial_result.get("confidence",0)},
            "final": {"final": final_result.get("final",""), "confidence": final_result.get("confidence",0)},
            "tone": {"tone": tone_result.get("tone",0), "confidence": tone_result.get("confidence",0)},
            "combo_valid": None,
        })
    
    # 零声母处理
    if initial == "":
        pinyin_base = _zero_initial_transform(final)
        if debug_info is not None:
            debug_info["zero_initial_transform"] = {
                "original_final": final,
                "transformed": pinyin_base,
            }
    else:
        pinyin_base = initial + final
    
    # 检查组合合法性
    if initial != "" and not _check_valid_combo(initial, final):
        # 组合不合法，降置信度
        confidence *= 0.5
        if debug_info is not None:
            debug_info["combo_valid"] = False
            debug_info["combo_issue"] = f"{initial}+{final} is not a valid Mandarin syllable"
    
    # 加声调标记
    pinyin_numbered = pinyin_base + str(tone) if tone > 0 else pinyin_base
    pinyin = _apply_tone_mark(pinyin_base, tone)
    
    return {
        "pinyin": pinyin,
        "pinyin_numbered": pinyin_numbered,
        "initial": initial,
        "final": final,
        "tone": tone,
        "confidence": round(confidence, 4),
        "debug_info": debug_info,
    }
