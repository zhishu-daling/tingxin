# 白盒声学拼音识别引擎 — 模块接口协议

## 文件清单与职责

```
whitebox_asr/
├── __init__.py              # 包入口，暴露 WhiteboxASR
├── MODULE_INTERFACES.md     # 本文件：接口协议
│
├── audio_loader.py          # L1: 音频加载 + 重采样 + VAD 切段
├── primitive_extractor.py   # L2: 声学基元提取封装
│
├── initial_classifier.py    # L3.1: 声母分类 (爆破/擦音/塞擦/鼻/边音)
├── final_classifier.py      # L3.2: 韵母分类 (单元音/复韵/鼻韵/卷舌)
├── tone_classifier.py       # L3.3: 声调分类 (F0 决策树)
├── syllable_fuser.py        # L3.4: 音节融合 (声母+韵母+声调→拼音)
│
├── pinyin_to_hanzi.py       # L4: 拼音→汉字转换
│
└── engine.py                # L5: 总装引擎，对外接口
```

## 模块接口协议

### audio_loader.py (L1)

```python
def load_and_vad(audio_path: str, sr: int = 16000) -> dict
    """
    加载音频 → VAD切段 → 返回有声段列表
    
    返回:
    {
        "file": str,                    # 源文件路径
        "sr": int,                      # 采样率
        "duration_s": float,            # 总时长
        "segments": [                    # 有声段列表
            {
                "index": int,
                "start_s": float,        # 起始时间
                "end_s": float,          # 结束时间
                "duration_s": float,
                "audio": np.ndarray,     # 波形 (float32, [-1,1])
            },
            ...
        ],
        "vad_details": [...]            # 原始VAD分段（含静音段）
    }
    """

def load_and_vad_from_array(audio: np.ndarray, sr: int = 16000) -> dict
    """
    从 numpy 数组加载（用于流式或内存音频）
    返回同上
    """
```

### primitive_extractor.py (L2)

```python
def extract_frame_features(audio: np.ndarray, sr: int) -> dict
    """
    逐帧提取声学特征 (帧长25ms, 帧移10ms)
    
    返回:
    {
        "mfcc": np.ndarray,             # (13, n_frames)
        "f0": np.ndarray,               # (n_frames,) 基频，无声处=0
        "voiced": np.ndarray,           # (n_frames,) bool
        "spectral_centroid": np.ndarray,# (n_frames,)
        "spectral_bandwidth": np.ndarray,
        "spectral_contrast": np.ndarray,
        "rms_energy": np.ndarray,       # (n_frames,)
        "zcr": np.ndarray,              # (n_frames,)
        "formants": np.ndarray,         # (4,) 整段共振峰
        "frame_times": np.ndarray,      # (n_frames,) 每帧时间戳
    }
    """

def extract_segment_features(frames: dict) -> dict
    """
    对帧特征做片段聚合
    
    返回:
    {
        "mfcc_mean": list[13],
        "mfcc_std": list[13],
        "pitch_mean": float,
        "pitch_std": float,
        "pitch_contour": np.ndarray,    # 去野点后的F0序列
        "spectral_centroid_mean": float,
        "energy_mean": float,
        "energy_std": float,
        "zcr_mean": float,
        "formants": list[4],
        "formant_tracks": dict,          # {f1: [...], f2: [...], ...}
        "duration_s": float,
        "frame_count": int,
    }
    """
```

### initial_classifier.py (L3.1)

```python
def classify_initial(segment_features: dict, frame_features: dict) -> dict
    """
    声母分类
    
    返回:
    {
        "initial": str,                 # 声母标签: "b","p","m","f","d","t","n","l",
                                        #            "g","k","h","j","q","x",
                                        #            "zh","ch","sh","r","z","c","s",
                                        #            ""(零声母)
        "initial_type": str,            # 类型: plosive/fricative/affricate/nasal/lateral/null
        "confidence": float,            # 0~1
        "evidence": dict,               # 触发证据（用于白盒回溯）
    }
    """

def detect_plosive(segment_features: dict, frame_features: dict) -> dict
def detect_fricative(segment_features: dict, frame_features: dict) -> dict
def detect_affricate(segment_features: dict, frame_features: dict) -> dict
def detect_nasal(segment_features: dict, frame_features: dict) -> dict
def detect_lateral(segment_features: dict, frame_features: dict) -> dict
```

### final_classifier.py (L3.2)

```python
def classify_final(segment_features: dict, frame_features: dict, templates: dict = None) -> dict
    """
    韵母分类
    
    返回:
    {
        "final": str,                   # 韵母标签: "a","o","e","i","u","ü",
                                        #            "ai","ei","ui","ao","ou","iu","ie","üe","er",
                                        #            "an","en","in","un","ün",
                                        #            "ang","eng","ing","ong", 等
        "final_type": str,              # 类型: monophthong/diphthong/nasal_final/retroflex
        "confidence": float,
        "evidence": dict,
    }
    """

def build_vowel_templates(template_dir: str) -> dict
    """
    录制/加载单元音模板
    返回 { "a": {"f1_mean": ..., "f2_mean": ..., "samples": [...]}, ... }
    """
```

### tone_classifier.py (L3.3)

```python
def classify_tone(f0_contour: np.ndarray, frame_times: np.ndarray, 
                  global_f0_median: float = None) -> dict
    """
    声调分类
    
    返回:
    {
        "tone": int,                    # 1/2/3/4/5(轻声) / 0(无法判断)
        "tone_label": str,              # "55","35","214","51","轻声"
        "confidence": float,
        "features": {                   # F0统计
            "f0_start": float,
            "f0_end": float,
            "f0_max": float,
            "f0_min": float,
            "f0_mean": float,
            "f0_std": float,
            "slope": float,             # 线性回归斜率
            "has_valley": bool,         # 是否有谷点（用于214）
        },
        "evidence": dict,
    }
    """
```

### syllable_fuser.py (L3.4)

```python
def fuse_syllable(initial_result: dict, final_result: dict, 
                  tone_result: dict, debug: bool = False) -> dict
    """
    声母 + 韵母 + 声调 → 带调拼音
    
    返回:
    {
        "pinyin": str,                  # 如 "wo3"
        "pinyin_without_tone": str,     # 如 "wo"
        "initial": str,                 # 声母
        "final": str,                   # 韵母
        "tone": int,                    # 声调
        "debug_info": dict,             # 白盒追溯信息
    }
    """
```

### pinyin_to_hanzi.py (L4)

```python
def pinyin_to_text(pinyin_list: list[dict], 
                   user_dict: dict = None) -> dict
    """
    拼音串 → 汉字
    
    返回:
    {
        "text": str,                    # 最终汉字文本
        "candidates": list[list[str]],  # 各位置候选字
        "debug_info": dict,
    }
    """
```

### engine.py (L5)

```python
class WhiteboxASR:
    def __init__(self, templates_dir: str = None, user_dict: dict = None)
    
    def transcribe(self, audio_path: str, debug: bool = False) -> dict
        """
        全流水线：音频 → 拼音 → 汉字
        
        返回:
        {
            "text": str,                # 最终汉字文本
            "pinyin_segments": [        # 逐段详细结果
                {
                    "segment_index": int,
                    "start_s": float,
                    "end_s": float,
                    "syllables": [       # 逐音节
                        {
                            "pinyin": str,
                            "initial": str,
                            "final": str,
                            "tone": int,
                            "confidence": float,
                            "evidence": dict,
                        }
                    ],
                    "text": str,        # 该段汉字
                }
            ],
            "debug": dict if debug else None,
        }
        """
    
    def transcribe_to_pinyin(self, audio_path: str) -> list[str]
        """仅输出拼音（调试用）"""
```
