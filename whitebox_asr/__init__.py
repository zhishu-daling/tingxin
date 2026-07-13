"""
白盒声学拼音识别引擎 (Whitebox ASR)

从音频到带调拼音的纯规则引擎。
- 无神经网络
- 无 GPU
- 白盒可追溯：每个拼音结果可回溯到哪个声学特征触发了哪个公式

架构:
    L0: 用户输入 (音频流/文件)
    L1: 音频预处理层 -> audio_loader.py
    L2: 声学基元提取层 -> primitive_extractor.py
    L3: 拼音白盒公式层
        L3.1 声母分类 -> initial_classifier.py
        L3.2 韵母分类 -> final_classifier.py
        L3.3 声调分类 -> tone_classifier.py
        L3.4 音节融合 -> syllable_fuser.py
    L4: 拼音→汉字转换 -> pinyin_to_hanzi.py
    L5: 输出与集成 -> engine.py

模块接口协议:
    每个模块接收 dict 参数，返回 dict 结果。
    所有中间数据都可通过 debug=True 获取。
"""

from .engine import WhiteboxASR

__version__ = "0.1.0"
