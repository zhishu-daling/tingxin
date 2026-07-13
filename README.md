# 听心 · TingXin

**端到端 AI 语音情感分析系统 — 用独创编码技术，让 AI 听懂人类的声音。**

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-yellow.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-GPU-orange.svg)](https://pytorch.org/)

---

## 这是什么

人类表达自己时，70% 的信息不在文字里 — 在语调、停顿、呼吸、语速中。但所有语音 AI 只做 ASR（把语音转成文字），丢掉了这些信息。

**听心**读取音频文件，输出的不只是文字转录，还有：

- 🧠 **三维情绪建模** — 唤醒度 / 效价 / 控制度，不是简单的"开心/难过"二分法
- 🛡️ **防御机制识别** — 压抑、否认、投射、合理化等 10 种心理防御机制
- ⚡ **声文矛盾检测** — 嘴上说的和声音传递的情绪是否一致
- 🎭 **伪装检测** — 过度控制 + 语境矛盾，识别刻意表演
- 💥 **爆发预测** — 基于累积压力模型预测情绪爆发风险
- 👥 **双人交互分析** — 能量耦合度、音高追随度、关系模式判定

---

## 核心架构

```
音频输入 (.mp3/.wav/.m4a)
        │
        ▼
┌──────────────────┐
│  音频预处理       │  重采样 16kHz + VAD 切句
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  语音识别 (ASR)   │  Whisper GPU 加速
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  说话人分离       │  MFCC + K-Means 聚类
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  声学特征提取     │  能量 / 音高 / 共振峰 / MFCC / 微停顿
└────────┬─────────┘  4 线程并行
         │
         ▼
┌──────────────────┐
│  ★ MEITR 分析    │  三维情绪 + 防御机制 + 矛盾检测 + 伪装
│                  │  + 爆发预测 + 双人交互
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  白话总结生成     │  技术指标 → 自然语言
└──────────────────┘
```

---

## 快速开始

### 环境要求

- Python 3.11+
- CUDA GPU（推荐，Whisper 转写加速）
- Windows / Linux

### 安装

```bash
git clone https://github.com/Darling-XiaoGan/tingxin.git
cd tingxin
pip install -r requirements.txt
```

### 使用

**命令行分析（最常用）**

```python
from core.pipeline import run

result = run("你的音频.mp3", output_dir="output/")
# result 包含逐句转录 + 情绪分析 + 防御机制 + 矛盾检测等
```

**Web 版**

```bash
python app.py
# 浏览器访问 http://localhost:5000
```

**桌面 GUI**

```bash
python gui.py
```

---

## 输出示例

```json
{
  "filename": "对话录音.mp3",
  "duration": 776.43,
  "num_speakers": 2,
  "segments": [
    {
      "start": 0.0,
      "end": 10.16,
      "text": "其实我没什么感觉，都还好",
      "speaker": "SPK1",
      "analysis": {
        "arousal": 0.53,
        "valence": -0.03,
        "dominance": 0.31,
        "label": "激动",
        "conflict_score": 0.38,
        "defense_mechanism": "压抑",
        "fabrication_score": 0.02,
        "prediction": {
          "p_suppress": 0.21,
          "alert": null
        }
      }
    }
  ]
}
```

> 说话人嘴上说"没什么感觉"，但声学分析显示高唤醒度 + 负效价 + 压抑防御 → **声文矛盾**。

---

## 应用场景

| 场景 | 能力 | 价值 |
|------|------|------|
| **心理咨询辅助** | 情绪曲线 + 防御机制 + 爆发预测 | AI 辅助咨询师，听见来访者没说出口的 |
| **客服质检** | 情绪拐点 + 伪装检测 + 投诉预警 | 不只听说了什么，听出怎么说的 |
| **内容创作** | 情绪可视化 + 关系分析 | 播客/访谈的情绪节奏一目了然 |
| **教育评估** | 犹豫检测 + 理解度判断 | 分辨学生"真懂了"还是"装懂了" |

---

## 性能

| 场景 | 耗时 |
|------|------|
| 8 分钟音频 (~500 句) | ~3.5 分钟 |
| 53 分钟音频 (~1500 句) | ~11 分钟 |

*基于 GPU 加速 + 4 线程并行特征提取*

---

## 技术栈

- **ASR**: OpenAI Whisper (GPU)
- **音频处理**: librosa, numpy, scipy
- **机器学习**: scikit-learn (K-Means 说话人分离)
- **分析引擎**: 自研 MEITR 五维协同框架
- **前端**: Flask (Web), Tkinter (GUI)

---

## 项目结构

```
tingxin/
├── app.py                  # Web 应用入口
├── gui.py                  # 桌面 GUI
├── run_analysis.py         # 命令行分析
├── core/
│   ├── preprocess.py       # 音频预处理
│   ├── asr.py              # 语音识别
│   ├── diarize.py          # 说话人分离
│   ├── acoustic_features.py # 声学特征提取
│   ├── analysis.py         # ★ MEITR 核心分析引擎
│   ├── call_analysis.py    # 客服场景分析
│   ├── meeting_minutes.py  # 会议纪要生成
│   ├── music_analysis.py   # 音乐分析
│   ├── stream_pipeline.py  # 管线编排
│   └── sound_primitives.py # 声音基元
└── templates/              # Web 模板
```

---

## 许可证

本项目使用 [GNU General Public License v3.0](LICENSE) 许可证。

---

## 联系方式

如有合作意向或技术交流，欢迎联系：

- GitHub: [@Darling-XiaoGan](https://github.com/Darling-XiaoGan)
- Email: a944984537@126.com
