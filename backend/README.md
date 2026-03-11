# Speech2Vec Backend

基于 **ECAPA-TDNN** + **SpeechMapper** 的双通道向量提取后端（纯 PyTorch 实现），实现语义-韵律分析。

## 架构

```
音频文件 (.wav/.mp3/...)
    │
    ▼
┌─────────────────────────────────────┐
│   mel_spectrogram.py                │
│   梅尔频谱图提取器                    │
│   STFT(n_fft=1024, win=64ms,       │
│        hop=16ms) → 80ch MelFilter  │
│   输出: [B, 80, T]                  │
└───────────┬─────────────┬───────────┘
            │             │
            ▼             ▼
┌───────────────────┐ ┌───────────────────┐
│ prosody_encoder.py│ │semantic_encoder.py│
│ ECAPA-TDNN        │ │ SpeechMapper      │
│ 韵律编码器        │ │ 语义编码器         │
│ SE-Res2Block ×3   │ │ Conv1D(k=6,s=2)  │
│ + ASP + FC        │ │ + Transformer ×6  │
│ ~14M 参数         │ │ + FFN Projector   │
│ 输出: [B, 192]    │ │ ~70M 参数         │
└───────┬───────────┘ │ 输出: [B, 768]    │
        │             └───────┬───────────┘
        │                     │
        ▼                     ▼
  192 维韵律向量        768 维语义向量
```

## 环境准备

```bash
conda activate d2l
pip install soundfile
```

依赖清单：`torch`, `torchaudio`, `numpy`, `soundfile` — 无需额外框架。

## 训练模型

本项目包含两套训练代码，分别用于训练韵律编码器和语义编码器。

### 1. 训练语义编码器 (SpeechMapper) - 跨模态蒸馏

你需要准备 `(语音, 文本)` 配对数据集。脚本会使用大语言模型（如 sentence-transformers）提取真实文本的高维向量，拉动语音模型的输出。

**准备数据集（测试用 LibriSpeech）**:
```bash
python download_dataset.py
# 将自动下载 LibriSpeech dev-clean 子集 (337MB) 并生成 metadata.csv
```

**单卡本地训练 (推荐 Windows / 单卡 8GB 显存适用)**:
```bash
python train_semantic.py --data_dir dataset_librispeech --batch_size 16 --epochs 50
```

### 2. 训练韵律编码器 (ECAPA-TDNN) - 自监督

无需文本标注，仅用音频重建梅尔频谱图。

```bash
# 单卡
python train.py --data_dir /path/to/wav/files --epochs 100

# 多卡 DDP (推荐 Linux 下使用)
torchrun --nproc_per_node=4 train.py --data_dir /path/to/wav/files
```

### 断点续训

```bash
torchrun --nproc_per_node=4 train.py \
    --data_dir /path/to/wav/files \
    --resume checkpoints/checkpoint_epoch50.pt
```

### 完整参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--data_dir` | (必填) | 音频文件目录 |
| `--output_dir` | checkpoints | 模型输出目录 |
| `--batch_size` | 32 | **每张卡**的 batch size |
| `--lr` | 0.001 | 基础学习率（自动按卡数线性缩放） |
| `--epochs` | 100 | 训练轮次 |
| `--segment_length` | 200 | 梅尔帧长度 (~3.2s) |
| `--resume` | None | 断点续训 checkpoint 路径 |

## 推理

### CLI 用法

```bash
# 使用训练好的模型（双通道）
python pipeline.py --audio ../test/audio.wav \
    --prosody_checkpoint checkpoints/encoder_best.pt \
    --semantic_checkpoint checkpoints/semantic_best.pt

# 保存向量
python pipeline.py --audio ../test/audio.wav --output vectors.npz

# JSON 输出
python pipeline.py --audio ../test/audio.wav --json

# 仅语义编码器
python semantic_encoder.py --audio ../test/audio.wav
```

### Python API

```python
from pipeline import Speech2VecPipeline

pipeline = Speech2VecPipeline(
    prosody_checkpoint="checkpoints/encoder_best.pt",
    semantic_checkpoint="checkpoints/semantic_best.pt",
)
result = pipeline.process("path/to/audio.wav")

prosody_vector  = result["prosody_vector"]    # np.ndarray [192]
semantic_vector = result["semantic_vector"]   # np.ndarray [768]
mel             = result["mel_spectrogram"]   # np.ndarray [80, T]
```

## 技术参数

| 参数 | 值 | 说明 |
|---|---|---|
| 采样率 | 16,000 Hz | 标准语音采样率 |
| FFT 大小 | 1,024 | 频率区间数 |
| 窗口长度 | 1,024 (64ms) | STFT 窗口 |
| 步幅 | 256 (16ms) | STFT hop |
| 梅尔通道 | 80 | 梅尔滤波器组 |
| 韵律嵌入维度 | 192 | ECAPA-TDNN 输出 |
| 语义嵌入维度 | 768 | SpeechMapper 输出 |
| 韵律编码器参数量 | ~14M | ECAPA-TDNN |
| 语义编码器参数量 | ~70M | SpeechMapper |

## 语义编码器 — SpeechMapper (ICASSP 2026)

基于论文 *"SpeechMapper: Speech-to-text Embedding Projector for LLMs"*
(Mohapatra, Boito & Calapodescu, ICASSP 2026) 的核心架构。

### 模型结构

| 层 | 参数 | 说明 |
|---|---|---|
| Input Projection | Conv1D ×2, 80→1024 | 梅尔通道映射到模型维度 |
| Conv1D 下采样 | kernel=6, stride=2 | 序列长度减半 |
| Transformer ×6 | d=1024, heads=8, ffn=2048 | Pre-LN, GELU 激活 |
| Attentive Pooling | attn_dim=256 | 变长→固定向量 |
| FFN Projector | 1024→2048→768 | 投射到语义空间 |

## 文件结构

```
backend/
├── mel_spectrogram.py   # 梅尔频谱图提取器
├── prosody_encoder.py   # ECAPA-TDNN 韵律编码器
├── semantic_encoder.py  # SpeechMapper 语义编码器 [NEW]
├── train.py             # 自监督训练脚本
├── pipeline.py          # Speech2Vec 统一管线 + CLI
├── test_verify.py       # 验证测试脚本
└── README.md            # 本文档
```
