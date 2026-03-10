# Prosody2Vec Backend

基于 **ECAPA-TDNN** 的韵律向量提取后端（纯 PyTorch 实现），实现 Prosody2Vec 论文的两个核心部件。

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
└───────────────┬─────────────────────┘
                │
                ▼
┌─────────────────────────────────────┐
│   prosody_encoder.py                │
│   ECAPA-TDNN 韵律编码器 (纯 PyTorch) │
│   SE-Res2Block × 3 + ASP + FC      │
│   ~14M 参数                         │
│   输出: [B, 192]                    │
└───────────────┬─────────────────────┘
                │
                ▼
         192 维韵律向量
```

## 环境准备

```bash
# 激活 conda 环境 (需要 PyTorch + torchaudio)
conda activate d2l

# 安装额外依赖 (仅需 soundfile)
pip install soundfile
```

依赖清单：`torch`, `torchaudio`, `numpy`, `soundfile` — 无需 SpeechBrain。

## 训练模型

支持单卡和 **PyTorch DDP 多卡分布式训练**。

### 单卡训练

```bash
python train.py --data_dir /path/to/wav/files --epochs 100
```

### 多卡 DDP 训练 (推荐)

```bash
# 单机 4 卡
torchrun --nproc_per_node=4 train.py \
    --data_dir /path/to/wav/files \
    --batch_size 32 \
    --epochs 100

# 多节点 (例如 2 节点 × 4 卡 = 8 卡)
torchrun --nnodes=2 --nproc_per_node=4 \
    --node_rank=0 \
    --master_addr=<MASTER_IP> --master_port=29500 \
    train.py --data_dir /path/to/wav/files --epochs 100
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

训练完成后会保存：
- `checkpoints/encoder_best.pt` — 最优编码器权重（推理使用）
- `checkpoints/encoder_final.pt` — 最终编码器权重
- `checkpoints/checkpoint_epochN.pt` — 完整 checkpoint（含解码器、优化器，可断点续训）

## 推理

### CLI 用法

```bash
# 使用训练好的模型
python pipeline.py --audio ../test/audio.wav --checkpoint checkpoints/encoder_best.pt

# 保存向量
python pipeline.py --audio ../test/audio.wav --checkpoint checkpoints/encoder_best.pt --output prosody.npy

# JSON 输出
python pipeline.py --audio ../test/audio.wav --checkpoint checkpoints/encoder_best.pt --json
```

### Python API

```python
from pipeline import Prosody2VecPipeline

pipeline = Prosody2VecPipeline(checkpoint_path="checkpoints/encoder_best.pt")
result = pipeline.process("path/to/audio.wav")

prosody_vector = result["prosody_vector"]   # np.ndarray [192]
mel = result["mel_spectrogram"]              # np.ndarray [80, T]
```

## 技术参数

| 参数 | 值 | 说明 |
|---|---|---|
| 采样率 | 16,000 Hz | 标准语音采样率 |
| FFT 大小 | 1,024 | 频率区间数 |
| 窗口长度 | 1,024 (64ms) | STFT 窗口 |
| 步幅 | 256 (16ms) | STFT hop |
| 梅尔通道 | 80 | 梅尔滤波器组 |
| 嵌入维度 | 192 | ECAPA-TDNN 输出 |
| 编码器参数量 | ~14M | ECAPA-TDNN |

## 文件结构

```
backend/
├── mel_spectrogram.py   # 梅尔频谱图提取器
├── prosody_encoder.py   # ECAPA-TDNN 编码器 (纯 PyTorch)
├── train.py             # 自监督训练脚本
├── pipeline.py          # 统一管线 + CLI
├── test_verify.py       # 验证测试脚本
├── requirements.txt     # Python 依赖
└── README.md            # 本文档
```
