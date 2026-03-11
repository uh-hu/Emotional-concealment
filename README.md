# Emotional Concealment

语义——韵律分析平台：多模态情感计算与认知解析系统。

## 项目结构

```
├── demo/           # 前端 Web 界面
│   ├── index.html
│   ├── script.js
│   ├── styles.css
│   └── assets/
├── backend/        # Speech2Vec 后端（韵律 + 语义双通道）
│   ├── mel_spectrogram.py   # 梅尔频谱图提取
│   ├── prosody_encoder.py   # ECAPA-TDNN 韵律编码器 → 192 维
│   ├── semantic_encoder.py  # SpeechMapper 语义编码器 → 768 维
│   ├── pipeline.py          # Speech2Vec 统一管线
│   ├── train.py             # 自监督训练脚本
│   └── README.md
└── test/           # 测试音频文件
```

## 快速开始

### 后端

```bash
conda activate d2l
cd backend
python pipeline.py --audio ../test/your_audio.wav
```

详见 [backend/README.md](backend/README.md)。
