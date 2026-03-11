from __future__ import annotations

"""
SpeechMapper 语义编码器训练脚本 — 专为单卡本地训练优化（RTX 5060等）
Semantic Encoder Training Script — Cross-modal Alignment Distillation

训练目标: 监督学习 / 跨模态知识蒸馏
利用预训练的大语言文本嵌入模型（如 sentence-transformers）作为教师网络，
引导 SpeechMapper（学生网络）将梅尔频谱图投射到相同的 768 维文本语义空间中。

架构:
    输入文本 [Text] → SentenceTransformer (Teacher) → Target 嵌入 [768]
    输入音频 [Wav] → SpeechMapper (Student) → Pred 嵌入 [768]
     Loss = MSE(Pred, Target) + 1 - CosineSimilarity(Pred, Target)

本地单卡（Windows）运行优化:
    1. DataLoader num_workers 默认设为 0 (避免 Windows 子进程报错)
    2. batch_size 适配 8GB 显存 (默认 16-32 流畅运行)
"""

import os
import random
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from mel_spectrogram import MelSpectrogramExtractor
from semantic_encoder import SpeechMapper

# 尝试导入 transformers，如果没有则提示用户安装
try:
    from transformers import AutoTokenizer, AutoModel
except ImportError:
    print("运行此脚本需要安装 transformers。")
    print("请先执行: pip install transformers")
    exit(1)


# ═══════════════════════════════════════════════════════════════
# 文本教师模型 (Text Embedding Teacher)
# ═══════════════════════════════════════════════════════════════


class TextEmbeddingTeacher(nn.Module):
    """文本嵌入模型 (教师网络)。
    
    使用 Hugging Face 的 sentence-transformers 模型提取高质量文本向量。
    模型权重被冻结，仅作为特征提取器。
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2", device: str = "cpu"):
        super().__init__()
        self.device = device
        print(f"[Teacher] Loading text model '{model_name}'...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        
        # 冻结教师模型权重
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.model.eval()
        
    @torch.no_grad()
    def get_embeddings(self, texts: list[str]) -> torch.Tensor:
        """提取一组文本的固定维度嵌入向量。
        
        方法: mean pooling 并经过 L2 normalization (sentence-transformers 的标准做法)
        """
        # Tokenization
        encoded_input = self.tokenizer(
            texts, padding=True, truncation=True, return_tensors='pt'
        ).to(self.device)
        
        # Forward pass
        model_output = self.model(**encoded_input)
        
        # Mean Pooling - 考虑 attention_mask
        attention_mask = encoded_input['attention_mask']
        token_embeddings = model_output[0] # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        sentence_embeddings = sum_embeddings / sum_mask
        
        # L2 归一化 (可选，但推荐对于余弦相似度极佳)
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        
        return sentence_embeddings


# ═══════════════════════════════════════════════════════════════
# 训练数据集
# ═══════════════════════════════════════════════════════════════


class SpeechTextDataset(Dataset):
    """语音-文本配对数据集。
    
    期望一个 metadata.csv 文件，格式为：
    audio_path|text
    
    Parameters
    ----------
    data_dir : str
        包含 metadata.csv 和音频文件的根目录。
    segment_length : int
        音频梅尔图切割的固定长度（帧数）。
    """
    
    def __init__(self, data_dir: str, segment_length: int = 400):
        self.data_dir = Path(data_dir)
        self.segment_length = segment_length
        self.mel_extractor = MelSpectrogramExtractor()
        
        metadata_file = self.data_dir / "metadata.csv"
        if not metadata_file.exists():
            raise FileNotFoundError(
                f"未找到 {metadata_file}。\n"
                f"请先运行 download_dataset.py 下载并生成映射文件！"
            )
            
        self.samples = []
        with open(metadata_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            # 跳过表头
            for line in lines[1:]:
                line = line.strip()
                if not line: continue
                parts = line.split("|", 1)
                if len(parts) == 2:
                    audio_rel_path, text = parts
                    self.samples.append({
                        "audio_path": str(self.data_dir / audio_rel_path),
                        "text": text
                    })
                    
        print(f"[Dataset]Loaded {len(self.samples)} speech-text pairs.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        audio_path = sample["audio_path"]
        text = sample["text"]

        try:
            result = self.mel_extractor.from_file(audio_path)
            mel = result["mel"].squeeze(0)  # [80, T]
        except Exception as e:
            # 读取失败时返回全零输入和空字符串
            mel = torch.zeros(80, self.segment_length)
            text = ""
            
        T = mel.shape[-1]
        
        # 统一音频长度
        if T > self.segment_length:
            # 随机截断
            start = random.randint(0, T - self.segment_length)
            mel = mel[:, start:start + self.segment_length]
        elif T < self.segment_length:
            # 补零
            mel = F.pad(mel, (0, self.segment_length - T))

        return mel, text


# ═══════════════════════════════════════════════════════════════
# 语义模型训练器
# ═══════════════════════════════════════════════════════════════


class SemanticTrainer:
    """跨模态蒸馏语义训练器。"""
    
    def __init__(
        self,
        data_dir: str,
        output_dir: str = "checkpoints",
        batch_size: int = 16,
        lr: float = 2e-4,
        epochs: int = 50,
        segment_length: int = 400,
        num_workers: int = 0,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.epochs = epochs
        
        print("\n" + "="*50)
        print("  SpeechMapper Semantic Distillation Training")
        print("="*50)
        print(f"Device:       {self.device}")
        print(f"VRAM safe:    Batch Size = {batch_size}")
        print(f"Workers:      {num_workers} (Windows recommends 0)")
        
        # ─── 1. 初始化模型 ───
        print("\n[Init] Building models...")
        self.teacher = TextEmbeddingTeacher(device=self.device)
        # 指定输出 768 维与 mpnet-base-v2 的输出对齐
        self.student = SpeechMapper(semantic_dim=768).to(self.device)
        
        # ─── 2. 准备数据集 ───
        print("\n[Init] Preparing dataset...")
        dataset = SpeechTextDataset(data_dir, segment_length=segment_length)
        # Windows 上多进程 DataLoader 很容易导致卡死，默认 num_workers=0
        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )
        
        # ─── 3. 优化器与损失 ───
        self.optimizer = torch.optim.AdamW(self.student.parameters(), lr=lr, weight_decay=1e-2)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)
        
        self.mse_loss = nn.MSELoss()
        self.cosine_loss = nn.CosineEmbeddingLoss()

    def train(self):
        best_loss = float('inf')
        
        print("\n[Train] Starts...")
        
        for epoch in range(1, self.epochs + 1):
            self.student.train()
            
            total_loss = 0.0
            num_batches = 0
            
            for batch_idx, (mels, texts) in enumerate(self.dataloader):
                mels = mels.to(self.device)
                
                # 有些音频读取失败时 text 为空，这里过滤一下
                if not any(texts):
                    continue
                
                # --- 前向传播教师（文本） ---
                # target 是来自 LLM 嵌入空间的真实锚点
                with torch.no_grad():
                    target_embs = self.teacher.get_embeddings(texts)  # [B, 768]
                
                # --- 前向传播学生（音频） ---
                pred_embs = self.student(mels)  # [B, 768]
                
                # 为了计算余弦损失，所有标签都为 1（表示让它们尽量相似）
                targets_for_cosine = torch.ones(mels.size(0)).to(self.device)
                
                # --- 计算混合损失 ---
                # MSE 让尺度不变，Cosine 让方向强对齐
                loss_mse = self.mse_loss(pred_embs, target_embs)
                loss_cos = self.cosine_loss(pred_embs, target_embs, targets_for_cosine)
                
                loss = loss_mse + loss_cos
                
                # --- 反向传播 ---
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=3.0)
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                if (batch_idx + 1) % 10 == 0:
                    avg = total_loss / num_batches
                    print(f"  Epoch [{epoch}/{self.epochs}] "
                          f"Batch [{batch_idx+1}/{len(self.dataloader)}] "
                          f"Loss: {loss.item():.4f} (MSE:{loss_mse.item():.4f} Cos:{loss_cos.item():.4f}) "
                          f"avg: {avg:.4f}")
            
            # --- Epoch 结束 ---
            avg_loss = total_loss / max(num_batches, 1)
            current_lr = self.scheduler.get_last_lr()[0]
            self.scheduler.step()
            
            print(f"> Epoch {epoch}/{self.epochs} Complete | "
                  f"Avg Loss: {avg_loss:.4f} | LR: {current_lr:.6f}")
            
            # 保存逻辑
            if avg_loss < best_loss:
                best_loss = avg_loss
                path = self.output_dir / "semantic_best.pt"
                torch.save(self.student.state_dict(), path)
                print(f"  -> 保存了新的最优模型: {path}")
                
            if epoch % 10 == 0:
                ckpt_path = self.output_dir / f"semantic_epoch{epoch}.pt"
                torch.save(self.student.state_dict(), ckpt_path)
                
        print(f"\nTraining Complete! Best Loss: {best_loss:.4f}")
        print("你现在可以使用这个最优模型调用管线:\n"
              "python pipeline.py --audio test/audio.wav --semantic_checkpoint checkpoints/semantic_best.pt")

# ─────────────────────────── CLI 入口 ───────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SpeechMapper Semantic Encoder Distillation Training (Single GPU)")
    parser.add_argument("--data_dir", required=True, help="Directory containing metadata.csv and paired audios")
    parser.add_argument("--output_dir", default="checkpoints", help="Output directory for checkpoints")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size (default: 16, safe for 8GB VRAM)")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    
    args = parser.parse_args()
    
    trainer = SemanticTrainer(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        num_workers=0  # 写死在 Windows 上，避免产生多进程卡死
    )
    trainer.train()

if __name__ == "__main__":
    main()
