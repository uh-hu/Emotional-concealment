from __future__ import annotations

"""
Prosody2Vec 训练脚本 — 支持 PyTorch DDP 多卡分布式训练
Prosody2Vec Training Script — PyTorch DistributedDataParallel

训练目标: 自监督学习——用梅尔频谱图解码器重建输入，
迫使 ECAPA-TDNN 编码器学习捕获韵律信息的 192 维向量。

训练架构:
    梅尔频谱图 [B, 80, T]
        → ECAPA-TDNN 编码器 → 192 维向量
        → 梅尔解码器 → 重建梅尔频谱图 [B, 80, T]
    损失 = MSE(原始梅尔, 重建梅尔)

训练完成后，只保留编码器权重用于推理。

启动方式:
    # 单卡
    python train.py --data_dir /path/to/wav --epochs 100

    # 多卡 DDP (推荐)
    torchrun --nproc_per_node=4 train.py --data_dir /path/to/wav --epochs 100

    # 多节点 DDP
    torchrun --nnodes=2 --nproc_per_node=4 --node_rank=0 \
             --master_addr=<MASTER_IP> --master_port=29500 \
             train.py --data_dir /path/to/wav --epochs 100
"""

import os
import math
import random
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from mel_spectrogram import MelSpectrogramExtractor
from prosody_encoder import ECAPA_TDNN


# ═══════════════════════════════════════════════════════════════
# 梅尔解码器 (仅用于训练)
# ═══════════════════════════════════════════════════════════════


class MelDecoder(nn.Module):
    """梅尔频谱图解码器。

    将 192 维韵律向量解码回梅尔频谱图，用于自监督训练。
    架构参照简化版 Tacotron2 解码器。

    Parameters
    ----------
    embedding_dim : int
        输入嵌入维度，默认 192。
    decoder_channels : int
        解码器中间通道数，默认 512。
    n_mels : int
        输出梅尔通道数，默认 80。
    """

    def __init__(
        self,
        embedding_dim: int = 192,
        decoder_channels: int = 512,
        n_mels: int = 80,
    ):
        super().__init__()

        # 将嵌入向量扩展为序列
        self.prenet = nn.Sequential(
            nn.Linear(embedding_dim, decoder_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(decoder_channels, decoder_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )

        # 转置卷积上采样网络
        self.upsample = nn.Sequential(
            # [B, 512, 1] → [B, 512, 4]
            nn.ConvTranspose1d(decoder_channels, decoder_channels, kernel_size=4, stride=4),
            nn.BatchNorm1d(decoder_channels),
            nn.ReLU(inplace=True),

            # [B, 512, 4] → [B, 256, 16]
            nn.ConvTranspose1d(decoder_channels, 256, kernel_size=4, stride=4),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),

            # [B, 256, 16] → [B, 128, 64]
            nn.ConvTranspose1d(256, 128, kernel_size=4, stride=4),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            # [B, 128, 64] → [B, 128, 256]
            nn.ConvTranspose1d(128, 128, kernel_size=4, stride=4),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )

        # 最终投影到梅尔维度
        self.postnet = nn.Sequential(
            nn.Conv1d(128, n_mels, kernel_size=5, padding=2),
            nn.BatchNorm1d(n_mels),
            nn.Tanh(),
            # 残差后处理网络
            nn.Conv1d(n_mels, n_mels, kernel_size=5, padding=2),
        )

    def forward(self, embedding: torch.Tensor, target_length: int) -> torch.Tensor:
        """解码嵌入向量为梅尔频谱图。

        Parameters
        ----------
        embedding : torch.Tensor
            韵律嵌入，形状 [batch, 192]。
        target_length : int
            目标梅尔频谱图的时间帧数 T。

        Returns
        -------
        mel : torch.Tensor
            重建的梅尔频谱图，形状 [batch, 80, target_length]。
        """
        # [B, 192] → [B, 512]
        x = self.prenet(embedding)

        # [B, 512] → [B, 512, 1]
        x = x.unsqueeze(-1)

        # 上采样
        x = self.upsample(x)  # [B, 128, 256]

        # 投影 + 后处理
        x = self.postnet(x)  # [B, 80, 256]

        # 调整到目标长度
        if x.shape[-1] > target_length:
            x = x[:, :, :target_length]
        elif x.shape[-1] < target_length:
            x = F.pad(x, (0, target_length - x.shape[-1]))

        return x


# ═══════════════════════════════════════════════════════════════
# 训练数据集
# ═══════════════════════════════════════════════════════════════


class AudioDataset(Dataset):
    """音频文件数据集。

    自动扫描目录下所有音频文件，提取梅尔频谱图。

    Parameters
    ----------
    data_dir : str
        包含音频文件的目录。
    segment_length : int
        每个训练样本的梅尔帧数（时间长度）。
        超过此长度的音频会被随机裁剪，不足的会被填充。
    """

    AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}

    def __init__(self, data_dir: str, segment_length: int = 200):
        self.segment_length = segment_length
        self.mel_extractor = MelSpectrogramExtractor()

        # 扫描音频文件
        data_path = Path(data_dir)
        self.audio_files = sorted([
            f for f in data_path.rglob("*")
            if f.suffix.lower() in self.AUDIO_EXTENSIONS
        ])

        if len(self.audio_files) == 0:
            raise ValueError(f"No audio files found in {data_dir}")

        print(f"[Dataset] Found {len(self.audio_files)} audio files in {data_dir}")

    def __len__(self) -> int:
        return len(self.audio_files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        audio_path = self.audio_files[idx]

        try:
            result = self.mel_extractor.from_file(audio_path)
            mel = result["mel"].squeeze(0)  # [80, T]
        except Exception as e:
            # 出错时返回随机噪声（避免训练中断）
            print(f"[Dataset] Error loading {audio_path}: {e}")
            mel = torch.randn(80, self.segment_length)

        T = mel.shape[-1]

        # 统一长度
        if T > self.segment_length:
            # 随机裁剪
            start = random.randint(0, T - self.segment_length)
            mel = mel[:, start:start + self.segment_length]
        elif T < self.segment_length:
            # 右侧填零
            mel = F.pad(mel, (0, self.segment_length - T))

        return mel  # [80, segment_length]


# ═══════════════════════════════════════════════════════════════
# DDP 工具函数
# ═══════════════════════════════════════════════════════════════


def setup_ddp():
    """初始化 DDP 分布式环境。

    Returns
    -------
    (rank, local_rank, world_size, is_distributed)
    """
    # 检查是否由 torchrun 启动
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)

        return rank, local_rank, world_size, True
    else:
        # 单卡 fallback
        return 0, 0, 1, False


def cleanup_ddp():
    """清理 DDP 分布式环境。"""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    """是否是主进程（rank 0）。"""
    return rank == 0


def log(msg: str, rank: int = 0):
    """只在主进程打印日志。"""
    if is_main_process(rank):
        print(msg)


# ═══════════════════════════════════════════════════════════════
# 训练器 (支持单卡 + DDP 多卡)
# ═══════════════════════════════════════════════════════════════


class Prosody2VecTrainer:
    """Prosody2Vec 训练器 — 支持 PyTorch DDP。

    Parameters
    ----------
    data_dir : str
        训练数据目录。
    output_dir : str
        模型输出目录。
    batch_size : int
        每张卡的 batch size。
    lr : float
        基础学习率（会根据 world_size 线性缩放）。
    epochs : int
    segment_length : int
        训练样本帧长度。
    resume : str, optional
        断点续训的 checkpoint 路径。
    """

    def __init__(
        self,
        data_dir: str,
        output_dir: str = "checkpoints",
        batch_size: int = 32,
        lr: float = 1e-3,
        epochs: int = 100,
        segment_length: int = 200,
        resume: str = None,
    ):
        # ─── DDP setup ───
        self.rank, self.local_rank, self.world_size, self.distributed = setup_ddp()
        self.device = torch.device(f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu")

        self.output_dir = Path(output_dir)
        if is_main_process(self.rank):
            self.output_dir.mkdir(parents=True, exist_ok=True)

        self.epochs = epochs
        self.start_epoch = 1

        # ─── 数据 ───
        log("[Trainer] Loading dataset...", self.rank)
        dataset = AudioDataset(data_dir, segment_length=segment_length)

        if self.distributed:
            self.sampler = DistributedSampler(
                dataset, num_replicas=self.world_size, rank=self.rank, shuffle=True
            )
            shuffle = False
        else:
            self.sampler = None
            shuffle = True

        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=self.sampler,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
        )

        # ─── 模型 ───
        log("[Trainer] Building models...", self.rank)
        self.encoder = ECAPA_TDNN(
            in_channels=80, channels=1024, embedding_dim=192
        ).to(self.device)

        self.decoder = MelDecoder(
            embedding_dim=192, decoder_channels=512, n_mels=80
        ).to(self.device)

        # DDP 包装
        if self.distributed:
            self.encoder = DDP(self.encoder, device_ids=[self.local_rank])
            self.decoder = DDP(self.decoder, device_ids=[self.local_rank])

        # ─── 优化器 ───
        # 线性缩放学习率: lr * world_size
        effective_lr = lr * self.world_size
        params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        self.optimizer = torch.optim.AdamW(params, lr=effective_lr, weight_decay=1e-4)

        # 学习率调度
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs
        )

        # 损失
        self.criterion = nn.MSELoss()

        # ─── 断点续训 ───
        if resume is not None:
            self._load_checkpoint(resume)

        # ─── 日志 ───
        enc_params = sum(p.numel() for p in self.encoder.parameters())
        dec_params = sum(p.numel() for p in self.decoder.parameters())
        log(f"[Trainer] Encoder params: {enc_params:,} ({enc_params/1e6:.1f}M)", self.rank)
        log(f"[Trainer] Decoder params: {dec_params:,} ({dec_params/1e6:.1f}M)", self.rank)
        log(f"[Trainer] Device: {self.device}", self.rank)
        log(f"[Trainer] World size: {self.world_size}", self.rank)
        log(f"[Trainer] Batch size per GPU: {batch_size}", self.rank)
        log(f"[Trainer] Effective batch size: {batch_size * self.world_size}", self.rank)
        log(f"[Trainer] Effective LR: {effective_lr:.6f}", self.rank)
        log(f"[Trainer] Epochs: {self.start_epoch}-{epochs}", self.rank)
        log(f"[Trainer] Segment length: {segment_length} frames (~{segment_length * 0.016:.1f}s)", self.rank)

    def train(self):
        """执行完整训练。"""
        log("\n" + "=" * 60, self.rank)
        log("  Prosody2Vec DDP Training", self.rank)
        log("=" * 60 + "\n", self.rank)

        best_loss = float("inf")

        for epoch in range(self.start_epoch, self.epochs + 1):
            # DDP: 每个 epoch 设置 sampler 的 epoch 以保证 shuffle 不同
            if self.distributed:
                self.sampler.set_epoch(epoch)

            self.encoder.train()
            self.decoder.train()

            total_loss = 0.0
            num_batches = 0

            for batch_idx, mel in enumerate(self.dataloader):
                mel = mel.to(self.device)  # [B, 80, T]

                # 前向传播
                embedding = self.encoder(mel)                              # [B, 192]
                mel_reconstructed = self.decoder(embedding, mel.shape[-1]) # [B, 80, T]

                # 计算损失
                loss = self.criterion(mel_reconstructed, mel)

                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.encoder.parameters()) + list(self.decoder.parameters()),
                    max_norm=5.0,
                )
                self.optimizer.step()

                total_loss += loss.item()
                num_batches += 1

                if (batch_idx + 1) % 50 == 0:
                    avg = total_loss / num_batches
                    log(f"  Epoch [{epoch}/{self.epochs}] "
                        f"Batch [{batch_idx+1}/{len(self.dataloader)}] "
                        f"Loss: {loss.item():.6f} (avg: {avg:.6f})", self.rank)

            # ─── Epoch 结束 ───
            avg_loss = total_loss / max(num_batches, 1)

            # DDP: 跨所有进程同步平均 loss
            if self.distributed:
                loss_tensor = torch.tensor(avg_loss, device=self.device)
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
                avg_loss = loss_tensor.item()

            current_lr = self.scheduler.get_last_lr()[0]
            self.scheduler.step()

            log(f"Epoch {epoch}/{self.epochs} | "
                f"Loss: {avg_loss:.6f} | LR: {current_lr:.6f}", self.rank)

            # 只在主进程保存 checkpoint
            if is_main_process(self.rank):
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    self._save_checkpoint(epoch, avg_loss, is_best=True)

                if epoch % 10 == 0:
                    self._save_checkpoint(epoch, avg_loss)

            # DDP: 同步所有进程（等主进程保存完）
            if self.distributed:
                dist.barrier()

        # 训练完成
        if is_main_process(self.rank):
            self._save_checkpoint(self.epochs, avg_loss, final=True)
            log(f"\nTraining complete! Best loss: {best_loss:.6f}", self.rank)
            log(f"Checkpoints saved to: {self.output_dir}", self.rank)

        cleanup_ddp()

    def _get_raw_model(self, model: nn.Module) -> nn.Module:
        """获取 DDP 包装下的原始模型。"""
        return model.module if self.distributed else model

    def _save_checkpoint(self, epoch: int, loss: float,
                         is_best: bool = False, final: bool = False):
        """保存模型权重（只在 rank 0 调用）。"""
        encoder_state = self._get_raw_model(self.encoder).state_dict()

        if is_best:
            path = self.output_dir / "encoder_best.pt"
            torch.save(encoder_state, path)
            print(f"  -> Saved best encoder: {path}")

        if final:
            path = self.output_dir / "encoder_final.pt"
            torch.save(encoder_state, path)
            print(f"  -> Saved final encoder: {path}")

        if epoch % 10 == 0 or final:
            full_ckpt = {
                "epoch": epoch,
                "loss": loss,
                "encoder_state_dict": encoder_state,
                "decoder_state_dict": self._get_raw_model(self.decoder).state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
            }
            path = self.output_dir / f"checkpoint_epoch{epoch}.pt"
            torch.save(full_ckpt, path)
            print(f"  -> Saved full checkpoint: {path}")

    def _load_checkpoint(self, path: str):
        """加载 checkpoint 用于断点续训。"""
        log(f"[Trainer] Resuming from: {path}", self.rank)
        ckpt = torch.load(path, map_location=self.device)

        self._get_raw_model(self.encoder).load_state_dict(ckpt["encoder_state_dict"])
        self._get_raw_model(self.decoder).load_state_dict(ckpt["decoder_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.start_epoch = ckpt["epoch"] + 1

        log(f"[Trainer] Resumed at epoch {self.start_epoch}, loss: {ckpt['loss']:.6f}", self.rank)


# ─────────────────────────── CLI 入口 ───────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Prosody2Vec DDP Training: Self-supervised ECAPA-TDNN Prosody Encoder"
    )
    parser.add_argument(
        "--data_dir", required=True,
        help="Directory containing training audio files (wav/mp3/flac)"
    )
    parser.add_argument(
        "--output_dir", default="checkpoints",
        help="Directory to save model checkpoints (default: checkpoints)"
    )
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size PER GPU (default: 32)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Base learning rate (scaled by world_size, default: 1e-3)")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument(
        "--segment_length", type=int, default=200,
        help="Mel segment length in frames (default: 200 = ~3.2s)"
    )
    parser.add_argument(
        "--resume", default=None,
        help="Path to checkpoint for resuming training"
    )

    args = parser.parse_args()

    trainer = Prosody2VecTrainer(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        segment_length=args.segment_length,
        resume=args.resume,
    )
    trainer.train()


if __name__ == "__main__":
    main()
