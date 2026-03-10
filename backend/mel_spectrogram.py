from __future__ import annotations

"""
声学特征预处理模块 — 梅尔频谱图提取器
Acoustic Feature Preprocessing — Mel-Spectrogram Extractor

根据 Prosody2Vec 论文参数设置：
- STFT frequency bins (n_fft): 1024
- Window size: 64ms → 1024 samples @ 16kHz
- Stride (hop length): 16ms → 256 samples @ 16kHz
- Mel filter bank channels: 80
"""

import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np
from pathlib import Path


class MelSpectrogramExtractor:
    """从原始音频波形中提取 80 通道梅尔频谱图。

    Parameters
    ----------
    sample_rate : int
        目标采样率，默认 16000 Hz。
    n_fft : int
        STFT 的 FFT 大小（频率区间），默认 1024。
    win_length : int
        STFT 窗口长度（采样点数），默认 1024（64ms @ 16kHz）。
    hop_length : int
        STFT 步幅（采样点数），默认 256（16ms @ 16kHz）。
    n_mels : int
        梅尔滤波器通道数，默认 80。
    log_scale : bool
        是否对梅尔频谱图取 log，默认 True。
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 1024,
        win_length: int = 1024,
        hop_length: int = 256,
        n_mels: int = 80,
        log_scale: bool = True,
    ):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.log_scale = log_scale

        # 构建梅尔频谱图变换
        self.mel_transform = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            power=2.0,  # 幅度谱的平方 → 功率谱
        )

        # 重采样缓存（源采样率 → 重采样器）
        self._resamplers: dict[int, T.Resample] = {}

    def load_audio(self, path: str | Path) -> torch.Tensor:
        """加载音频文件，转换为单声道并重采样至目标采样率。

        Parameters
        ----------
        path : str or Path
            音频文件路径，支持 wav, mp3, flac, ogg 等格式。

        Returns
        -------
        waveform : torch.Tensor
            形状 [1, num_samples] 的单声道波形张量。
        """
        path = str(path)
        waveform, sr = torchaudio.load(path)

        # 多声道 → 单声道（取均值）
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # 重采样
        if sr != self.sample_rate:
            if sr not in self._resamplers:
                self._resamplers[sr] = T.Resample(
                    orig_freq=sr, new_freq=self.sample_rate
                )
            waveform = self._resamplers[sr](waveform)

        return waveform

    def extract(self, waveform: torch.Tensor) -> torch.Tensor:
        """从波形中提取梅尔频谱图。

        Parameters
        ----------
        waveform : torch.Tensor
            形状 [batch, num_samples] 或 [1, num_samples] 的波形张量。

        Returns
        -------
        mel : torch.Tensor
            形状 [batch, n_mels, T] 的梅尔频谱图张量。
            其中 T = ceil(num_samples / hop_length)。
        """
        # 确保至少 2D
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        mel = self.mel_transform(waveform)

        if self.log_scale:
            # log 压缩，添加小常数防止 log(0)
            mel = torch.log(mel + 1e-9)

        return mel

    def from_file(self, path: str | Path) -> dict:
        """一站式接口：加载音频 → 提取梅尔频谱图。

        Parameters
        ----------
        path : str or Path
            音频文件路径。

        Returns
        -------
        result : dict
            包含以下键：
            - "mel": torch.Tensor, 形状 [1, 80, T]
            - "waveform": torch.Tensor, 形状 [1, num_samples]
            - "sample_rate": int
            - "duration_sec": float, 音频时长（秒）
            - "num_frames": int, 梅尔频谱图帧数 T
        """
        waveform = self.load_audio(path)
        mel = self.extract(waveform)

        duration = waveform.shape[-1] / self.sample_rate

        return {
            "mel": mel,
            "waveform": waveform,
            "sample_rate": self.sample_rate,
            "duration_sec": round(duration, 3),
            "num_frames": mel.shape[-1],
        }


# ─────────────────────────── CLI 入口 ───────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="提取梅尔频谱图")
    parser.add_argument("--audio", required=True, help="音频文件路径")
    args = parser.parse_args()

    extractor = MelSpectrogramExtractor()
    result = extractor.from_file(args.audio)

    mel = result["mel"]
    print("=" * 50)
    print("梅尔频谱图提取结果")
    print("=" * 50)
    print(f"  音频时长:    {result['duration_sec']}s")
    print(f"  采样率:      {result['sample_rate']}Hz")
    print(f"  梅尔通道数:  {mel.shape[1]}")
    print(f"  帧数 (T):    {mel.shape[2]}")
    print(f"  张量形状:    {list(mel.shape)}")
    print(f"  值域范围:    [{mel.min().item():.4f}, {mel.max().item():.4f}]")
    print("=" * 50)
