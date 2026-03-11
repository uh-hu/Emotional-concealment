from __future__ import annotations

"""
Speech2Vec 统一管线
Speech2Vec Unified Pipeline

双通道分析管线：音频文件 → 192 维韵律向量 + 768 维语义向量。

韵律通道: MelSpectrogram → ECAPA-TDNN → 192-dim prosody vector
语义通道: MelSpectrogram → SpeechMapper → 768-dim semantic vector
"""

import torch
import numpy as np
import json
from pathlib import Path

from mel_spectrogram import MelSpectrogramExtractor
from prosody_encoder import ProsodyEncoder
from semantic_encoder import SemanticEncoder


class Speech2VecPipeline:
    """Speech2Vec 完整管线（韵律 + 语义双通道）。

    音频文件 → 梅尔频谱图 → ECAPA-TDNN → 192 维韵律向量
                           → SpeechMapper → 768 维语义向量

    Parameters
    ----------
    pretrained_dir : str, optional
        SpeechBrain 预训练模型的本地目录（韵律编码器）。
    prosody_checkpoint : str, optional
        韵律编码器权重路径（.pt 文件）。
    semantic_checkpoint : str, optional
        语义编码器权重路径（.pt 文件）。
    semantic_dim : int
        语义向量维度，默认 768。
    device : str, optional
        计算设备。None 则自动选择。
    """

    def __init__(
        self,
        pretrained_dir: str = None,
        prosody_checkpoint: str = None,
        semantic_checkpoint: str = None,
        semantic_dim: int = 768,
        device: str = None,
    ):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print("[Pipeline] Initializing mel-spectrogram extractor...")
        self.mel_extractor = MelSpectrogramExtractor()

        print("[Pipeline] Initializing prosody encoder (ECAPA-TDNN)...")
        self.prosody_encoder = ProsodyEncoder(
            pretrained_dir=pretrained_dir,
            checkpoint_path=prosody_checkpoint,
            device=self.device,
        )

        print("[Pipeline] Initializing semantic encoder (SpeechMapper)...")
        self.semantic_encoder = SemanticEncoder(
            checkpoint_path=semantic_checkpoint,
            device=self.device,
            semantic_dim=semantic_dim,
        )

        print(f"[Pipeline] Ready | Device: {self.device}")
        print(f"[Pipeline] Prosody dim: {self.prosody_encoder.get_embedding_dim()} "
              f"| Semantic dim: {self.semantic_encoder.get_embedding_dim()}")

    def process(self, audio_path: str | Path) -> dict:
        """处理单个音频文件，提取韵律向量和语义向量。

        Parameters
        ----------
        audio_path : str or Path
            音频文件路径。

        Returns
        -------
        result : dict
            - "prosody_vector": np.ndarray [192]
            - "semantic_vector": np.ndarray [768]
            - "mel_spectrogram": np.ndarray [80, T]
            - "metadata": dict
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Step 1: 提取梅尔频谱图
        mel_result = self.mel_extractor.from_file(audio_path)
        mel = mel_result["mel"]  # [1, 80, T]

        # Step 2: 韵律编码
        prosody_emb = self.prosody_encoder.encode(mel)  # [1, 192]

        # Step 3: 语义编码
        semantic_emb = self.semantic_encoder.encode(mel)  # [1, 768]

        # 转 numpy
        prosody_vector = prosody_emb.squeeze(0).cpu().numpy()   # [192]
        semantic_vector = semantic_emb.squeeze(0).cpu().numpy()  # [768]
        mel_np = mel.squeeze(0).cpu().numpy()                    # [80, T]

        return {
            "prosody_vector": prosody_vector,
            "semantic_vector": semantic_vector,
            "mel_spectrogram": mel_np,
            "metadata": {
                "audio_path": str(audio_path.resolve()),
                "duration_sec": mel_result["duration_sec"],
                "sample_rate": mel_result["sample_rate"],
                "num_frames": mel_result["num_frames"],
                "prosody_dim": prosody_vector.shape[0],
                "semantic_dim": semantic_vector.shape[0],
                "device": self.device,
            },
        }

    def process_batch(self, audio_paths: list) -> list:
        """批量处理多个音频文件。"""
        results = []
        for i, path in enumerate(audio_paths):
            print(f"  [{i+1}/{len(audio_paths)}] Processing: {Path(path).name}")
            result = self.process(path)
            results.append(result)
        return results


# 保持向后兼容
Prosody2VecPipeline = Speech2VecPipeline


# ─────────────────────────── CLI ───────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Speech2Vec Pipeline: Audio -> Prosody Vector + Semantic Vector"
    )
    parser.add_argument("--audio", required=True, help="Audio file path")
    parser.add_argument("--pretrained_dir", default=None,
                        help="SpeechBrain pretrained model directory")
    parser.add_argument("--prosody_checkpoint", default=None,
                        help="Prosody encoder checkpoint (.pt)")
    parser.add_argument("--semantic_checkpoint", default=None,
                        help="Semantic encoder checkpoint (.pt)")
    parser.add_argument("--semantic_dim", type=int, default=768,
                        help="Semantic vector dimension (default: 768)")
    parser.add_argument("--device", default=None, help="Device (cuda/cpu)")
    parser.add_argument("--output", default=None,
                        help="Save vectors to .npz file")
    parser.add_argument("--json", action="store_true",
                        help="Output metadata as JSON")

    # 向后兼容旧参数名
    parser.add_argument("--checkpoint", default=None,
                        help="(Deprecated) Alias for --prosody_checkpoint")

    args = parser.parse_args()

    # 向后兼容
    prosody_ckpt = args.prosody_checkpoint or args.checkpoint

    pipeline = Speech2VecPipeline(
        pretrained_dir=args.pretrained_dir,
        prosody_checkpoint=prosody_ckpt,
        semantic_checkpoint=args.semantic_checkpoint,
        semantic_dim=args.semantic_dim,
        device=args.device,
    )

    result = pipeline.process(args.audio)

    p_vec = result["prosody_vector"]
    s_vec = result["semantic_vector"]
    meta = result["metadata"]

    print()
    print("=" * 60)
    print("  Speech2Vec Result")
    print("=" * 60)
    print(f"  Audio:            {Path(args.audio).name}")
    print(f"  Duration:         {meta['duration_sec']}s")
    print(f"  Mel frames:       {meta['num_frames']}")
    print(f"  Device:           {meta['device']}")
    print("-" * 60)
    print(f"  Prosody vector:   dim={meta['prosody_dim']}  "
          f"L2={np.linalg.norm(p_vec):.4f}")
    print(f"    First 5 dims:   {p_vec[:5].tolist()}")
    print(f"  Semantic vector:  dim={meta['semantic_dim']}  "
          f"L2={np.linalg.norm(s_vec):.4f}")
    print(f"    First 5 dims:   {s_vec[:5].tolist()}")
    print("=" * 60)

    if args.output:
        np.savez(
            args.output,
            prosody_vector=p_vec,
            semantic_vector=s_vec,
        )
        print(f"\n  Vectors saved to: {args.output}")

    if args.json:
        json_output = {
            "prosody_vector": p_vec.tolist(),
            "semantic_vector": s_vec.tolist(),
            "metadata": meta,
        }
        print("\n" + json.dumps(json_output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
