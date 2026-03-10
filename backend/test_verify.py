"""Quick verification test for Prosody2Vec backend (pure PyTorch)."""
from __future__ import annotations
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("  Prosody2Vec Backend Verification")
print("=" * 60)

# ─── Test 1: Mel Spectrogram ───
print("\n[TEST 1] Mel Spectrogram Extractor")
try:
    from mel_spectrogram import MelSpectrogramExtractor
    import torch

    extractor = MelSpectrogramExtractor()

    # Synthetic signal
    wav = torch.randn(1, 16000)
    mel = extractor.extract(wav)
    assert mel.shape[1] == 80, f"Expected 80 mels, got {mel.shape[1]}"
    print(f"  Synthetic: shape={list(mel.shape)} PASS")

    # Real audio
    test_audio = "../test/03-10-2026 13.52.wav"
    if os.path.exists(test_audio):
        result = extractor.from_file(test_audio)
        m = result["mel"]
        assert m.shape[1] == 80
        print(f"  Real audio: shape={list(m.shape)}, "
              f"duration={result['duration_sec']}s, "
              f"frames={result['num_frames']} PASS")
    else:
        print(f"  SKIP: {test_audio} not found")

except Exception as e:
    print(f"  FAIL: {e}")
    import traceback; traceback.print_exc()

# ─── Test 2: ECAPA-TDNN Encoder ───
print("\n[TEST 2] ECAPA-TDNN Prosody Encoder (untrained)")
try:
    from prosody_encoder import ProsodyEncoder, ECAPA_TDNN
    import torch

    # Model architecture check
    model = ECAPA_TDNN()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {n_params:,} ({n_params/1e6:.1f}M)")

    # Forward pass
    mel_input = torch.randn(2, 80, 100)  # batch=2, 80 mels, 100 frames
    emb = model(mel_input)
    assert emb.shape == (2, 192), f"Expected (2, 192), got {emb.shape}"
    print(f"  Forward pass: input={list(mel_input.shape)} -> output={list(emb.shape)} PASS")

    # ProsodyEncoder wrapper
    encoder = ProsodyEncoder(device="cpu")
    mel_test = torch.randn(1, 80, 50)
    vec = encoder.encode(mel_test)
    assert vec.shape == (1, 192)
    print(f"  Encoder wrapper: {list(vec.shape)} PASS")

except Exception as e:
    print(f"  FAIL: {e}")
    import traceback; traceback.print_exc()

# ─── Test 3: Full Pipeline ───
print("\n[TEST 3] Full Pipeline (untrained model)")
try:
    from pipeline import Prosody2VecPipeline

    pipeline = Prosody2VecPipeline(device="cpu")

    test_audio = "../test/03-10-2026 13.52.wav"
    if os.path.exists(test_audio):
        result = pipeline.process(test_audio)
        vec = result["prosody_vector"]
        mel = result["mel_spectrogram"]
        meta = result["metadata"]

        assert vec.shape[0] == 192, f"Expected 192-dim, got {vec.shape[0]}"
        assert mel.shape[0] == 80, f"Expected 80 mels, got {mel.shape[0]}"
        print(f"  prosody_vector: {vec.shape} PASS")
        print(f"  mel_spectrogram: {mel.shape} PASS")
        print(f"  duration: {meta['duration_sec']}s")
    else:
        print(f"  SKIP: {test_audio} not found")

except Exception as e:
    print(f"  FAIL: {e}")
    import traceback; traceback.print_exc()

# ─── Test 4: Training imports ───
print("\n[TEST 4] Training script imports")
try:
    from train import MelDecoder, AudioDataset, Prosody2VecTrainer
    import torch

    # Test decoder
    decoder = MelDecoder()
    emb = torch.randn(2, 192)
    mel_out = decoder(emb, target_length=100)
    assert mel_out.shape == (2, 80, 100), f"Expected (2,80,100), got {mel_out.shape}"
    print(f"  MelDecoder: {list(mel_out.shape)} PASS")

    dec_params = sum(p.numel() for p in decoder.parameters())
    print(f"  Decoder params: {dec_params:,} ({dec_params/1e6:.1f}M)")

except Exception as e:
    print(f"  FAIL: {e}")
    import traceback; traceback.print_exc()

print("\n" + "=" * 60)
print("  All tests completed!")
print("=" * 60)
