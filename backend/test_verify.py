"""Quick verification test for Speech2Vec backend (Prosody + Semantic)."""
from __future__ import annotations
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("  Speech2Vec Backend Verification")
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

# ─── Test 2: ECAPA-TDNN Prosody Encoder ───
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

# ─── Test 3: SpeechMapper Semantic Encoder ───
print("\n[TEST 3] SpeechMapper Semantic Encoder (untrained)")
try:
    from semantic_encoder import SemanticEncoder, SpeechMapper
    import torch

    # Model architecture check
    model = SpeechMapper()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {n_params:,} ({n_params/1e6:.1f}M)")

    # Forward pass
    mel_input = torch.randn(2, 80, 100)  # batch=2, 80 mels, 100 frames
    emb = model(mel_input)
    assert emb.shape == (2, 768), f"Expected (2, 768), got {emb.shape}"
    print(f"  Forward pass: input={list(mel_input.shape)} -> output={list(emb.shape)} PASS")

    # Test with different input lengths
    mel_short = torch.randn(1, 80, 30)
    emb_short = model(mel_short)
    assert emb_short.shape == (1, 768), f"Expected (1, 768), got {emb_short.shape}"
    print(f"  Variable length: input={list(mel_short.shape)} -> output={list(emb_short.shape)} PASS")

    mel_long = torch.randn(1, 80, 500)
    emb_long = model(mel_long)
    assert emb_long.shape == (1, 768), f"Expected (1, 768), got {emb_long.shape}"
    print(f"  Long sequence:   input={list(mel_long.shape)} -> output={list(emb_long.shape)} PASS")

    # SemanticEncoder wrapper
    encoder = SemanticEncoder(device="cpu")
    mel_test = torch.randn(1, 80, 50)
    vec = encoder.encode(mel_test)
    assert vec.shape == (1, 768)
    print(f"  Encoder wrapper: {list(vec.shape)} PASS")

except Exception as e:
    print(f"  FAIL: {e}")
    import traceback; traceback.print_exc()

# ─── Test 4: Full Pipeline ───
print("\n[TEST 4] Full Pipeline (untrained models)")
try:
    from pipeline import Speech2VecPipeline

    pipeline = Speech2VecPipeline(device="cpu")

    test_audio = "../test/03-10-2026 13.52.wav"
    if os.path.exists(test_audio):
        result = pipeline.process(test_audio)
        p_vec = result["prosody_vector"]
        s_vec = result["semantic_vector"]
        mel = result["mel_spectrogram"]
        meta = result["metadata"]

        assert p_vec.shape[0] == 192, f"Expected 192-dim prosody, got {p_vec.shape[0]}"
        assert s_vec.shape[0] == 768, f"Expected 768-dim semantic, got {s_vec.shape[0]}"
        assert mel.shape[0] == 80, f"Expected 80 mels, got {mel.shape[0]}"
        print(f"  prosody_vector:  {p_vec.shape} PASS")
        print(f"  semantic_vector: {s_vec.shape} PASS")
        print(f"  mel_spectrogram: {mel.shape} PASS")
        print(f"  duration: {meta['duration_sec']}s")
    else:
        print(f"  SKIP: {test_audio} not found")

except Exception as e:
    print(f"  FAIL: {e}")
    import traceback; traceback.print_exc()

# ─── Test 5: Training imports ───
print("\n[TEST 5] Training script imports")
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

# ─── Test 6: Backward Compatibility ───
print("\n[TEST 6] Backward Compatibility")
try:
    from pipeline import Prosody2VecPipeline, Speech2VecPipeline
    assert Prosody2VecPipeline is Speech2VecPipeline
    print(f"  Prosody2VecPipeline alias: PASS")
except Exception as e:
    print(f"  FAIL: {e}")
    import traceback; traceback.print_exc()

print("\n" + "=" * 60)
print("  All tests completed!")
print("=" * 60)
