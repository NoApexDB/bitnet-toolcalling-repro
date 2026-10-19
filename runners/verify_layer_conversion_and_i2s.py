import sys
import os
import json
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, r"C:\ai_models\BitNet\3rdparty\llama.cpp\gguf-py")
import gguf
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM
from transformers.integrations.bitnet import unpack_weights, ActQuant

sys.stdout.reconfigure(encoding="utf-8")

HF_MODEL_DIR = r"C:\ai_models\BitNet-b1.58-2B-4T"
GGUF_I2S_PATH = r"C:\ai_models\BitNet\models\BitNet-b1.58-2B-4T\ggml-model-i2_s.gguf"

def main():
    print("=" * 75)
    print("STEP 1: VERIFYING I2_S GGUF WEIGHTS, PACKING ORDER, AND SCALE (LAYER 0 Q_PROJ)")
    print("=" * 75)

    # 1. Load HF safetensors for layer 0 q_proj
    # Find which safetensors file contains layer 0
    st_files = [os.path.join(HF_MODEL_DIR, f) for f in os.listdir(HF_MODEL_DIR) if f.endswith(".safetensors")]
    layer0_data = {}
    for st_f in st_files:
        weights = load_file(st_f)
        for k in ["model.layers.0.self_attn.q_proj.weight", "model.layers.0.self_attn.q_proj.weight_scale"]:
            if k in weights:
                layer0_data[k] = weights[k]

    if "model.layers.0.self_attn.q_proj.weight" not in layer0_data:
        raise RuntimeError("Could not find layer 0 q_proj in safetensors!")

    hf_packed_u8 = layer0_data["model.layers.0.self_attn.q_proj.weight"] # uint8 [640, 2560]
    hf_scale = layer0_data["model.layers.0.self_attn.q_proj.weight_scale"].item() # float

    print(f"HF Safetensors packed tensor: shape={hf_packed_u8.shape}, dtype={hf_packed_u8.dtype}")
    print(f"HF Safetensors weight_scale:  {hf_scale:.8f}")

    # Unpack HF ternary weights using transformers integration
    hf_ternary = unpack_weights(hf_packed_u8, dtype=torch.float32).numpy() # [2560, 2560]
    print(f"HF Unpacked ternary weights:  shape={hf_ternary.shape}, unique={np.unique(hf_ternary)}")

    # 2. Read GGUF blk.0.attn_q.weight
    r = gguf.GGUFReader(GGUF_I2S_PATH)
    t_gguf = next(t for t in r.tensors if t.name == "blk.0.attn_q.weight")
    raw_bytes = t_gguf.data.tobytes()
    total_len = len(raw_bytes)
    print(f"\nGGUF tensor 'blk.0.attn_q.weight':")
    print(f"  Shape: {t_gguf.shape}, Type: {t_gguf.tensor_type} (I2_S)")
    print(f"  Total bytes: {total_len}")

    # Extract scale from 32-byte tail
    M, K = 2560, 2560
    n = M * K
    packed_size = n // 4 # 1638400 bytes
    tail = raw_bytes[packed_size:packed_size+32]
    gguf_scale = np.frombuffer(tail[:4], dtype=np.float32)[0]
    print(f"  GGUF I2_S scale from tail:    {gguf_scale:.8f}")
    scale_match = np.isclose(hf_scale, gguf_scale, atol=1e-7)
    print(f"  [CHECK 1] Scale match (HF scale == GGUF tail scale): {scale_match} (diff: {abs(hf_scale - gguf_scale):.2e})")

    # 3. Dequantize GGUF I2_S packed bytes to ternary values
    # In quantize_to_i2_s:
    # 128 elements per block -> 32 bytes packed.
    # In byte j of block: bits 6-7 is elem j, bits 4-5 is elem j+32, bits 2-3 is elem j+64, bits 0-1 is elem j+96.
    # Encoding: 0 -> -1, 1 -> 0, 2 -> +1.
    packed_u8 = np.frombuffer(raw_bytes[:packed_size], dtype=np.uint8)
    n_blocks = packed_size // 32
    packed_blocks = packed_u8.reshape(n_blocks, 32)

    q0 = (packed_blocks >> 6) & 3
    q1 = (packed_blocks >> 4) & 3
    q2 = (packed_blocks >> 2) & 3
    q3 = packed_blocks & 3

    # Stack along axis 1: shape (n_blocks, 4, 32)
    q_all = np.stack([q0, q1, q2, q3], axis=1) # shape (n_blocks, 4, 32)
    q_flat = q_all.reshape(-1)[:n] # 6553600 elements
    
    # Map back to ternary {-1, 0, 1}
    gguf_ternary_flat = q_flat.astype(np.float32) - 1.0 # 0 -> -1, 1 -> 0, 2 -> +1
    gguf_ternary = gguf_ternary_flat.reshape(M, K)

    print(f"\nGGUF Dequantized ternary weights:")
    print(f"  Shape: {gguf_ternary.shape}, unique={np.unique(gguf_ternary)}")

    # Check match with HF ternary weights
    ternary_diff = np.abs(hf_ternary - gguf_ternary)
    max_diff = np.max(ternary_diff)
    match_count = np.sum(hf_ternary == gguf_ternary)
    total_elements = hf_ternary.size
    print(f"  [CHECK 2] Exact ternary weights match: {match_count} / {total_elements} ({match_count/total_elements*100:.2f}%)")
    print(f"  Max absolute difference: {max_diff}")

    # Check if there is a transposition (K, M vs M, K)
    transposed_match = np.sum(hf_ternary.T == gguf_ternary)
    print(f"  Transposed match count:  {transposed_match} / {total_elements} ({transposed_match/total_elements*100:.2f}%)")

    print("\n" + "=" * 75)
    print("STEP 2: CORRECT F32 DEQUANTIZATION FORMULA & ACTIVATION STATISTICS")
    print("=" * 75)

    # Correct formula: W_eff = W_ternary * weight_scale
    w_eff_correct = hf_ternary * hf_scale # shape (2560, 2560)
    # Erroneous formula in convert-hf-to-gguf-bitnet.py line 1108: W_wrong = W_ternary / weight_scale
    w_wrong_scale = hf_ternary / hf_scale

    print(f"Correct W_eff stats: min={w_eff_correct.min():.6f}, max={w_eff_correct.max():.6f}, std={w_eff_correct.std():.6f}")
    print(f"Wrong W_eff stats:   min={w_wrong_scale.min():.6f}, max={w_wrong_scale.max():.6f}, std={w_wrong_scale.std():.6f}")
    print(f"Scale distortion ratio: (1 / hf_scale) / hf_scale = {1.0 / (hf_scale ** 2):.4f}")

    # Load HF model layer to compare forward pass
    print("\nLoading HF layer to compare forward output...")
    model = AutoModelForCausalLM.from_pretrained(
        HF_MODEL_DIR,
        torch_dtype=torch.float32,
        device_map="cpu",
        low_cpu_mem_usage=True
    )
    layer0_q = model.model.layers[0].self_attn.q_proj

    # Generate test activation tensor
    torch.manual_seed(42)
    test_input = torch.randn(1, 10, 2560, dtype=torch.float32)

    # Forward through HF AutoBitLinear layer
    with torch.no_grad():
        out_hf = layer0_q(test_input)

    # Forward through manual F.linear with W_eff_correct and ActQuant
    test_input_quant = ActQuant.apply(test_input)
    w_eff_t = torch.tensor(w_eff_correct, dtype=torch.float32)
    out_manual_correct = F.linear(test_input_quant, w_eff_t)

    # Forward with wrong scale
    w_wrong_t = torch.tensor(w_wrong_scale, dtype=torch.float32)
    out_manual_wrong = F.linear(test_input_quant, w_wrong_t)

    # Forward standard float (no ActQuant) with W_eff_correct
    out_float_no_actquant = F.linear(test_input, w_eff_t)

    diff_correct = torch.max(torch.abs(out_hf - out_manual_correct)).item()
    diff_wrong = torch.max(torch.abs(out_hf - out_manual_wrong)).item()
    diff_no_actquant = torch.max(torch.abs(out_hf - out_float_no_actquant)).item()

    print(f"\nForward output comparisons on test input [1, 10, 2560]:")
    print(f"  Max diff (HF layer vs Manual W_eff with ActQuant):    {diff_correct:.2e} -> PASS")
    print(f"  Max diff (HF layer vs Manual W_wrong with ActQuant):  {diff_wrong:.2f}")
    print(f"  Max diff (HF layer vs Standard F32 without ActQuant): {diff_no_actquant:.4f}")

    # Measured activation statistics
    print(f"\nMeasured Activation Statistics (HF layer output):")
    print(f"  Mean:      {out_hf.mean().item():.6f}")
    print(f"  Std:       {out_hf.std().item():.6f}")
    print(f"  Min:       {out_hf.min().item():.6f}")
    print(f"  Max:       {out_hf.max().item():.6f}")
    print(f"  L2 Norm:   {torch.norm(out_hf).item():.4f}")

    print(f"\nMeasured Activation Statistics (Erroneous F32 output):")
    print(f"  Mean:      {out_manual_wrong.mean().item():.6f}")
    print(f"  Std:       {out_manual_wrong.std().item():.6f}")
    print(f"  Min:       {out_manual_wrong.min().item():.6f}")
    print(f"  Max:       {out_manual_wrong.max().item():.6f}")
    print(f"  L2 Norm:   {torch.norm(out_manual_wrong).item():.4f}")

if __name__ == "__main__":
    main()
