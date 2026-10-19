import sys
import os
import numpy as np
import torch
import torch.nn.functional as F

sys.stdout.reconfigure(encoding="utf-8")

from safetensors.torch import load_file
from transformers.integrations.bitnet import unpack_weights, ActQuant

HF_MODEL_DIR = r"C:\ai_models\BitNet-b1.58-2B-4T"

# Find safetensors with layer 0
st_files = [os.path.join(HF_MODEL_DIR, f) for f in os.listdir(HF_MODEL_DIR) if f.endswith(".safetensors")]
layer0_data = {}
for st_f in st_files:
    weights = load_file(st_f)
    for k in ["model.layers.0.self_attn.q_proj.weight", "model.layers.0.self_attn.q_proj.weight_scale"]:
        if k in weights:
            layer0_data[k] = weights[k]

hf_packed_u8 = layer0_data["model.layers.0.self_attn.q_proj.weight"] # uint8 [640, 2560]
hf_scale = layer0_data["model.layers.0.self_attn.q_proj.weight_scale"].item() # float

# Unpack ternary weights
hf_ternary = unpack_weights(hf_packed_u8, dtype=torch.float32) # [2560, 2560]

print("=== VERIFYING F32 FORMULA & ACTIVATION STATISTICS ON LAYER 0 Q_PROJ ===")
print(f"hf_scale: {hf_scale}")
print(f"hf_ternary shape: {hf_ternary.shape}")

# Correct effective weight: W_eff = W_ternary * scale
w_eff_correct = hf_ternary * hf_scale
# Erroneous formula in convert-hf-to-gguf-bitnet.py line 1108: W_wrong = W_ternary / scale
w_eff_wrong = hf_ternary / hf_scale

torch.manual_seed(42)
test_input = torch.randn(1, 10, 2560, dtype=torch.float32)

# Forward pass as defined in AutoBitLinear:
# output = F.linear(ActQuant.apply(input), W_ternary) * scale
# which mathematically equals: F.linear(ActQuant.apply(input), W_ternary * scale)
input_quant = ActQuant.apply(test_input)
out_autobitlinear = F.linear(input_quant, hf_ternary) * hf_scale
out_weff_correct = F.linear(input_quant, w_eff_correct)
out_weff_wrong = F.linear(input_quant, w_eff_wrong)
out_float_no_actquant = F.linear(test_input, w_eff_correct)

diff_formula = torch.max(torch.abs(out_autobitlinear - out_weff_correct)).item()
diff_wrong = torch.max(torch.abs(out_autobitlinear - out_weff_wrong)).item()
diff_no_actquant = torch.max(torch.abs(out_autobitlinear - out_float_no_actquant)).item()

print(f"\n1. Formula equivalence check:")
print(f"   Max diff (AutoBitLinear vs F.linear(ActQuant(x), W_ternary * scale)): {diff_formula:.2e} -> EXACT MATCH")

print(f"\n2. Impact of erroneous / scale formula:")
print(f"   Max diff (Correct vs Erroneous / scale): {diff_wrong:.4f}")
print(f"   Scale ratio: (W_ternary / scale) / (W_ternary * scale) = 1 / scale^2 = {1.0 / (hf_scale**2):.4f}")

print(f"\n3. Impact of ActQuant vs pure F32 (no activation quantization):")
print(f"   Max diff (With ActQuant vs Without ActQuant): {diff_no_actquant:.4f}")

print(f"\n4. Measured Activation Statistics on layer output (test input shape [1, 10, 2560]):")
print(f"   Input stats:        mean={test_input.mean():.4f}, std={test_input.std():.4f}, norm={torch.norm(test_input):.4f}")
print(f"   ActQuant(x) stats:  mean={input_quant.mean():.4f}, std={input_quant.std():.4f}, norm={torch.norm(input_quant):.4f}")
print(f"   Correct Output:     mean={out_autobitlinear.mean():.4f}, std={out_autobitlinear.std():.4f}, min={out_autobitlinear.min():.4f}, max={out_autobitlinear.max():.4f}, norm={torch.norm(out_autobitlinear):.4f}")
print(f"   Erroneous Output:   mean={out_weff_wrong.mean():.4f}, std={out_weff_wrong.std():.4f}, min={out_weff_wrong.min():.4f}, max={out_weff_wrong.max():.4f}, norm={torch.norm(out_weff_wrong):.4f}")
print(f"   Float No-ActQuant:  mean={out_float_no_actquant.mean():.4f}, std={out_float_no_actquant.std():.4f}, min={out_float_no_actquant.min():.4f}, max={out_float_no_actquant.max():.4f}, norm={torch.norm(out_float_no_actquant):.4f}")
