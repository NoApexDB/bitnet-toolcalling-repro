import sys
import torch
import safetensors.torch
from transformers import AutoModelForCausalLM
import transformers.integrations.bitnet as bn

# Load HF model layer
print("Loading layer 0 down_proj from HF model...")
model = AutoModelForCausalLM.from_pretrained(
    r"C:\ai_models\BitNet-b1.58-2B-4T",
    torch_dtype=torch.float32,
    device_map="cpu",
    low_cpu_mem_usage=True
)
hf_layer = model.model.layers[0].mlp.down_proj
hf_weight = hf_layer.weight.data  # Unpacked ternary in AutoBitLinear
hf_scale = hf_layer.weight_scale.data.item()

print(f"HF weight shape: {hf_weight.shape}, dtype: {hf_weight.dtype}")
print(f"HF weight unique values: {torch.unique(hf_weight).tolist()}")
print(f"HF weight_scale: {hf_scale}")

# Effective weight as used in AutoBitLinear forward pass:
# forward computes: F.linear(ActQuant(x), weight) * weight_scale
# which is equivalent to linear with effective_weight = weight * weight_scale
hf_effective_weight = hf_weight * hf_scale
print(f"HF effective weight unique values: {torch.unique(hf_effective_weight).tolist()}")

# Load raw safetensors tensor
st = safetensors.torch.load_file(r"C:\ai_models\BitNet-b1.58-2B-4T\model.safetensors")
raw_packed = st["model.layers.0.mlp.down_proj.weight"]
raw_scale = st["model.layers.0.mlp.down_proj.weight_scale"].item()

# Unpack raw packed uint8 using transformers unpack_weights
unpacked_st = bn.unpack_weights(raw_packed, dtype=torch.float32)
print(f"Unpacked safetensors shape: {unpacked_st.shape}")
print(f"Unpacked safetensors unique values: {torch.unique(unpacked_st).tolist()}")

# Unit test 1: Does unpacked safetensors match HF layer weight exactly?
weight_diff = (hf_weight - unpacked_st).abs().max().item()
print(f"\n[UNIT TEST 1] Unpacked safetensors == HF layer ternary weight: diff={weight_diff} -> {'PASS' if weight_diff == 0 else 'FAIL'}")

# Unit test 2: Check converter formula:
# Converter line 1108 did: data_torch = data_torch / scale_map
# AutoBitLinear does: output = output * weight_scale -> W_eff = W * scale
converter_formula_val = (unpacked_st / raw_scale)
correct_formula_val = (unpacked_st * raw_scale)

print(f"\n[UNIT TEST 2] Effective weight calculation:")
print(f"  Correct effective weight (W * scale) range: [{correct_formula_val.min():.4f}, {correct_formula_val.max():.4f}]")
print(f"  Converter formula (W / scale) range:         [{converter_formula_val.min():.4f}, {converter_formula_val.max():.4f}]")
print(f"  Ratio (Correct / Converter) = scale^2 = {raw_scale**2:.4f}")

# Check with a test activation vector
x = torch.randn(1, 10, hf_layer.in_features)
with torch.no_grad():
    y_autobitlinear = hf_layer(x)
    # Manual forward using effective weight:
    x_quant = bn.ActQuant.apply(x)
    y_manual_correct = torch.nn.functional.linear(x_quant, correct_formula_val)
    y_manual_converter = torch.nn.functional.linear(x_quant, converter_formula_val)

diff_correct = (y_autobitlinear - y_manual_correct).abs().max().item()
diff_converter = (y_autobitlinear - y_manual_converter).abs().max().item()
print(f"\n[UNIT TEST 3] Forward pass on test activation:")
print(f"  AutoBitLinear vs Manual (W * scale): max diff = {diff_correct:.6e} -> {'EXACT MATCH' if diff_correct < 1e-4 else 'MISMATCH'}")
print(f"  AutoBitLinear vs Converter (W / scale): max diff = {diff_converter:.4f} -> Severe divergence")
