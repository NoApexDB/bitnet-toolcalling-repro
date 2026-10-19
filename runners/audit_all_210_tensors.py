import os
import sys
import json
import time
import numpy as np
import torch

sys.path.insert(0, r"C:\ai_models\BitNet\3rdparty\llama.cpp\gguf-py")
import gguf
from safetensors import safe_open
from transformers.integrations.bitnet import unpack_weights

sys.stdout.reconfigure(encoding="utf-8")

SAFETENSORS_PATH = r"C:\ai_models\BitNet-b1.58-2B-4T\model.safetensors"
GGUF_I2S_PATH = r"C:\ai_models\BitNet\models\BitNet-b1.58-2B-4T\ggml-model-i2_s.gguf"
OUTPUT_JSON = r"C:\ai_models\paper_artifacts\all_210_tensors_weight_audit.json"

PROJ_MAP = [
    ("self_attn.q_proj", "attn_q"),
    ("self_attn.k_proj", "attn_k"),
    ("self_attn.v_proj", "attn_v"),
    ("self_attn.o_proj", "attn_output"),
    ("mlp.gate_proj", "ffn_gate"),
    ("mlp.up_proj", "ffn_up"),
    ("mlp.down_proj", "ffn_down"),
]

def main():
    print("=" * 80)
    print("AUDITING ALL 210 LINEAR TENSORS: HF SAFETENSORS VS GGUF I2_S")
    print("=" * 80)

    print("Opening GGUF file and building tensor lookup map...")
    reader = gguf.GGUFReader(GGUF_I2S_PATH)
    gguf_tensors = {t.name: t for t in reader.tensors}
    print(f"GGUF loaded with {len(gguf_tensors)} tensors total.")

    audit_results = []
    total_tensors = 0
    exact_match_tensors = 0
    scale_matches = 0
    max_scale_diff_global = 0.0
    max_scale_rel_diff_global = 0.0

    t0_start = time.perf_counter()

    with safe_open(SAFETENSORS_PATH, framework="pt", device="cpu") as st:
        st_keys = set(st.keys())

        for layer_idx in range(30):
            t0_layer = time.perf_counter()
            layer_mismatches = 0

            for hf_sub, gguf_sub in PROJ_MAP:
                total_tensors += 1
                hf_prefix = f"model.layers.{layer_idx}.{hf_sub}"
                hf_weight_key = f"{hf_prefix}.weight"
                hf_scale_key = f"{hf_prefix}.weight_scale"
                gguf_name = f"blk.{layer_idx}.{gguf_sub}.weight"

                if hf_weight_key not in st_keys:
                    raise KeyError(f"Missing {hf_weight_key} in safetensors!")
                if gguf_name not in gguf_tensors:
                    raise KeyError(f"Missing {gguf_name} in GGUF!")

                # 1. HF data
                hf_packed = st.get_tensor(hf_weight_key)
                hf_scale = st.get_tensor(hf_scale_key).item()
                hf_ternary = unpack_weights(hf_packed, dtype=torch.float32).numpy()
                M, K = hf_ternary.shape
                n = M * K

                # 2. GGUF data
                t_gguf = gguf_tensors[gguf_name]
                raw_bytes = t_gguf.data.tobytes()
                packed_size = n // 4
                tail = raw_bytes[packed_size:packed_size + 32]
                gguf_scale = np.frombuffer(tail[:4], dtype=np.float32)[0]

                # Dequantize I2_S packed bytes
                packed_u8 = np.frombuffer(raw_bytes[:packed_size], dtype=np.uint8)
                n_blocks = packed_size // 32
                packed_blocks = packed_u8.reshape(n_blocks, 32)

                q0 = (packed_blocks >> 6) & 3
                q1 = (packed_blocks >> 4) & 3
                q2 = (packed_blocks >> 2) & 3
                q3 = packed_blocks & 3
                q_all = np.stack([q0, q1, q2, q3], axis=1)
                q_flat = q_all.reshape(-1)[:n]
                gguf_ternary = (q_flat.astype(np.float32) - 1.0).reshape(M, K)

                # Comparisons
                matches = int(np.sum(hf_ternary == gguf_ternary))
                is_exact_match = (matches == n)
                scale_diff = abs(hf_scale - gguf_scale)
                scale_rel_diff = scale_diff / hf_scale if hf_scale > 0 else 0.0

                if is_exact_match:
                    exact_match_tensors += 1
                else:
                    layer_mismatches += 1

                max_scale_diff_global = max(max_scale_diff_global, scale_diff)
                max_scale_rel_diff_global = max(max_scale_rel_diff_global, scale_rel_diff)

                audit_results.append({
                    "layer": layer_idx,
                    "hf_key": hf_weight_key,
                    "gguf_name": gguf_name,
                    "shape": [M, K],
                    "total_elements": n,
                    "matching_elements": matches,
                    "match_pct": (matches / n) * 100.0,
                    "is_exact_ternary_match": is_exact_match,
                    "hf_scale": hf_scale,
                    "gguf_scale": float(gguf_scale),
                    "scale_diff": float(scale_diff),
                    "scale_rel_diff": float(scale_rel_diff)
                })

            elapsed_l = time.perf_counter() - t0_layer
            status = "ALL PASS (7/7)" if layer_mismatches == 0 else f"MISMATCHES: {layer_mismatches}"
            print(f"Layer {layer_idx:2d}/29 | {status} | {elapsed_l:.2f}s")

    t_total = time.perf_counter() - t0_start
    print("\n" + "=" * 80)
    print("AUDIT SUMMARY (210 LINEAR TENSORS):")
    print(f"  Total Linear Tensors:      {total_tensors}")
    print(f"  Exact Ternary Matches:     {exact_match_tensors} / {total_tensors} ({exact_match_tensors/total_tensors*100:.2f}%)")
    print(f"  Max Absolute Scale Diff:   {max_scale_diff_global:.2e}")
    print(f"  Max Relative Scale Diff:   {max_scale_rel_diff_global*100:.4f}%")
    print(f"  Total Time:                {t_total:.2f}s")
    print("=" * 80)

    # Save complete JSON
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump({
            "summary": {
                "total_tensors": total_tensors,
                "exact_ternary_matches": exact_match_tensors,
                "all_ternary_matched": (exact_match_tensors == total_tensors),
                "max_scale_diff": float(max_scale_diff_global),
                "max_scale_rel_diff_pct": float(max_scale_rel_diff_global * 100.0),
                "elapsed_sec": t_total
            },
            "tensors": audit_results
        }, f, indent=2, ensure_ascii=False)
    print(f"Saved full audit JSON to: {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
