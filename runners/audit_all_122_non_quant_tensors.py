import os
import sys
import json
import time
import numpy as np
import torch

sys.path.insert(0, r"C:\ai_models\BitNet\3rdparty\llama.cpp\gguf-py")
import gguf
from safetensors import safe_open

sys.stdout.reconfigure(encoding="utf-8")

SAFETENSORS_PATH = r"C:\ai_models\BitNet-b1.58-2B-4T\model.safetensors"
GGUF_I2S_PATH = r"C:\ai_models\BitNet\models\BitNet-b1.58-2B-4T\ggml-model-i2_s.gguf"
OUTPUT_JSON = r"C:\ai_models\paper_artifacts\non_quantized_122_tensors_audit.json"

NORM_MAP = [
    ("input_layernorm.weight", "attn_norm.weight"),
    ("self_attn.attn_sub_norm.weight", "attn_sub_norm.weight"),
    ("post_attention_layernorm.weight", "ffn_norm.weight"),
    ("mlp.ffn_sub_norm.weight", "ffn_sub_norm.weight"),
]

def main():
    print("=" * 80)
    print("FULL AUDIT OF ALL 122 NON-QUANTIZED TENSORS: HF SAFETENSORS VS GGUF")
    print("=" * 80)

    t0_start = time.perf_counter()
    reader = gguf.GGUFReader(GGUF_I2S_PATH)
    gguf_tensors = {t.name: t for t in reader.tensors}
    print(f"Loaded GGUF file with {len(gguf_tensors)} tensors.")

    with safe_open(SAFETENSORS_PATH, framework="pt", device="cpu") as st:
        st_keys = set(st.keys())
        print(f"Loaded Safetensors with {len(st_keys)} keys.")

        audit_records = []

        # 1. Auditing Embedding Tensor
        print("\n--- 1. Auditing token_embd.weight (1/122) ---")
        hf_emb_key = "model.embed_tokens.weight"
        gguf_emb_key = "token_embd.weight"

        if hf_emb_key not in st_keys or gguf_emb_key not in gguf_tensors:
            raise KeyError(f"Missing embedding tensor: {hf_emb_key} or {gguf_emb_key}")

        hf_emb_raw = st.get_tensor(hf_emb_key) # torch.bfloat16
        t_gguf_emb = gguf_tensors[gguf_emb_key]
        gguf_emb_arr = t_gguf_emb.data # float16 numpy array

        # Compare with HF rounded to float16
        hf_emb_f16 = hf_emb_raw.to(torch.float16).numpy()
        hf_emb_f32 = hf_emb_raw.to(torch.float32).numpy()
        gguf_emb_f32 = gguf_emb_arr.astype(np.float32)

        emb_total_elems = int(hf_emb_raw.numel())
        emb_exact_f16_matches = int(np.sum(hf_emb_f16 == gguf_emb_arr))
        emb_f16_diff = np.abs(hf_emb_f16.astype(np.float32) - gguf_emb_f32)
        emb_max_abs_diff_f16 = float(np.max(emb_f16_diff))
        emb_mean_abs_diff_f16 = float(np.mean(emb_f16_diff))

        # Diff against original bfloat16 (converted to f32)
        emb_raw_diff = np.abs(hf_emb_f32 - gguf_emb_f32)
        emb_max_abs_diff_raw = float(np.max(emb_raw_diff))
        emb_mean_abs_diff_raw = float(np.mean(emb_raw_diff))

        emb_record = {
            "index": 0,
            "category": "embedding",
            "hf_name": hf_emb_key,
            "gguf_name": gguf_emb_key,
            "hf_shape": list(hf_emb_raw.shape),
            "gguf_shape": list(t_gguf_emb.shape),
            "hf_dtype": str(hf_emb_raw.dtype),
            "gguf_dtype": t_gguf_emb.tensor_type.name,
            "total_elements": emb_total_elems,
            "exact_matches_vs_f16_rounded": emb_exact_f16_matches,
            "exact_matches_pct_vs_f16": float(emb_exact_f16_matches / emb_total_elems * 100.0),
            "max_abs_diff_vs_f16": emb_max_abs_diff_f16,
            "mean_abs_diff_vs_f16": emb_mean_abs_diff_f16,
            "max_abs_diff_vs_bfloat16_raw": emb_max_abs_diff_raw,
            "mean_abs_diff_vs_bfloat16_raw": emb_mean_abs_diff_raw,
        }
        audit_records.append(emb_record)

        print(f"Embedding elements: {emb_total_elems:,}")
        print(f"  Exact matches vs F16 rounded: {emb_exact_f16_matches:,} / {emb_total_elems:,} ({emb_record['exact_matches_pct_vs_f16']:.4f}%)")
        print(f"  Max absolute diff vs F16 rounded: {emb_max_abs_diff_f16:.2e}")
        print(f"  Mean absolute diff vs F16 rounded: {emb_mean_abs_diff_f16:.2e}")
        print(f"  Max absolute diff vs bfloat16 raw: {emb_max_abs_diff_raw:.2e}")
        print(f"  Mean absolute diff vs bfloat16 raw: {emb_mean_abs_diff_raw:.2e}")

        # 2. Auditing 120 Layer Norm Tensors (30 layers * 4 norms)
        print("\n--- 2. Auditing 120 Layer Normalization Tensors ---")
        norm_idx = 1
        max_norm_abs_diff_global = 0.0
        max_norm_rel_diff_global = 0.0

        for layer_idx in range(30):
            for hf_sub, gguf_sub in NORM_MAP:
                hf_name = f"model.layers.{layer_idx}.{hf_sub}"
                gguf_name = f"blk.{layer_idx}.{gguf_sub}"

                if hf_name not in st_keys:
                    raise KeyError(f"Missing {hf_name} in safetensors!")
                if gguf_name not in gguf_tensors:
                    raise KeyError(f"Missing {gguf_name} in GGUF!")

                hf_t = st.get_tensor(hf_name).float().numpy()
                t_gguf = gguf_tensors[gguf_name]
                gguf_t = t_gguf.data.astype(np.float32)

                n_elems = hf_t.size
                abs_diff = np.abs(hf_t - gguf_t)
                max_diff = float(np.max(abs_diff))
                mean_diff = float(np.mean(abs_diff))
                denom = np.maximum(np.abs(hf_t), 1e-9)
                rel_diff = float(np.max(abs_diff / denom))
                exact_matches = int(np.sum(hf_t == gguf_t))

                max_norm_abs_diff_global = max(max_norm_abs_diff_global, max_diff)
                max_norm_rel_diff_global = max(max_norm_rel_diff_global, rel_diff)

                record = {
                    "index": norm_idx,
                    "category": "layer_norm",
                    "layer": layer_idx,
                    "hf_name": hf_name,
                    "gguf_name": gguf_name,
                    "hf_shape": list(hf_t.shape),
                    "gguf_shape": list(t_gguf.shape),
                    "hf_dtype": "torch.bfloat16",
                    "gguf_dtype": t_gguf.tensor_type.name,
                    "total_elements": n_elems,
                    "exact_matches": exact_matches,
                    "exact_matches_pct": float(exact_matches / n_elems * 100.0),
                    "max_abs_diff": max_diff,
                    "mean_abs_diff": mean_diff,
                    "max_rel_diff": rel_diff
                }
                audit_records.append(record)
                norm_idx += 1

            if layer_idx % 5 == 0 or layer_idx == 29:
                print(f"  Layer {layer_idx:2d}/29 norms verified (max abs diff so far: {max_norm_abs_diff_global:.2e})")

        # 3. Auditing Final Output Norm Tensor
        print("\n--- 3. Auditing Final output_norm.weight (122/122) ---")
        hf_norm_key = "model.norm.weight"
        gguf_norm_key = "output_norm.weight"

        hf_out_t = st.get_tensor(hf_norm_key).float().numpy()
        t_gguf_out = gguf_tensors[gguf_norm_key]
        gguf_out_t = t_gguf_out.data.astype(np.float32)

        out_n_elems = hf_out_t.size
        out_abs_diff = np.abs(hf_out_t - gguf_out_t)
        out_max_diff = float(np.max(out_abs_diff))
        out_mean_diff = float(np.mean(out_abs_diff))
        out_denom = np.maximum(np.abs(hf_out_t), 1e-9)
        out_rel_diff = float(np.max(out_abs_diff / out_denom))
        out_exact = int(np.sum(hf_out_t == gguf_out_t))

        out_record = {
            "index": 121,
            "category": "output_norm",
            "layer": None,
            "hf_name": hf_norm_key,
            "gguf_name": gguf_norm_key,
            "hf_shape": list(hf_out_t.shape),
            "gguf_shape": list(t_gguf_out.shape),
            "hf_dtype": "torch.bfloat16",
            "gguf_dtype": t_gguf_out.tensor_type.name,
            "total_elements": out_n_elems,
            "exact_matches": out_exact,
            "exact_matches_pct": float(out_exact / out_n_elems * 100.0),
            "max_abs_diff": out_max_diff,
            "mean_abs_diff": out_mean_diff,
            "max_rel_diff": out_rel_diff
        }
        audit_records.append(out_record)
        print(f"Final output norm verified: max abs diff {out_max_diff:.2e}, rel diff {out_rel_diff:.2e}")

    t_total = time.perf_counter() - t0_start
    print("\n" + "=" * 80)
    print("AUDIT SUMMARY (122 NON-QUANTIZED TENSORS):")
    print(f"  Total Non-Quantized Tensors:   {len(audit_records)}")
    print(f"  Embedding max diff vs F16:     {emb_max_abs_diff_f16:.2e} (mean: {emb_mean_abs_diff_f16:.2e})")
    print(f"  Embedding max diff vs BF16:    {emb_max_abs_diff_raw:.2e} (mean: {emb_mean_abs_diff_raw:.2e})")
    print(f"  RMSNorm max abs diff:          {max_norm_abs_diff_global:.2e}")
    print(f"  RMSNorm max rel diff:          {max_norm_rel_diff_global*100:.4f}%")
    print(f"  Total Time:                    {t_total:.2f}s")
    print("=" * 80)

    # Save to JSON
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    summary_data = {
        "total_tensors": len(audit_records),
        "embedding": emb_record,
        "max_norm_abs_diff": float(max(max_norm_abs_diff_global, out_max_diff)),
        "max_norm_rel_diff_pct": float(max(max_norm_rel_diff_global, out_rel_diff) * 100.0),
        "elapsed_sec": t_total
    }
    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer, np.unsignedinteger)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump({
            "summary": summary_data,
            "tensors": audit_records
        }, f, indent=2, ensure_ascii=False, cls=NpEncoder)
    print(f"Saved JSON artifact to: {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
