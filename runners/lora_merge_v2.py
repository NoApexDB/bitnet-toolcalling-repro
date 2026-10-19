# -*- coding: utf-8 -*-
"""
PHASE 2.3a (v2): Merge balanced agent LoRA into BitNet base and save a checkpoint in the
ORIGINAL BitNet HF format (packed uint8 ternary codes + weight_scale), so the stock
convert-hf-to-gguf-bitnet.py -> llama-quantize I2_S pipeline works unchanged.

Why custom save: transformers 5.17.0 save_pretrained() crashes reverting BitNetDeserialize
('reverse_op' AttributeError); and the merged float weights must be RE-TERNARIZED anyway
(BitNet b1.58 rule: scale = mean|W|, codes = round(W/scale) clipped to {-1,0,+1}) — this
re-ternarization is precisely the experimental question: does SFT survive it?

Round-trip safety: the packing layout is verified byte-exact against the original
checkpoint (pack(unpack(original)) == original) before anything is saved.
"""
import json
import os
import shutil
import sys
import time

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

sys.stdout.reconfigure(encoding="utf-8")

BASE = r"C:\ai_models\BitNet-b1.58-2B-4T"
ADAPTER = r"C:\ai_models\bitnet_agent_lora_balanced"
OUT_DIR = r"C:\ai_models\bitnet_merged_balanced"
MERGE_REPORT = r"C:\ai_models\paper_artifacts\lora_merge_report.json"


def unpack(packed_uint8):
    """Exactly the write_tensors unpack of convert-hf-to-gguf-bitnet.py (lines 800-807)."""
    origin_shape = packed_uint8.shape
    shift = torch.tensor([0, 2, 4, 6], dtype=torch.uint8).reshape((4, *(1 for _ in range(len(origin_shape)))))
    planes = packed_uint8.unsqueeze(0).expand((4, *origin_shape)) >> shift
    planes = planes & 3
    ternary = (planes.float() - 1).reshape((origin_shape[0] * 4, *origin_shape[1:]))
    return ternary  # {-1,0,1}


def pack(ternary):
    """Inverse of unpack(): ternary {-1,0,1} [4M,K] -> packed uint8 [M,K].
    Layout (verified by round-trip against the original checkpoint):
    logical quarter p (rows [p*M:(p+1)*M]) goes to bit-plane 2p of each byte."""
    codes = (ternary + 1).to(torch.uint8)  # {0,1,2}
    M_log = codes.shape[0]
    planes = codes.reshape((4, M_log // 4, *codes.shape[1:]))
    packed = planes[0] | (planes[1].to(torch.uint8) << 2) \
        | (planes[2].to(torch.uint8) << 4) | (planes[3].to(torch.uint8) << 6)
    return packed


def reternarize(W, scale=None):
    """Re-ternarize merged float weights back onto the ORIGINAL ternary lattice.

    Empirically established convention (verified byte-exact on the original checkpoint):
    transformers deserializes BitNet checkpoints as W = codes / weight_scale, so
    codes = clip(round(W * weight_scale), -1, 0, 1) reproduces the original codes
    exactly for unchanged weights; only the LoRA deltas can shift codes.
    (round(W / scale) is WRONG — that collapses to all zeros.)"""
    codes = torch.clamp(torch.round(W * scale), -1.0, 1.0)
    return codes, scale


def main():
    t0 = time.perf_counter()

    # ---- 0) verify pack/unpack round-trip on the ORIGINAL checkpoint ----
    orig = load_file(os.path.join(BASE, "model.safetensors"))
    scale_keys = [k[:-len(".weight_scale")] for k in orig if k.endswith(".weight_scale")]
    print(f"original checkpoint: {len(orig)} tensors, {len(scale_keys)} quantized (with weight_scale)")
    probe = "model.layers.0.mlp.down_proj"
    rt = pack(unpack(orig[probe + ".weight"]))
    assert torch.equal(rt, orig[probe + ".weight"]), "pack/unpack round-trip FAILED on original checkpoint!"
    # verify on ALL quantized tensors (not just the probe)
    for base_name in scale_keys:
        if not torch.equal(pack(unpack(orig[base_name + ".weight"])), orig[base_name + ".weight"]):
            raise AssertionError(f"round-trip failed for {base_name}")
    print(f"pack/unpack round-trip: BYTE-EXACT on all {len(scale_keys)} quantized tensors")

    # ---- 1) load base, attach adapter, merge ----
    print("Loading base BitNet FP32 on CPU...")
    base = AutoModelForCausalLM.from_pretrained(
        BASE, torch_dtype=torch.float32, device_map="cpu", low_cpu_mem_usage=True
    )
    model = PeftModel.from_pretrained(base, ADAPTER)
    t1 = time.perf_counter()
    merged = model.merge_and_unload()
    print(f"merge_and_unload in {time.perf_counter()-t1:.1f}s")
    sd = merged.state_dict()

    # map state_dict names -> checkpoint names (state_dict uses plain module names)
    # find, for each scale_key, the corresponding state_dict tensor
    missing, reterr_stats = [], []
    out_tensors = {}
    for k, v in orig.items():
        if k.endswith(".weight_scale"):
            continue
        base_name = k[:-len(".weight")] if k.endswith(".weight") else None
        if base_name is not None and base_name in scale_keys:
            # quantized tensor: re-ternarize merged float weights ONTO THE ORIGINAL LATTICE
            W = sd[k].to(torch.float32)
            s_orig = orig[base_name + ".weight_scale"].to(torch.float32)  # scalar or per-tensor scale
            codes, scale = reternarize(W, scale=s_orig)
            out_tensors[k] = pack(codes).contiguous()
            out_tensors[base_name + ".weight_scale"] = orig[base_name + ".weight_scale"]
            # weight-level error of re-ternarization (delta-induced only)
            Wq = codes / scale
            rel = (W - Wq).abs().mean().item() / max(W.abs().mean().item(), 1e-9)
            reterr_stats.append({"tensor": k, "rel_l1": round(rel, 5)})
        else:
            # non-quantized tensor: keep as is (bf16 like original)
            out_tensors[k] = sd[k].to(orig[k].dtype).contiguous() if k in sd else orig[k]
            if k not in sd:
                missing.append(k)

    if missing:
        print(f"[warn] {len(missing)} tensors not in merged state_dict, kept from original: {missing[:3]}")

    rels = np.array([s["rel_l1"] for s in reterr_stats])
    print(f"re-ternarization relative L1 error over {len(rels)} tensors: "
          f"mean {rels.mean()*100:.2f}%, max {rels.max()*100:.2f}%")

    # ---- 2) save ----
    if os.path.exists(OUT_DIR):
        shutil.rmtree(OUT_DIR)
    os.makedirs(OUT_DIR)
    save_file(out_tensors, os.path.join(OUT_DIR, "model.safetensors"))
    for f in ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
              "special_tokens_map.json"):
        src = os.path.join(BASE, f)
        if os.path.exists(src):
            shutil.copyfile(src, os.path.join(OUT_DIR, f))

    report = {
        "base": BASE,
        "adapter": ADAPTER,
        "adapter_config": json.load(open(os.path.join(ADAPTER, "adapter_config.json"), encoding="utf-8")),
        "out_dir": OUT_DIR,
        "roundtrip_verified": True,
        "quantized_tensors": len(reterr_stats),
        "reternarization_rel_l1_mean_pct": round(float(rels.mean()) * 100, 3),
        "reternarization_rel_l1_max_pct": round(float(rels.max()) * 100, 3),
        "reternarization_per_tensor": reterr_stats,
        "total_seconds": round(time.perf_counter() - t0, 1),
        "files": {f: os.path.getsize(os.path.join(OUT_DIR, f)) for f in os.listdir(OUT_DIR)},
    }
    with open(MERGE_REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps(report["files"], ensure_ascii=False))
    print(f"Report -> {MERGE_REPORT}")
    print("MERGE V2 DONE")


if __name__ == "__main__":
    main()
