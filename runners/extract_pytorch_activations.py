import os
import sys
import json
import math
import numpy as np
import torch
from transformers import AutoModelForCausalLM

sys.stdout.reconfigure(encoding="utf-8")

TOKENS_JSON = r"C:\ai_models\paper_artifacts\diagnostic_input_tokens.json"
MODEL_DIR = r"C:\ai_models\BitNet-b1.58-2B-4T"
OUTPUT_JSON = r"C:\ai_models\paper_artifacts\pytorch_layer_activations.json"

def compute_stats(t_tensor, save_full=False):
    # t_tensor: torch.Tensor [batch, seq, hidden] or [seq, hidden]
    arr = t_tensor.detach().float().cpu().numpy()
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0] # [seq, hidden]

    n_elems = arr.size
    g_l2 = float(np.linalg.norm(arr))
    g_mean = float(np.mean(arr))
    g_std = float(np.std(arr))
    g_min = float(np.min(arr))
    g_max = float(np.max(arr))

    # Last token
    last_tok = arr[-1]
    tok_l2 = float(np.linalg.norm(last_tok))
    tok_mean = float(np.mean(last_tok))
    tok_std = float(np.std(last_tok))
    tok_min = float(np.min(last_tok))
    tok_max = float(np.max(last_tok))
    tok_first10 = [float(x) for x in last_tok[:10]]

    res = {
        "shape": list(arr.shape),
        "global_l2": g_l2,
        "global_mean": g_mean,
        "global_std": g_std,
        "global_min": g_min,
        "global_max": g_max,
        "last_tok_l2": tok_l2,
        "last_tok_mean": tok_mean,
        "last_tok_std": tok_std,
        "last_tok_min": tok_min,
        "last_tok_max": tok_max,
        "last_tok_first10": tok_first10
    }
    if save_full:
        res["last_token_full"] = [float(x) for x in last_tok]
    return res

def main():
    print("=== EXTRACTING PYTORCH INTERMEDIATE ACTIVATIONS ===")
    with open(TOKENS_JSON, "r", encoding="utf-8") as f:
        tok_data = json.load(f)
    tokens = tok_data["tokens"]
    n_tokens = len(tokens)
    print(f"Loaded {n_tokens} input tokens.")

    input_ids = torch.tensor([tokens], dtype=torch.long)

    print("Loading PyTorch BitNet model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        dtype=torch.float32,
        device_map="cpu"
    )
    model.eval()
    print("Model loaded successfully.")

    captured = {}

    def get_hook(name, save_full=False):
        def hook(module, inp, out):
            val = out
            if isinstance(val, tuple):
                val = val[0]
            captured[name] = compute_stats(val, save_full=save_full)
        return hook

    # Register hooks
    # 1. Embedding
    model.model.embed_tokens.register_forward_hook(get_hook("token_embd", save_full=False))

    # 2. Layer 0 custom forward to extract gate, up, relu, sqr, gate_par, sub_norm, down
    layer0_mlp = model.model.layers[0].mlp
    def layer0_mlp_forward(x):
        gate = layer0_mlp.gate_proj(x)
        up = layer0_mlp.up_proj(x)
        relu_val = torch.relu(gate)
        sqr_val = relu_val ** 2
        mul_val = sqr_val * up
        sub_norm_val = layer0_mlp.ffn_sub_norm(mul_val)
        down = layer0_mlp.down_proj(sub_norm_val)

        captured["ffn_gate-0"] = compute_stats(gate, save_full=True)
        captured["ffn_up-0"] = compute_stats(up, save_full=True)
        captured["ffn_relu-0"] = compute_stats(relu_val, save_full=True)
        captured["ffn_sqr(relu)-0"] = compute_stats(sqr_val, save_full=True)
        captured["ffn_gate_par-0"] = compute_stats(mul_val, save_full=True)
        captured["ffn_sub_norm-0"] = compute_stats(sub_norm_val, save_full=True)
        captured["ffn_down-0"] = compute_stats(down, save_full=True)
        return down
    layer0_mlp.forward = layer0_mlp_forward

    # Attention & layer hooks for layer 0
    layer0 = model.model.layers[0]
    layer0.input_layernorm.register_forward_hook(get_hook("attn_norm-0", save_full=True))
    layer0.self_attn.attn_sub_norm.register_forward_hook(get_hook("attn_sub_norm-0", save_full=True))
    layer0.self_attn.o_proj.register_forward_hook(get_hook("attn_out-0", save_full=True))
    layer0.post_attention_layernorm.register_forward_hook(get_hook("ffn_norm-0", save_full=True))
    layer0.register_forward_hook(get_hook("l_out-0", save_full=True))

    # Remaining layers 1..29
    for i in range(1, 30):
        layer = model.model.layers[i]
        layer.input_layernorm.register_forward_hook(get_hook(f"attn_norm-{i}"))
        layer.self_attn.attn_sub_norm.register_forward_hook(get_hook(f"attn_sub_norm-{i}"))
        layer.self_attn.o_proj.register_forward_hook(get_hook(f"attn_out-{i}"))
        layer.post_attention_layernorm.register_forward_hook(get_hook(f"ffn_norm-{i}"))
        layer.mlp.ffn_sub_norm.register_forward_hook(get_hook(f"ffn_sub_norm-{i}"))
        layer.mlp.down_proj.register_forward_hook(get_hook(f"ffn_down-{i}"))
        layer.register_forward_hook(get_hook(f"l_out-{i}"))

    # 3. Final norm & output
    model.model.norm.register_forward_hook(get_hook("result_norm", save_full=True))
    model.lm_head.register_forward_hook(get_hook("result_output", save_full=True))

    print("Executing forward pass...")
    with torch.no_grad():
        outputs = model(input_ids)
    print(f"Forward pass completed! Captured {len(captured)} tensor stages.")

    # Logits stats for last token
    logits_last = outputs.logits[0, -1].float()
    logits_arr = logits_last.cpu().numpy()
    n_vocab = len(logits_arr)

    l2 = float(np.linalg.norm(logits_arr))
    mean = float(np.mean(logits_arr))
    std = float(np.std(logits_arr))
    min_v = float(np.min(logits_arr))
    max_v = float(np.max(logits_arr))

    # Softmax probabilities
    probs = torch.softmax(logits_last, dim=-1).cpu().numpy()
    top_indices = np.argsort(logits_arr)[::-1][:10]

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)

    print("\n--- PyTorch Top 5 Next Token Predictions ---")
    top10_list = []
    for rank, idx in enumerate(top_indices[:10]):
        piece = tok.decode([int(idx)])
        prob = float(probs[idx])
        logit = float(logits_arr[idx])
        top10_list.append({
            "rank": rank + 1,
            "id": int(idx),
            "logit": logit,
            "prob": prob,
            "piece": piece
        })
        if rank < 5:
            print(f"  Rank {rank+1}: ID {idx:6d} | logit: {logit:8.4f} | prob: {prob*100.0:6.2f}% | piece: {repr(piece)}")

    result = {
        "n_tokens": n_tokens,
        "total_tensors_captured": len(captured),
        "logits_stats": {
            "l2": l2,
            "mean": mean,
            "std": std,
            "min": min_v,
            "max": max_v,
            "full_logits": [float(x) for x in logits_arr],
            "top10": top10_list
        },
        "tensors": captured
    }

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\nSaved PyTorch activations to: {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
