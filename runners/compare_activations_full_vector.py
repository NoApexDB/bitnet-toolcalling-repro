import os
import sys
import json
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

PT_JSON = r"C:\ai_models\paper_artifacts\pytorch_layer_activations.json"
CPP_JSON = r"C:\ai_models\paper_artifacts\cpp_layer_activations.json"
OUT_REPORT = r"C:\ai_models\paper_artifacts\layer_activation_full_vector_comparison.json"

def cos_sim(u, v):
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    if norm_u == 0 or norm_v == 0:
        return 0.0
    return float(np.dot(u, v) / (norm_u * norm_v))

def compare_vectors(pt_vec, cpp_vec, name, desc):
    u = np.array(pt_vec, dtype=np.float64)
    v = np.array(cpp_vec, dtype=np.float64)

    assert len(u) == len(v), f"Dimension mismatch for {name}: {len(u)} vs {len(v)}"
    dim = len(u)

    pt_l2 = float(np.linalg.norm(u))
    cpp_l2 = float(np.linalg.norm(v))
    rel_l2_diff_pct = float(abs(pt_l2 - cpp_l2) / max(pt_l2, 1e-12) * 100.0)

    diff = np.abs(u - v)
    mae = float(np.mean(diff))
    max_ae = float(np.max(diff))
    rmse = float(np.sqrt(np.mean(diff ** 2)))

    # Full vector cosine similarity
    full_cos = cos_sim(u, v)

    # First 10 coordinates cosine similarity (for explicit historical comparison)
    cos_f10 = cos_sim(u[:10], v[:10])

    return {
        "name": name,
        "description": desc,
        "dim": dim,
        "pt_l2": pt_l2,
        "cpp_l2": cpp_l2,
        "rel_l2_diff_pct": rel_l2_diff_pct,
        "mae": mae,
        "max_ae": max_ae,
        "rmse": rmse,
        "full_vector_cosine": full_cos,
        "cos_f10": cos_f10
    }

def main():
    print("=== RIGOROUS FULL-VECTOR ACTIVATION COMPARISON (PyTorch vs bitnet.cpp with ReLU^2) ===")

    with open(PT_JSON, "r", encoding="utf-8") as f:
        pt_data = json.load(f)
    with open(CPP_JSON, "r", encoding="utf-8") as f:
        cpp_data = json.load(f)

    # Map C++ tensors by name (last occurrence)
    cpp_tensors = {}
    for t in cpp_data["tensors"]:
        cpp_tensors[t["name"]] = t

    pt_tensors = pt_data["tensors"]

    # Stages requested by user:
    # gate, up, ReLU, квадрата, умножения на up, ffn_sub_norm, down, logits
    stages_to_compare = [
        ("ffn_norm-0", "ffn_norm-0", "FFN Input RMSNorm (before Gate/Up)"),
        ("ffn_gate-0", "ffn_gate-0", "Gate projection (W_gate * x)"),
        ("ffn_up-0", "ffn_up-0", "Up projection (W_up * x)"),
        ("ffn_relu-0", "ffn_relu-0", "Activation ReLU (relu(gate))"),
        ("ffn_sqr(relu)-0", "ffn_sqr(relu)-0", "Activation Square (relu(gate)^2)"),
        ("ffn_gate_par-0", "ffn_sub_out-0", "Gated Multiply (relu(gate)^2 * up)"),
        ("ffn_sub_norm-0", "ffn_sub_norm-0", "FFN Sub-Norm (RMSNorm before Down)"),
        ("ffn_down-0", "ffn_down-0", "Down projection (W_down * sub_norm)"),
        ("l_out-0", "l_out-0", "Layer 0 Residual Output"),
    ]

    results = []

    print(f"\n{'Stage':<20} | {'Dim':<6} | {'PT L2':<10} | {'CPP L2':<10} | {'Rel Diff %':<10} | {'Max AE':<10} | {'Full Cosine':<12} | {'cos_f10':<10}")
    print("-" * 105)

    for pt_name, cpp_name, desc in stages_to_compare:
        if pt_name not in pt_tensors:
            print(f"Missing in PT: {pt_name}")
            continue
        if cpp_name not in cpp_tensors:
            print(f"Missing in CPP: {cpp_name}")
            continue

        pt_entry = pt_tensors[pt_name]
        cpp_entry = cpp_tensors[cpp_name]

        pt_vec = pt_entry.get("last_token_full", [])
        cpp_vec = cpp_entry.get("last_token_full", [])

        if not pt_vec or not cpp_vec:
            print(f"Missing full vector for {pt_name} / {cpp_name}")
            continue

        res = compare_vectors(pt_vec, cpp_vec, pt_name, desc)
        results.append(res)

        print(f"{pt_name:<20} | {res['dim']:<6} | {res['pt_l2']:<10.4f} | {res['cpp_l2']:<10.4f} | {res['rel_l2_diff_pct']:<9.4f}% | {res['max_ae']:<10.4f} | {res['full_vector_cosine']:<12.6f} | {res['cos_f10']:<10.6f}")

    # Compare Logits
    pt_logits = pt_data["logits_stats"]["full_logits"]
    cpp_logits = cpp_data["logits_stats"]["full_logits"]
    logits_res = compare_vectors(pt_logits, cpp_logits, "result_output (logits)", "Next-token prediction logits")
    results.append(logits_res)

    print(f"{'logits':<20} | {logits_res['dim']:<6} | {logits_res['pt_l2']:<10.4f} | {logits_res['cpp_l2']:<10.4f} | {logits_res['rel_l2_diff_pct']:<9.4f}% | {logits_res['max_ae']:<10.4f} | {logits_res['full_vector_cosine']:<12.6f} | {logits_res['cos_f10']:<10.6f}")

    # Save detailed report
    out_dict = {
        "summary": "Full vector numerical comparison between PyTorch (float32) and bitnet.cpp with LLM_FFN_RELU_SQR on 227 identical tokens",
        "stages": results,
        "pt_top5": pt_data["logits_stats"]["top10"][:5],
        "cpp_top5": cpp_data["logits_stats"]["top10"][:5]
    }

    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        json.dump(out_dict, f, indent=2, ensure_ascii=False)
    print(f"\nDetailed report saved to: {OUT_REPORT}")

if __name__ == "__main__":
    main()
