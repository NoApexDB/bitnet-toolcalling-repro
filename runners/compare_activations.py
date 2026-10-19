import json
import math
import numpy as np

CPP_JSON = r"C:\ai_models\paper_artifacts\cpp_layer_activations.json"
PT_JSON = r"C:\ai_models\paper_artifacts\pytorch_layer_activations.json"
OUT_REPORT = r"C:\ai_models\paper_artifacts\layer_activation_discrepancy_report.json"

STAGES = [
    ("attn_norm", "Input RMSNorm (before Self-Attn)"),
    ("attn_sub_norm", "Attn Sub-Norm (after Attn, before O_proj)"),
    ("attn_out", "Self-Attn Output (O_proj)"),
    ("ffn_norm", "FFN Norm (before MLP)"),
    ("ffn_sub_norm", "FFN Sub-Norm (before Down_proj)"),
    ("ffn_down", "MLP Down Output"),
    ("l_out", "Layer Residual Output (Final layer output)")
]

def main():
    with open(CPP_JSON, "r", encoding="utf-8") as f:
        d_cpp = json.load(f)
    with open(PT_JSON, "r", encoding="utf-8") as f:
        d_pt = json.load(f)

    # Index CPP tensors by name (last occurrence if multiple)
    cpp_map = {}
    for t in d_cpp["tensors"]:
        cpp_map[t["name"]] = t

    pt_map = d_pt["tensors"]

    print("=" * 105)
    print("STEP-BY-STEP COMPARISON: PYTORCH VS BITNET.CPP (CASE 01, 227 IDENTICAL INPUT TOKENS)")
    print("=" * 105)
    header = f"{'Layer / Stage':<30} | {'PT L2 Norm':<12} | {'CPP L2 Norm':<12} | {'Rel L2 Diff (%)':<16} | {'Max First10 Diff':<16} | Status"
    print(header)
    print("-" * 105)

    comparison_results = []
    first_divergence = None

    for layer in range(30):
        for stage_prefix, stage_desc in STAGES:
            key = f"{stage_prefix}-{layer}"
            if key not in pt_map or key not in cpp_map:
                continue

            pt_t = pt_map[key]
            cpp_t = cpp_map[key]

            pt_l2 = pt_t["last_tok_l2"]
            cpp_l2 = cpp_t["last_tok_l2"]

            # Relative difference of L2 norm
            l2_denom = max(pt_l2, 1e-9)
            rel_l2_diff = abs(pt_l2 - cpp_l2) / l2_denom * 100.0

            # First 10 elements diff
            pt_f10 = np.array(pt_t["last_tok_first10"])
            cpp_f10 = np.array(cpp_t["last_tok_first10"])
            max_f10_diff = float(np.max(np.abs(pt_f10 - cpp_f10)))
            cos_f10 = float(np.dot(pt_f10, cpp_f10) / (np.linalg.norm(pt_f10) * np.linalg.norm(cpp_f10) + 1e-9))

            status = "MATCH" if rel_l2_diff < 1.0 else ("DRIFT" if rel_l2_diff < 10.0 else "DIVERGED")

            if first_divergence is None and rel_l2_diff >= 5.0:
                first_divergence = {
                    "layer": layer,
                    "stage": stage_prefix,
                    "desc": stage_desc,
                    "rel_l2_diff_pct": rel_l2_diff,
                    "pt_l2": pt_l2,
                    "cpp_l2": cpp_l2,
                    "max_f10_diff": max_f10_diff
                }

            row = f"{key:<30} | {pt_l2:12.4f} | {cpp_l2:12.4f} | {rel_l2_diff:15.4f}% | {max_f10_diff:16.4f} | {status}"
            # Print select rows (e.g. layer 0, layer 1, layer 2, layer 5, layer 10, layer 20, layer 29)
            if layer in (0, 1, 2, 5, 10, 20, 29) or rel_l2_diff >= 10.0:
                print(row)

            comparison_results.append({
                "layer": layer,
                "stage": stage_prefix,
                "desc": stage_desc,
                "pt_l2": pt_l2,
                "cpp_l2": cpp_l2,
                "rel_l2_diff_pct": rel_l2_diff,
                "max_f10_diff": max_f10_diff,
                "cos_f10": cos_f10,
                "status": status
            })

    # Final Norm & Output
    for key in ["result_norm", "result_output"]:
        if key in pt_map and key in cpp_map:
            pt_t = pt_map[key]
            cpp_t = cpp_map[key]
            pt_l2 = pt_t["last_tok_l2"]
            cpp_l2 = cpp_t["last_tok_l2"]
            rel_l2_diff = abs(pt_l2 - cpp_l2) / max(pt_l2, 1e-9) * 100.0
            pt_f10 = np.array(pt_t["last_tok_first10"])
            cpp_f10 = np.array(cpp_t["last_tok_first10"])
            max_f10_diff = float(np.max(np.abs(pt_f10 - cpp_f10)))
            status = "MATCH" if rel_l2_diff < 1.0 else ("DRIFT" if rel_l2_diff < 10.0 else "DIVERGED")
            print(f"{key:<30} | {pt_l2:12.4f} | {cpp_l2:12.4f} | {rel_l2_diff:15.4f}% | {max_f10_diff:16.4f} | {status}")
            comparison_results.append({
                "layer": -1,
                "stage": key,
                "desc": "Final Norm / Logits",
                "pt_l2": pt_l2,
                "cpp_l2": cpp_l2,
                "rel_l2_diff_pct": rel_l2_diff,
                "max_f10_diff": max_f10_diff,
                "status": status
            })

    print("-" * 105)
    print("\n=== LOGITS COMPARISON ===")
    pt_logits = d_pt["logits_stats"]
    cpp_logits = d_cpp["logits_stats"]
    print(f"PyTorch logits L2 norm: {pt_logits['l2']:.4f} | Mean: {pt_logits['mean']:.4f} | Std: {pt_logits['std']:.4f}")
    print(f"bitnet.cpp logits L2 norm: {cpp_logits['l2']:.4f} | Mean: {cpp_logits['mean']:.4f} | Std: {cpp_logits['std']:.4f}")
    rel_logits_l2 = abs(pt_logits['l2'] - cpp_logits['l2']) / pt_logits['l2'] * 100.0
    print(f"Logits L2 relative difference: {rel_logits_l2:.2f}%")

    print("\nTop-5 Predictions:")
    print("  Rank | PyTorch (ID, piece, prob, logit)       | bitnet.cpp (ID, piece, prob, logit)")
    print("  " + "-" * 75)
    for r in range(5):
        pt_entry = pt_logits["top10"][r]
        cpp_entry = cpp_logits["top10"][r]
        pt_s = f"ID {pt_entry['id']:6d} {repr(pt_entry['piece']):<8} {pt_entry['prob']*100:5.2f}% ({pt_entry['logit']:6.2f})"
        cpp_s = f"ID {cpp_entry['id']:6d} {repr(cpp_entry['piece']):<8} {cpp_entry['prob']*100:5.2f}% ({cpp_entry['logit']:6.2f})"
        print(f"  {r+1:4d} | {pt_s:<38} | {cpp_s}")

    print("\n" + "=" * 105)
    if first_divergence:
        print(f"FIRST SIGNIFICANT DIVERGENCE DETECTED:")
        print(f"  Stage:       {first_divergence['stage']} (Layer {first_divergence['layer']}) - {first_divergence['desc']}")
        print(f"  Rel L2 diff: {first_divergence['rel_l2_diff_pct']:.4f}%")
        print(f"  PT L2:       {first_divergence['pt_l2']:.4f}")
        print(f"  CPP L2:      {first_divergence['cpp_l2']:.4f}")
        print(f"  Max F10 diff:{first_divergence['max_f10_diff']:.4f}")
    else:
        print("NO STAGE WITH >5% L2 DIVERGENCE FOUND.")
    print("=" * 105)

    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        json.dump({
            "first_divergence": first_divergence,
            "logits_comparison": {
                "pt_l2": pt_logits["l2"],
                "cpp_l2": cpp_logits["l2"],
                "rel_l2_diff_pct": rel_logits_l2,
                "pt_top1": pt_logits["top10"][0],
                "cpp_top1": cpp_logits["top10"][0]
            },
            "stages": comparison_results
        }, f, indent=2, ensure_ascii=False)
    print(f"\nSaved full comparison report to: {OUT_REPORT}")

if __name__ == "__main__":
    main()
