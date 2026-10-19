import json

with open(r"C:\ai_models\benchmark_results_bitnet_base.json", "r", encoding="utf-8") as f:
    pt_base = json.load(f)

with open(r"C:\ai_models\paper_artifacts\symmetric_dense_vs_ternary.json", "r", encoding="utf-8") as f:
    cpp_data = json.load(f)

cpp_bitnet = cpp_data["bitnet_2b_i2_s_base"]["detailed_results"]
cpp_qwen15 = cpp_data["qwen2.5_1.5b_q4_k_m"]["detailed_results"]
cpp_qwen3 = cpp_data["qwen2.5_3b_q4_k_m"]["detailed_results"]

print("=== AUDIT OF 20 TEST CASES ===")
for i, case in enumerate(pt_base["detailed_results"]):
    cid = case["id"]
    cat = case["category"]
    query = case.get("user_query") or case.get("messages", [{}])[-1].get("content", "")
    exp_tool = case["expected_tool"]
    
    # PyTorch response
    pt_resp = case["model_response"]
    pt_trig = case["attempted_tool"]
    pt_exec = case["execution_success"]
    pt_neg = case["negative_success"]

    # C++ BitNet response
    c_bn = cpp_bitnet[i]
    c_bn_resp = c_bn["response"]
    c_bn_trig = c_bn["trigger_success"]
    c_bn_exec = c_bn["execution_success"]
    c_bn_neg = c_bn["negative_success"]

    print(f"\n--- CASE {cid:02d} [{cat}] ---")
    print(f"Query: {query}")
    print(f"Expected Tool: {exp_tool}")
    print(f"PyTorch Base: Trig={pt_trig}, Exec={pt_exec}, Neg={pt_neg} | Output: {repr(pt_resp[:90])}")
    print(f"C++ BitNet:   Trig={c_bn_trig}, Exec={c_bn_exec}, Neg={c_bn_neg} | Output: {repr(c_bn_resp[:90])}")
