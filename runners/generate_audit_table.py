import json

with open(r"C:\ai_models\benchmark_results_bitnet_base.json", "r", encoding="utf-8") as f:
    pt_base = json.load(f)

with open(r"C:\ai_models\paper_artifacts\symmetric_dense_vs_ternary.json", "r", encoding="utf-8") as f:
    cpp_data = json.load(f)

cpp_bitnet = cpp_data["bitnet_2b_i2_s_base"]["detailed_results"]
cpp_qwen3 = cpp_data["qwen2.5_3b_q4_k_m"]["detailed_results"]

# Generate detailed markdown audit table
lines = []
lines.append("# Аудит и ручная переоценка 20 пилотных кейсов\n")
lines.append("## 1. Покейсовая верификационная таблица\n")
lines.append("| ID | Категория | Запрос | Ожидаемый тул | PyTorch BitNet (FP32) | C++ BitNet (I2_S) | C++ Qwen 3B (Q4_K_M) |")
lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")

for i in range(20):
    pt_c = pt_base["detailed_results"][i]
    cp_bn = cpp_bitnet[i]
    cp_qw = cpp_qwen3[i]

    cid = pt_c["id"]
    cat = pt_c["category"]
    query = pt_c.get("user_query") or pt_c.get("messages", [{}])[-1].get("content", "")
    query_clean = query.replace("\n", " ")
    exp_tool = str(pt_c["expected_tool"])

    # PyTorch evaluation
    pt_out = pt_c["model_response"].replace("\n", " ").replace("|", "\\|")
    pt_trig = pt_c["attempted_tool"]
    pt_exec = pt_c["execution_success"]
    pt_status = "OK" if (pt_exec if pt_c["needs_tool"] else pt_c["negative_success"]) else "FAIL"

    # C++ BitNet evaluation
    bn_out = cp_bn["response"].replace("\n", " ").replace("|", "\\|")
    bn_trig = cp_bn["trigger_success"]
    bn_exec = cp_bn["execution_success"]
    bn_status = "OK" if (bn_exec if cp_bn["needs_tool"] else cp_bn["negative_success"]) else "FAIL"

    # C++ Qwen 3B evaluation
    qw_out = cp_qw["response"].replace("\n", " ").replace("|", "\\|")
    qw_trig = cp_qw["trigger_success"]
    qw_exec = cp_qw["execution_success"]
    qw_status = "OK" if (qw_exec if cp_qw["needs_tool"] else cp_qw["negative_success"]) else "FAIL"

    pt_summary = f"[{pt_status}] Trig={pt_trig}, ValidSchema={pt_exec}<br>`{pt_out[:60]}...`"
    bn_summary = f"[{bn_status}] Trig={bn_trig}, ValidSchema={bn_exec}<br>`{bn_out[:60]}...`"
    qw_summary = f"[{qw_status}] Trig={qw_trig}, ValidSchema={qw_exec}<br>`{qw_out[:60]}...`"

    lines.append(f"| {cid:02d} | {cat} | {query_clean} | `{exp_tool}` | {pt_summary} | {bn_summary} | {qw_summary} |")

out_md = "\n".join(lines)
with open(r"C:\ai_models\paper_artifacts\AUDIT_20_CASES.md", "w", encoding="utf-8") as f:
    f.write(out_md)

with open(r"C:\Users\Рам\Desktop\BitNet_Research_Reports\AUDIT_20_CASES.md", "w", encoding="utf-8") as f:
    f.write(out_md)

print("AUDIT_20_CASES.md generated successfully!")
