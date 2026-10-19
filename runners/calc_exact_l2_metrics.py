import json
import numpy as np

pt_data = json.load(open('C:/ai_models/paper_artifacts/pytorch_layer_activations.json', 'r', encoding='utf-8'))
cpp_data = json.load(open('C:/ai_models/paper_artifacts/cpp_layer_activations_relu2.json', 'r', encoding='utf-8'))

cpp_tensors = {t['name']: t for t in cpp_data['tensors']}
pt_tensors = pt_data['tensors']

stages = [
    ('ffn_norm-0', 'ffn_norm-0', 'FFN Norm'),
    ('ffn_gate-0', 'ffn_gate-0', 'Gate (W_gate*x)'),
    ('ffn_up-0', 'ffn_up-0', 'Up (W_up*x)'),
    ('ffn_relu-0', 'ffn_relu-0', 'ReLU'),
    ('ffn_sqr(relu)-0', 'ffn_sqr(relu)-0', 'Square (ReLU^2)'),
    ('ffn_gate_par-0', 'ffn_sub_out-0', 'Gated Mul (ReLU^2*up)'),
    ('ffn_sub_norm-0', 'ffn_sub_norm-0', 'FFN Sub-Norm'),
    ('ffn_down-0', 'ffn_down-0', 'Down (W_down*sub)'),
    ('l_out-0', 'l_out-0', 'Layer 0 Out')
]

print(f"{'Stage':<22} | {'Dim':<6} | {'PT L2':<10} | {'CPP L2':<10} | {'Rel Norm Diff':<14} | {'Rel L2 Vec Err':<15} | {'Cosine':<10}")
print("-" * 105)

for pt_n, cpp_n, desc in stages:
    u = np.array(pt_tensors[pt_n]['last_token_full'], dtype=np.float64)
    v = np.array(cpp_tensors[cpp_n]['last_token_full'], dtype=np.float64)
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    rel_norm_diff = abs(norm_u - norm_v) / norm_u * 100.0
    rel_vec_err = np.linalg.norm(u - v) / norm_u * 100.0
    cos = np.dot(u, v) / (norm_u * norm_v)
    print(f"{desc:<22} | {len(u):<6} | {norm_u:<10.4f} | {norm_v:<10.4f} | {rel_norm_diff:<13.4f}% | {rel_vec_err:<14.4f}% | {cos:<10.6f}")

# Logits
u_log = np.array(pt_data['logits_stats']['full_logits'], dtype=np.float64)
v_log = np.array(cpp_data['logits_stats']['full_logits'], dtype=np.float64)
norm_u_log = np.linalg.norm(u_log)
norm_v_log = np.linalg.norm(v_log)
rel_norm_log = abs(norm_u_log - norm_v_log) / norm_u_log * 100.0
l2_diff_log = np.linalg.norm(u_log - v_log)
rel_vec_log = l2_diff_log / norm_u_log * 100.0
cos_log = np.dot(u_log, v_log) / (norm_u_log * norm_v_log)
rmse_log = np.sqrt(np.mean((u_log - v_log)**2))

print(f"{'Logits (128k)':<22} | {len(u_log):<6} | {norm_u_log:<10.4f} | {norm_v_log:<10.4f} | {rel_norm_log:<13.4f}% | {rel_vec_log:<14.4f}% | {cos_log:<10.6f}")
print("\nExact Logits Breakdown:")
print(f"  ||PT||_2:                      {norm_u_log:.6f}")
print(f"  ||CPP||_2:                     {norm_v_log:.6f}")
print(f"  Разница норм | ||PT||-||CPP|| |: {abs(norm_u_log - norm_v_log):.6f}")
print(f"  Относительная разница норм:    {rel_norm_log:.6f}%")
print(f"  Евклидова ошибка ||PT - CPP||_2: {l2_diff_log:.6f}")
print(f"  Относительная L2-ошибка вектора: {rel_vec_log:.6f}%")
print(f"  RMSE:                          {rmse_log:.6f}")
print(f"  Косинусное сходство:           {cos_log:.6f}")
