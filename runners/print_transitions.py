import json

d = json.load(open('C:/ai_models/paper_artifacts/heldout_30_cases_paired_transitions.json', encoding='utf-8'))
for t in d['paired_transitions']:
    c1 = t['states']['cell_1_swiglu_string']
    c2 = t['states']['cell_2_swiglu_tokens']
    c3 = t['states']['cell_3_relu2_string']
    c4 = t['states']['cell_4_relu2_tokens']
    r3 = t['responses']['cell_3'].replace('\n', ' ')[:35]
    r4 = t['responses']['cell_4'].replace('\n', ' ')[:35]
    print(f"Case {t['case_id']:02d} [{t['category'][:11]:11s}]: c1={c1:28s} -> c3={c3:28s} -> c4={c4:25s} | c3='{r3}' | c4='{r4}'")
