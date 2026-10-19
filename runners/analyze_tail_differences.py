import json
import sys
sys.stdout.reconfigure(encoding='utf-8')

with open("C:/ai_models/paper_artifacts/token_alignment_full_audit.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# The base system prompt ends at some position. Let's see Case 1 vs other cases.
# Base prompt differences account for 39 tokens across all cases.
# Let's inspect where the extra tokens appear in the user queries for Cases 7, 9, 10, 14, 17, 18.

with open("C:/ai_models/test_suite_standardized.json", "r", encoding="utf-8") as f:
    test_cases = {c["id"]: c for c in json.load(f)}

print("=== EXTRA TOKEN DISCREPANCIES IN USER QUERIES ===")
for cid in [7, 9, 10, 14, 17, 18]:
    c_audit = next(c for c in data["cases"] if c["case_id"] == cid)
    tc = test_cases[cid]
    user_text = tc["messages"][-1]["content"]
    extra = c_audit["delta_cpp_minus_hf_nobos"] - 39
    print(f"\nCase {cid} (Extra tokens: +{extra}):")
    print(f"  User Query: {user_text}")
    
    # Compare user query tokenization alone
    # Let's find tokens corresponding to the user query in HF vs GGUF
    # By taking the last tokens
    hf_pieces = c_audit["hf_pieces"]
    cpp_pieces = c_audit["cpp_pieces"]
    
    print(f"  HF Tail (last 10 pieces): {hf_pieces[-10:]}")
    print(f"  CPP Tail (last 10 pieces): {cpp_pieces[-10:]}")
