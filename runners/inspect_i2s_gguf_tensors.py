import sys
import os
sys.path.insert(0, r"C:\ai_models\BitNet\3rdparty\llama.cpp\gguf-py")
import gguf

GGUF_PATH = r"C:\ai_models\BitNet\models\BitNet-b1.58-2B-4T\ggml-model-i2_s.gguf"

r = gguf.GGUFReader(GGUF_PATH)
print("=== GGUF METADATA FIELDS ===")
for k, v in r.fields.items():
    if not "token" in k and not "merges" in k:
        try:
            val = v.contents()
        except Exception:
            val = "<raw>"
        print(f"  {k}: {val}")

print(f"\nTotal tensors in GGUF: {len(r.tensors)}")
print("Sample tensors:")
for i, t in enumerate(r.tensors[:10]):
    print(f"  [{i:2d}] {t.name:35s} | shape: {str(t.shape):20s} | type: {t.tensor_type}")

# Find layer 0 attention q tensor
q0 = next((t for t in r.tensors if "blk.0.attn_q.weight" in t.name), None)
if q0:
    print(f"\nFound blk.0.attn_q.weight:")
    print(f"  Name: {q0.name}")
    print(f"  Shape: {q0.shape}")
    print(f"  Tensor Type: {q0.tensor_type} (GGML type {int(q0.tensor_type)})")
    print(f"  Raw data len (bytes): {len(q0.data.tobytes())}")
