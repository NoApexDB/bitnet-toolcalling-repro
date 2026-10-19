# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, r"C:\ai_models\BitNet\3rdparty\llama.cpp\gguf-py")
sys.path.insert(0, r"C:\ai_models\BitNet\utils")
from transformers import AutoTokenizer
import gguf

# 1. Read existing GGUF metadata
i2s_path = r"C:\ai_models\BitNet\models\BitNet-b1.58-2B-4T\ggml-model-i2_s.gguf"
reader = gguf.GGUFReader(i2s_path)
print("=== Existing GGUF metadata ===")
pre_found = False
for f in reader.fields.values():
    if "tokenizer.ggml" in f.name:
        parts = [f.parts[p] for p in f.data] if f.data else []
        val_str = str(parts[0]) if parts else "EMPTY"
        if f.name in ["tokenizer.ggml.model", "tokenizer.ggml.pre", "tokenizer.ggml.add_bos_token", "tokenizer.ggml.bos_token_id"]:
            if isinstance(parts[0], (bytes, bytearray, memoryview)):
                val_str = bytes(parts[0]).decode("utf-8", errors="replace")
            print(f"  {f.name} = {val_str}")
        if f.name == "tokenizer.ggml.pre":
            pre_found = True

print(f"tokenizer.ggml.pre present in existing GGUF: {pre_found}")

# 2. Check HF tokenizer profile
tok_dir = r"C:\ai_models\BitNet-b1.58-2B-4T"
tokenizer = AutoTokenizer.from_pretrained(tok_dir)
print(f"HF tokenizer class: {tokenizer.__class__.__name__}")
print(f"HF vocab size: {len(tokenizer.vocab)}")

import runpy
# check what convert script does
with open(r"C:\ai_models\BitNet\utils\convert-hf-to-gguf-bitnet.py", "r", encoding="utf-8") as f:
    content = f.read()

# Let's inspect get_vocab_base_pre from convert-hf-to-gguf-bitnet
from pathlib import Path
class Dummy:
    dir_model = Path(tok_dir)
    hparams = {"vocab_size": 128256}

import importlib
convert_mod = runpy.run_path(r"C:\ai_models\BitNet\utils\convert-hf-to-gguf-bitnet.py")
ModelClass = convert_mod["Model"]
model_inst = convert_mod["BitnetModel"](Path(tok_dir), 0, Path("dummy.gguf"), False, False)
try:
    pre_val = model_inst.get_vocab_base_pre(tokenizer)
    print(f"get_vocab_base_pre result: {pre_val}")
except Exception as e:
    print(f"get_vocab_base_pre raised: {type(e).__name__}: {e}")
