import os
import sys
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Ensure required module paths are in sys.path
sys.path.insert(0, r"C:\ai_models\BitNet\utils")
sys.path.insert(0, r"C:\ai_models\BitNet\3rdparty\llama.cpp")
sys.path.insert(0, r"C:\ai_models\BitNet\3rdparty\llama.cpp\gguf-py")

src_dir = r"C:\ai_models\BitNet-b1.58-2B-4T"
out_file = r"C:\ai_models\BitNet\models\BitNet-b1.58-2B-4T\ggml-model-f32.gguf"

os.makedirs(os.path.dirname(out_file), exist_ok=True)

# Set sys.argv for the converter script
sys.argv = [
    "convert-hf-to-gguf-bitnet.py",
    src_dir,
    "--outtype", "f32",
    "--outfile", out_file
]

print("Starting GGUF F32 conversion...")
print(f"Source: {src_dir}")
print(f"Target: {out_file}", flush=True)

# Load and execute converter
import importlib.util
spec = importlib.util.spec_from_file_location("convert_bitnet", r"C:\ai_models\BitNet\utils\convert-hf-to-gguf-bitnet.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

mod.args = mod.parse_args()
mod.main()
print("Conversion completed successfully!", flush=True)
