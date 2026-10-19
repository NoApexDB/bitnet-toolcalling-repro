# -*- coding: utf-8 -*-
"""Wrapper: run convert-hf-to-gguf-bitnet.py with its directory on sys.path
(ComfyUI python_embeded runs in isolated ._pth mode: script dir and PYTHONPATH are ignored)."""
import sys

sys.path.insert(0, r"C:\ai_models\BitNet\3rdparty\llama.cpp\gguf-py")
sys.path.insert(0, r"C:\ai_models\BitNet\utils")
import gguf  # verify the right one loads
assert hasattr(gguf.MODEL_ARCH, "BITNET_B158"), "wrong gguf package loaded"

import runpy
runpy.run_path(r"C:\ai_models\BitNet\utils\convert-hf-to-gguf-bitnet.py", run_name="__main__")
