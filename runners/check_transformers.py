import sys
try:
    from transformers.integrations.bitnet import AutoBitLinear
    print("BitNet integration: FOUND")
except ImportError as e:
    print("BitNet integration:", e)
