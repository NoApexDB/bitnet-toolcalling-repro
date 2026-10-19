import torch
from transformers.integrations.bitnet import unpack_weights

t = torch.tensor([[0b10100001]], dtype=torch.uint8)
res = unpack_weights(t, torch.float32)
print("SUCCESS! Unpack result:", res)
