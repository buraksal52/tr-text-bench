import torch

# Phase 0 check: which device can PyTorch use on this machine?
print("PyTorch version:", torch.__version__)

if torch.cuda.is_available():
    device_name = "cuda"
elif torch.backends.mps.is_available():
    device_name = "mps"
else:
    device_name = "cpu"

print("Device to use:", device_name)

# A tiny calculation on the chosen device proves that it really works.
numbers = torch.ones(3, device=device_name)
print("Test calculation:", (numbers * 2).tolist())
