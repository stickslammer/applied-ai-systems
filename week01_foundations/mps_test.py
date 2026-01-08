import torch

print("Torch:", torch.__version__)
print("MPS available:", torch.backends.mps.is_available())
print("MPS built:", torch.backends.mps.is_built())

if torch.backends.mps.is_available():
    x = torch.randn(1024, 1024, device="mps")
    y = x @ x
    print("Success, device:", y.device)
