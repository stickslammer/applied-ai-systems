import time
import torch
from torch import nn

def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def main():
    device = pick_device()
    print("Torch:", torch.__version__)
    print("Using device:", device)

    # Reproducibility
    torch.manual_seed(42)

    # Synthetic regression data (small, fast, portable)
    # y = Xw + b + noise
    n_samples = 8192
    n_features = 16
    X = torch.randn(n_samples, n_features)
    true_w = torch.randn(n_features, 1)
    true_b = torch.randn(1)
    y = X @ true_w + true_b + 0.1 * torch.randn(n_samples, 1)

    # Move data to device
    X = X.to(device)
    y = y.to(device)

    # Simple model
    model = nn.Sequential(
        nn.Linear(n_features, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
    ).to(device)

    print("Model parameter device:", next(model.parameters()).device)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

    # Training loop
    epochs = 10
    batch_size = 512

    start = time.time()
    for epoch in range(1, epochs + 1):
        perm = torch.randperm(n_samples, device=device)
        epoch_loss = 0.0

        model.train()
        for i in range(0, n_samples, batch_size):
            idx = perm[i:i + batch_size]
            xb = X[idx]
            yb = y[idx]

            pred = model(xb)
            loss = loss_fn(pred, yb)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / (n_samples / batch_size)
        print(f"Epoch {epoch:02d}/{epochs} | avg loss: {avg_loss:.6f}")

    end = time.time()
    print(f"Training time: {end - start:.2f}s")

    # Inference check + latency timing
    model.eval()
    with torch.no_grad():
        test_x = torch.randn(1024, n_features, device=device)
        t0 = time.time()
        _ = model(test_x)
        t1 = time.time()
        print(f"Inference time (single batch): {(t1 - t0) * 1000:.3f} ms")

    # Autograd sanity check (simple scalar)
    x = torch.tensor(2.0, requires_grad=True, device=device)
    z = x ** 2 + 3 * x
    z.backward()
    print("Autograd check: x.grad should be 2x + 3 = 7. Got:", x.grad.item())

if __name__ == "__main__":
    main()
