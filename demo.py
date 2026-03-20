"""Demo: fit a LiquidNetwork on sine-wave prediction, then export weights."""
import numpy as np
from lnn import LiquidNetwork, Trainer, export_c_header

# ------------------------------------------------------------------
# Build dataset: predict next value of a sine wave
# ------------------------------------------------------------------
T = 60
t = np.linspace(0, 4 * np.pi, T)
signal = np.sin(t)

X = signal[:-1].reshape(-1, 1)   # (T-1, 1) — input  at step t
y = signal[1:]                   # (T-1,)   — target at step t+1

# ------------------------------------------------------------------
# Build network: 1 cell, hidden_dim=8, input_dim=1
# ------------------------------------------------------------------
net = LiquidNetwork.from_spec(input_dim=1, hidden_dim=8, n_cells=1, seed=42)

# ------------------------------------------------------------------
# Measure initial loss
# ------------------------------------------------------------------
trainer = Trainer(net, epsilon=1e-4)
initial_loss = trainer._mse(X, y)
print(f"Initial MSE : {initial_loss:.6f}")

# ------------------------------------------------------------------
# Train  (small network + few epochs so the demo runs fast)
# ------------------------------------------------------------------
losses = trainer.fit(X, y, epochs=5, lr=0.05)
final_loss = losses[-1]
print(f"Final   MSE : {final_loss:.6f}")
print(f"Improved    : {final_loss < initial_loss}")

# ------------------------------------------------------------------
# Export weights
# ------------------------------------------------------------------
export_c_header(net, "weights.h")
print("Weights exported to weights.h")
