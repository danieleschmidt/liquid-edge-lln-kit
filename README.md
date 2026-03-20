# liquid-edge-lln-kit

Tiny **Liquid Neural Networks (LNN)** for edge robots — designed to run on Cortex-M and ESP32.

---

## What is a Liquid Neural Network?

LNNs are continuous-time recurrent networks whose hidden state evolves via an ODE:

```
dh/dt = (-h + tanh(W_in x + W_rec h + bias)) / tau
h_new = h + dt * dh/dt          # Euler integration
```

`tau` is a learned time-constant vector that controls how fast each neuron reacts — giving the network adaptive, physically-interpretable dynamics.

---

## Install

```bash
pip install numpy
```

---

## Quick Start

```python
from lnn import LiquidNetwork, Trainer, export_c_header
import numpy as np

# Build a 1-cell network: 1 input → 8 hidden
net = LiquidNetwork.from_spec(input_dim=1, hidden_dim=8, n_cells=1, seed=42)

# Train on sine wave
T = 60
t = np.linspace(0, 4 * np.pi, T)
X = np.sin(t)[:-1].reshape(-1, 1)
y = np.sin(t)[1:]

trainer = Trainer(net)
losses = trainer.fit(X, y, epochs=20, lr=0.01)
print(f"MSE: {losses[0]:.4f} → {losses[-1]:.4f}")

# Export weights for embedded use
export_c_header(net, "weights.h")
```

---

## Package Layout

```
lnn/
  cell.py      — LiquidCell: single ODE cell
  network.py   — LiquidNetwork: stack of cells, time-series forward pass
  trainer.py   — Trainer: numerical-gradient MSE minimisation
  export.py    — export_c_header(): write weights as C header
demo.py        — sine-wave demo + C export
tests/         — pytest suite (10+ tests)
```

---

## Embedded Deployment

`export_c_header(net, "weights.h")` writes a plain C header:

```c
#define N_CELLS 1
#define HIDDEN_DIM 8
float W_in_0[8][1] = {{...}};
float tau_0[8]     = {...};
```

Include in your Cortex-M / ESP32 firmware and implement the two-line Euler step in C — no framework required.

---

## Run Tests

```bash
pytest tests/
```

---

## License

MIT
