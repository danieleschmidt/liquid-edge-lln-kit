"""LiquidCell: A single liquid neural network cell using Euler ODE integration."""
import numpy as np


class LiquidCell:
    """
    A Liquid Neural Network cell that integrates hidden state via Euler method.

    ODE: dh/dt = (-h + tanh(W_in @ x + W_rec @ h + bias)) / tau
    Update: h_new = h + dt * dh/dt

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input.
    hidden_dim : int
        Dimensionality of the hidden state.
    dt : float
        Euler integration step size (default 0.1).
    seed : int, optional
        Random seed for reproducibility.
    """

    def __init__(self, input_dim: int, hidden_dim: int, dt: float = 0.1, seed: int = None):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dt = dt

        rng = np.random.default_rng(seed)
        scale = 0.1
        self.W_in = rng.normal(0, scale, (hidden_dim, input_dim))
        self.W_rec = rng.normal(0, scale, (hidden_dim, hidden_dim))
        self.tau = np.ones(hidden_dim)          # time constants (trainable)
        self.bias = np.zeros(hidden_dim)

    # ------------------------------------------------------------------
    # Core step
    # ------------------------------------------------------------------

    def step(self, x: np.ndarray, h: np.ndarray) -> np.ndarray:
        """
        Perform one Euler integration step.

        Parameters
        ----------
        x : ndarray, shape (input_dim,)
        h : ndarray, shape (hidden_dim,)

        Returns
        -------
        h_new : ndarray, shape (hidden_dim,)
        """
        x = np.asarray(x, dtype=float).ravel()
        h = np.asarray(h, dtype=float).ravel()
        tau = np.clip(self.tau, 1e-6, None)     # prevent division by zero
        pre = self.W_in @ x + self.W_rec @ h + self.bias
        dh = (-h + np.tanh(pre)) / tau
        return h + self.dt * dh

    def init_state(self) -> np.ndarray:
        """Return a zero initial hidden state."""
        return np.zeros(self.hidden_dim)

    # ------------------------------------------------------------------
    # Parameter helpers (used by Trainer for numerical gradients)
    # ------------------------------------------------------------------

    def get_params(self) -> list:
        """Return list of all parameter arrays (references)."""
        return [self.W_in, self.W_rec, self.tau, self.bias]

    def set_params(self, params: list):
        """Set parameters from a list of arrays."""
        self.W_in, self.W_rec, self.tau, self.bias = params
