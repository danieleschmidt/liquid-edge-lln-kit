"""Trainer: fit a LiquidNetwork on time-series data using numerical gradients."""
import numpy as np
from .network import LiquidNetwork


class Trainer:
    """
    Train a LiquidNetwork by minimising MSE via numerical (finite-difference) gradients.

    Parameters
    ----------
    network : LiquidNetwork
    epsilon : float
        Perturbation size for numerical gradient estimation.
    """

    def __init__(self, network: LiquidNetwork, epsilon: float = 1e-4):
        self.network = network
        self.epsilon = epsilon

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _mse(self, X: np.ndarray, y: np.ndarray) -> float:
        preds = self.network.forward(X)
        # y may be 1-D (T,) or 2-D (T, out_dim)
        if y.ndim == 1:
            preds = preds.mean(axis=1)
        diff = preds - y
        return float(np.mean(diff ** 2))

    def _update_param(self, param: np.ndarray, grad: np.ndarray, lr: float):
        """In-place gradient-descent step."""
        param -= lr * grad

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray,
            epochs: int = 100, lr: float = 0.01) -> list:
        """
        Fit the network to (X, y) via numerical gradients.

        Parameters
        ----------
        X : ndarray, shape (T, input_dim)
        y : ndarray, shape (T,) or (T, output_dim)
        epochs : int
        lr : float

        Returns
        -------
        losses : list[float]
            MSE after each epoch.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        eps = self.epsilon
        losses = []

        for epoch in range(epochs):
            base_loss = self._mse(X, y)
            losses.append(base_loss)

            # Iterate over each cell's parameters
            for cell in self.network.cells:
                for param in cell.get_params():
                    grad = np.zeros_like(param)
                    it = np.nditer(param, flags=["multi_index"])
                    while not it.finished:
                        idx = it.multi_index
                        orig = param[idx]

                        param[idx] = orig + eps
                        loss_plus = self._mse(X, y)

                        param[idx] = orig - eps
                        loss_minus = self._mse(X, y)

                        grad[idx] = (loss_plus - loss_minus) / (2 * eps)
                        param[idx] = orig
                        it.iternext()

                    self._update_param(param, grad, lr)

        # Append final loss
        losses.append(self._mse(X, y))
        return losses
