"""LiquidNetwork: a stack of LiquidCells that processes time-series data."""
import numpy as np
from .cell import LiquidCell


class LiquidNetwork:
    """
    Stack of LiquidCells that processes a sequence of inputs.

    Parameters
    ----------
    cells : list[LiquidCell]
        Ordered list of LiquidCell instances.
    """

    def __init__(self, cells: list):
        if not cells:
            raise ValueError("cells must be a non-empty list of LiquidCell instances.")
        self.cells = cells

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_spec(cls, input_dim: int, hidden_dim: int, n_cells: int = 1,
                  dt: float = 0.1, seed: int = None) -> "LiquidNetwork":
        """Build a network of *n_cells* identical cells stacked in series."""
        cells = []
        for i in range(n_cells):
            in_d = input_dim if i == 0 else hidden_dim
            cells.append(LiquidCell(in_d, hidden_dim, dt=dt,
                                    seed=None if seed is None else seed + i))
        return cls(cells)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, x_sequence: np.ndarray) -> np.ndarray:
        """
        Run the network over a time sequence.

        Parameters
        ----------
        x_sequence : ndarray, shape (T, input_dim)

        Returns
        -------
        outputs : ndarray, shape (T, hidden_dim_of_last_cell)
        """
        x_sequence = np.asarray(x_sequence, dtype=float)
        T = x_sequence.shape[0]

        # Initialise hidden states for every cell
        states = [cell.init_state() for cell in self.cells]
        outputs = []

        for t in range(T):
            x = x_sequence[t]
            for i, cell in enumerate(self.cells):
                h_new = cell.step(x, states[i])
                states[i] = h_new
                x = h_new          # feed output of this cell into next cell
            outputs.append(states[-1].copy())

        return np.array(outputs)   # (T, hidden_dim)
