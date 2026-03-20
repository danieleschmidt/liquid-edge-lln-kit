"""Tests for liquid-edge-lln-kit."""
import os
import tempfile
import numpy as np
import pytest

from lnn import LiquidCell, LiquidNetwork, Trainer, export_c_header


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def cell():
    return LiquidCell(input_dim=2, hidden_dim=4, seed=0)


@pytest.fixture
def net():
    return LiquidNetwork.from_spec(input_dim=2, hidden_dim=4, n_cells=1, seed=0)


@pytest.fixture
def net2():
    """Network with 2 stacked cells."""
    return LiquidNetwork.from_spec(input_dim=2, hidden_dim=4, n_cells=2, seed=0)


# ============================================================
# LiquidCell tests
# ============================================================

def test_cell_output_shape(cell):
    """Output of a single step must match hidden_dim."""
    x = np.zeros(2)
    h = cell.init_state()
    h_new = cell.step(x, h)
    assert h_new.shape == (4,), f"Expected shape (4,), got {h_new.shape}"


def test_cell_hidden_state_updates(cell):
    """Hidden state must change after a step (non-zero input)."""
    x = np.array([1.0, 0.5])
    h = cell.init_state()
    h_new = cell.step(x, h)
    assert not np.allclose(h_new, h), "Hidden state did not update"


def test_cell_tau_shape(cell):
    """tau parameter must have shape (hidden_dim,)."""
    assert cell.tau.shape == (4,), f"Expected tau shape (4,), got {cell.tau.shape}"


def test_cell_tau_initial_value(cell):
    """tau should be initialised to ones."""
    np.testing.assert_array_equal(cell.tau, np.ones(4))


def test_cell_different_inputs_produce_different_outputs(cell):
    """Different inputs should produce different hidden state updates."""
    h = cell.init_state()
    x1 = np.array([1.0, 0.0])
    x2 = np.array([0.0, 1.0])
    h1 = cell.step(x1, h)
    h2 = cell.step(x2, h)
    assert not np.allclose(h1, h2), "Different inputs gave identical outputs"


def test_cell_init_state_zeros(cell):
    """Initial hidden state should be all zeros."""
    h = cell.init_state()
    np.testing.assert_array_equal(h, np.zeros(4))


# ============================================================
# LiquidNetwork tests
# ============================================================

def test_network_forward_sequence_length(net):
    """Forward pass over T timesteps should return T outputs."""
    T = 10
    X = np.random.randn(T, 2)
    out = net.forward(X)
    assert out.shape[0] == T, f"Expected {T} outputs, got {out.shape[0]}"


def test_network_output_shape_matches_hidden_dim(net):
    """Output at each timestep should have the cell's hidden_dim."""
    T = 5
    X = np.random.randn(T, 2)
    out = net.forward(X)
    assert out.shape == (T, 4), f"Expected shape ({T}, 4), got {out.shape}"


def test_network_two_cells(net2):
    """Network with 2 stacked cells should still produce (T, hidden_dim) output."""
    T = 7
    X = np.random.randn(T, 2)
    out = net2.forward(X)
    assert out.shape == (T, 4), f"Expected shape ({T}, 4), got {out.shape}"


def test_network_output_changes_over_time(net):
    """The output should not be constant across all timesteps."""
    T = 10
    X = np.random.randn(T, 2)
    out = net.forward(X)
    assert not np.allclose(out[0], out[-1]), "Output identical at all timesteps"


# ============================================================
# export_c_header tests
# ============================================================

def test_export_creates_file(net):
    """export_c_header should create a file on disk."""
    with tempfile.NamedTemporaryFile(suffix=".h", delete=False) as f:
        fname = f.name
    try:
        export_c_header(net, fname)
        assert os.path.exists(fname), "Header file was not created"
    finally:
        if os.path.exists(fname):
            os.remove(fname)


def test_export_contains_float_keyword(net):
    """Exported header must contain the 'float' keyword."""
    with tempfile.NamedTemporaryFile(suffix=".h", delete=False, mode="w") as f:
        fname = f.name
    try:
        export_c_header(net, fname)
        content = open(fname).read()
        assert "float" in content, "'float' keyword not found in exported header"
    finally:
        if os.path.exists(fname):
            os.remove(fname)


def test_export_contains_dimension_defines(net):
    """Exported header must contain #define N_CELLS and #define HIDDEN_DIM."""
    with tempfile.NamedTemporaryFile(suffix=".h", delete=False, mode="w") as f:
        fname = f.name
    try:
        export_c_header(net, fname)
        content = open(fname).read()
        assert "#define N_CELLS" in content, "#define N_CELLS missing"
        assert "#define HIDDEN_DIM" in content, "#define HIDDEN_DIM missing"
    finally:
        if os.path.exists(fname):
            os.remove(fname)


# ============================================================
# Trainer tests
# ============================================================

def test_trainer_reduces_mse():
    """Training should reduce MSE compared to the initial loss."""
    np.random.seed(7)
    T = 30
    t = np.linspace(0, 2 * np.pi, T)
    X = np.sin(t)[:-1].reshape(-1, 1)
    y = np.sin(t)[1:]

    net = LiquidNetwork.from_spec(input_dim=1, hidden_dim=4, n_cells=1, seed=7)
    trainer = Trainer(net, epsilon=1e-4)
    initial_loss = trainer._mse(X, y)
    losses = trainer.fit(X, y, epochs=3, lr=0.05)
    final_loss = losses[-1]
    assert final_loss < initial_loss, (
        f"MSE did not decrease: initial={initial_loss:.6f}, final={final_loss:.6f}"
    )


def test_demo_sine_wave_improves():
    """End-to-end: training on sine wave should reduce MSE."""
    np.random.seed(42)
    T = 40
    t = np.linspace(0, 4 * np.pi, T)
    X = np.sin(t)[:-1].reshape(-1, 1)
    y = np.sin(t)[1:]

    net = LiquidNetwork.from_spec(input_dim=1, hidden_dim=8, n_cells=1, seed=42)
    trainer = Trainer(net, epsilon=1e-4)
    initial_loss = trainer._mse(X, y)
    losses = trainer.fit(X, y, epochs=5, lr=0.05)
    final_loss = losses[-1]
    assert final_loss < initial_loss, (
        f"Sine wave MSE did not improve: {initial_loss:.6f} -> {final_loss:.6f}"
    )
