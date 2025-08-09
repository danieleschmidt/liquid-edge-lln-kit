"""Performance tests for liquid edge neural networks."""

import pytest
import jax
import jax.numpy as jnp
from liquid_edge.core import LiquidNN, LiquidConfig
from liquid_edge.layers import LiquidCell, LiquidRNN


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return LiquidConfig(
        input_size=10,
        hidden_size=20,
        output_size=5,
        time_constant=1.0
    )


@pytest.fixture
def sample_input():
    """Sample input data for benchmarking."""
    return jax.random.normal(jax.random.PRNGKey(42), (32, 10))


@pytest.mark.benchmark
def test_liquid_nn_inference_speed(benchmark, sample_config, sample_input):
    """Benchmark inference speed of LiquidNN."""
    model = LiquidNN(sample_config)
    
    # Initialize parameters
    key = jax.random.PRNGKey(0)
    params = model.init(key, sample_input)
    
    # Benchmark inference
    def inference():
        return model.apply(params, sample_input)
    
    result = benchmark(inference)
    
    # Performance assertion - should complete in < 10ms for this small model
    assert benchmark.stats['mean'] < 0.01  # 10ms


@pytest.mark.benchmark
def test_liquid_cell_computation_speed(benchmark, sample_input):
    """Benchmark single liquid cell computation."""
    model = LiquidCell(features=20)
    hidden = jnp.zeros((32, 20))
    
    key = jax.random.PRNGKey(0)
    params = model.init(key, sample_input, hidden)
    
    def cell_forward():
        return model.apply(params, sample_input, hidden)
    
    result = benchmark(cell_forward)
    
    # Should be very fast for single cell
    assert benchmark.stats['mean'] < 0.001  # 1ms


@pytest.mark.benchmark
def test_liquid_rnn_sequence_processing(benchmark):
    """Benchmark RNN processing on sequences."""
    seq_length = 100
    batch_size = 16
    input_size = 10
    
    model = LiquidRNN(features=20)
    sequence_input = jax.random.normal(
        jax.random.PRNGKey(42), 
        (batch_size, seq_length, input_size)
    )
    
    key = jax.random.PRNGKey(0)
    params = model.init(key, sequence_input)
    
    def rnn_forward():
        return model.apply(params, sequence_input)
    
    result = benchmark(rnn_forward)
    
    # Sequence processing should be reasonable
    assert benchmark.stats['mean'] < 0.1  # 100ms for 100-step sequence


@pytest.mark.benchmark
def test_memory_efficiency():
    """Test memory usage of liquid neural networks."""
    config = LiquidConfig(
        input_dim=100,
        hidden_dim=200,
        output_dim=50
    )
    
    model = LiquidNN(config)
    large_input = jax.random.normal(jax.random.PRNGKey(42), (128, 100))
    
    key = jax.random.PRNGKey(0)
    params = model.init(key, large_input)
    
    # Check parameter count is reasonable for edge deployment
    param_count = sum(p.size for p in jax.tree_util.tree_leaves(params))
    
    # Should be < 100K parameters for edge efficiency
    assert param_count < 100000, f"Model has {param_count} parameters, too large for edge"


@pytest.mark.benchmark
@pytest.mark.parametrize("batch_size", [1, 8, 32])
def test_scalability_with_batch_size(benchmark, batch_size):
    """Test how performance scales with batch size."""
    config = LiquidConfig(input_size=10, hidden_size=20, output_size=5)
    model = LiquidNN(config)
    
    input_batch = jax.random.normal(jax.random.PRNGKey(42), (batch_size, 10))
    
    key = jax.random.PRNGKey(0)
    params = model.init(key, input_batch)
    
    def batched_inference():
        return model.apply(params, input_batch)
    
    result = benchmark(batched_inference)
    
    # Performance should scale roughly linearly with batch size
    expected_max_time = 0.001 + (batch_size * 0.0001)  # Base + per-sample cost
    assert benchmark.stats['mean'] < expected_max_time