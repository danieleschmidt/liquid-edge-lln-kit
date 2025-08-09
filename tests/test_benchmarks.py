"""Performance benchmark tests for Liquid Edge LLN Kit."""

import time
import pytest
import jax
import jax.numpy as jnp
from liquid_edge import LiquidNN, LiquidConfig


class TestPerformanceBenchmarks:
    """Benchmark test suite for performance regression detection."""

    @pytest.mark.benchmark
    def test_training_speed_benchmark(self, basic_config, rng_key, sensor_data, motor_targets):
        """Benchmark training speed."""
        model = LiquidNN(basic_config)
        params = model.init(rng_key, sensor_data[:1])
        
        # Warmup
        for _ in range(5):
            loss, grads = jax.value_and_grad(
                lambda p: jnp.mean((model.apply(p, sensor_data)[0] - motor_targets) ** 2)
            )(params)
        
        # Benchmark
        start_time = time.perf_counter()
        for _ in range(100):
            loss, grads = jax.value_and_grad(
                lambda p: jnp.mean((model.apply(p, sensor_data)[0] - motor_targets) ** 2)
            )(params)
        end_time = time.perf_counter()
        
        avg_time_ms = (end_time - start_time) * 1000 / 100
        
        # Performance regression threshold: should be < 10ms per iteration
        assert avg_time_ms < 10.0, f"Training too slow: {avg_time_ms:.2f}ms > 10ms"

    @pytest.mark.benchmark
    def test_inference_speed_benchmark(self, basic_config, rng_key, sample_batch):
        """Benchmark inference speed."""
        model = LiquidNN(basic_config)
        params = model.init(rng_key, sample_batch[:1])
        
        # Warmup
        for _ in range(10):
            _ = model.apply(params, sample_batch)
        
        # Benchmark
        start_time = time.perf_counter()
        for _ in range(1000):
            output, _ = model.apply(params, sample_batch)
        end_time = time.perf_counter()
        
        avg_time_us = (end_time - start_time) * 1_000_000 / 1000
        throughput = len(sample_batch) * 1000 / ((end_time - start_time))
        
        # Performance thresholds
        assert avg_time_us < 500.0, f"Inference too slow: {avg_time_us:.2f}μs > 500μs"
        assert throughput > 10000, f"Throughput too low: {throughput:.0f} samples/s < 10k/s"

    @pytest.mark.benchmark
    def test_memory_usage_benchmark(self, basic_config, rng_key, sample_input):
        """Benchmark memory usage."""
        model = LiquidNN(basic_config)
        params = model.init(rng_key, sample_input)
        
        # Calculate parameter count
        param_count = sum(
            param.size for param in jax.tree_util.tree_leaves(params)
        )
        
        # Memory thresholds for edge deployment
        assert param_count < 50000, f"Too many parameters: {param_count} > 50k"
        
        # Estimate memory usage (4 bytes per float32 parameter)
        memory_kb = param_count * 4 / 1024
        assert memory_kb < 200, f"Memory usage too high: {memory_kb:.1f}KB > 200KB"

    @pytest.mark.benchmark
    def test_compilation_time_benchmark(self, basic_config, rng_key, sample_input):
        """Benchmark JAX compilation time."""
        model = LiquidNN(basic_config)
        params = model.init(rng_key, sample_input)
        
        @jax.jit
        def compiled_inference(params, x):
            return model.apply(params, x)
        
        # Measure compilation time
        start_time = time.perf_counter()
        _ = compiled_inference(params, sample_input)
        compilation_time = time.perf_counter() - start_time
        
        # Compilation should be fast for development
        assert compilation_time < 5.0, f"Compilation too slow: {compilation_time:.2f}s > 5s"

    @pytest.mark.benchmark
    @pytest.mark.parametrize("sparsity", [0.0, 0.3, 0.5, 0.7, 0.9])
    def test_sparsity_performance(self, rng_key, sample_input, sparsity):
        """Benchmark performance across different sparsity levels."""
        config = LiquidConfig(
            input_dim=4,
            hidden_dim=16,
            output_dim=2,
            use_sparse=True,
            sparsity=sparsity
        )
        
        model = LiquidNN(config)
        params = model.init(rng_key, sample_input)
        
        # Warmup
        for _ in range(5):
            _ = model.apply(params, sample_input)
        
        # Benchmark
        start_time = time.perf_counter()
        for _ in range(100):
            output = model.apply(params, sample_input)
        end_time = time.perf_counter()
        
        avg_time_us = (end_time - start_time) * 1_000_000 / 100
        
        # Higher sparsity should generally be faster
        expected_max_time = 200 + (1 - sparsity) * 300  # Adaptive threshold
        assert avg_time_us < expected_max_time, (
            f"Sparsity {sparsity}: {avg_time_us:.1f}μs > {expected_max_time:.1f}μs"
        )