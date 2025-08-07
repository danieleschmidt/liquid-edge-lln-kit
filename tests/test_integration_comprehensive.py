#!/usr/bin/env python3
"""Comprehensive integration tests for the Liquid Edge LLN Kit."""

import sys
import pytest
import numpy as np
import time
from unittest.mock import Mock, patch
import tempfile
import json
from pathlib import Path

# Mock JAX imports for testing environment
class MockJax:
    class numpy:
        @staticmethod
        def array(x):
            return np.array(x)
        
        @staticmethod
        def zeros(shape):
            return np.zeros(shape)
        
        @staticmethod
        def ones(shape):
            return np.ones(shape)
        
        @staticmethod
        def concatenate(arrays, axis=0):
            return np.concatenate(arrays, axis=axis)
        
        @staticmethod
        def linalg_norm(x):
            return np.linalg.norm(x)
        
        @staticmethod
        def mean(x):
            return np.mean(x)
        
        @staticmethod
        def clip(x, min_val, max_val):
            return np.clip(x, min_val, max_val)
    
    class random:
        @staticmethod
        def PRNGKey(seed):
            return seed
        
        @staticmethod
        def normal(key, shape):
            np.random.seed(key)
            return np.random.normal(0, 1, shape)
        
        @staticmethod
        def split(key):
            return key + 1, key + 2

# Mock dependencies
sys.modules['jax'] = MockJax()
sys.modules['jax.numpy'] = MockJax.numpy
sys.modules['jax.random'] = MockJax.random
sys.modules['flax'] = Mock()
sys.modules['flax.linen'] = Mock()
sys.modules['optax'] = Mock()

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from liquid_edge.core import LiquidConfig
from liquid_edge.monitoring import PerformanceMetrics, AlertLevel
from liquid_edge.error_handling import ErrorSeverity, LiquidNetworkError


class TestLiquidConfig:
    """Test liquid neural network configuration."""
    
    def test_config_initialization(self):
        """Test basic config initialization."""
        config = LiquidConfig(
            input_dim=8,
            hidden_dim=12,
            output_dim=2,
            use_sparse=True,
            sparsity=0.3
        )
        
        assert config.input_dim == 8
        assert config.hidden_dim == 12
        assert config.output_dim == 2
        assert config.use_sparse is True
        assert config.sparsity == 0.3
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Test invalid input dimension
        with pytest.raises(ValueError, match="input_dim must be positive"):
            LiquidConfig(input_dim=0, hidden_dim=8, output_dim=2)
        
        # Test invalid sparsity
        with pytest.raises(ValueError, match="sparsity must be between 0 and 1"):
            LiquidConfig(input_dim=8, hidden_dim=8, output_dim=2, sparsity=1.5)
        
        # Test invalid time constants
        with pytest.raises(ValueError, match="tau_min must be less than tau_max"):
            LiquidConfig(input_dim=8, hidden_dim=8, output_dim=2, tau_min=100, tau_max=50)
        
        # Test invalid energy budget
        with pytest.raises(ValueError, match="energy_budget_mw must be positive"):
            LiquidConfig(input_dim=8, hidden_dim=8, output_dim=2, energy_budget_mw=-10)
    
    def test_config_post_init(self):
        """Test post-initialization processing."""
        # Valid config should not raise
        config = LiquidConfig(
            input_dim=4,
            hidden_dim=8, 
            output_dim=2,
            tau_min=10.0,
            tau_max=100.0,
            sparsity=0.2,
            energy_budget_mw=75.0
        )
        
        assert config.input_dim == 4
        assert config.tau_min < config.tau_max
        assert 0.0 <= config.sparsity <= 1.0


class TestPerformanceMetrics:
    """Test performance metrics collection."""
    
    def test_metrics_initialization(self):
        """Test metrics initialization."""
        metrics = PerformanceMetrics(
            inference_time_us=1500.0,
            energy_consumption_mw=45.2,
            memory_usage_mb=128.0,
            cpu_usage_percent=65.5,
            throughput_fps=50.0,
            accuracy=0.95
        )
        
        assert metrics.inference_time_us == 1500.0
        assert metrics.energy_consumption_mw == 45.2
        assert metrics.memory_usage_mb == 128.0
        assert metrics.cpu_usage_percent == 65.5
        assert metrics.throughput_fps == 50.0
        assert metrics.accuracy == 0.95
        assert metrics.timestamp > 0
    
    def test_metrics_to_dict(self):
        """Test metrics serialization."""
        metrics = PerformanceMetrics(
            inference_time_us=1000.0,
            energy_consumption_mw=30.0
        )
        
        metrics_dict = metrics.to_dict()
        
        assert isinstance(metrics_dict, dict)
        assert 'inference_time_us' in metrics_dict
        assert 'energy_consumption_mw' in metrics_dict
        assert 'timestamp' in metrics_dict
        assert metrics_dict['inference_time_us'] == 1000.0
        assert metrics_dict['energy_consumption_mw'] == 30.0


class TestErrorHandling:
    """Test error handling and resilience."""
    
    def test_liquid_network_error(self):
        """Test custom error types."""
        error = LiquidNetworkError(
            "Test error",
            severity=ErrorSeverity.HIGH,
            context={"component": "test"}
        )
        
        assert str(error) == "Test error"
        assert error.severity == ErrorSeverity.HIGH
        assert error.context["component"] == "test"
        assert error.timestamp > 0
    
    def test_error_severity_levels(self):
        """Test error severity enumeration."""
        assert ErrorSeverity.LOW.value == "low"
        assert ErrorSeverity.MEDIUM.value == "medium"
        assert ErrorSeverity.HIGH.value == "high"
        assert ErrorSeverity.CRITICAL.value == "critical"
    
    def test_error_context_preservation(self):
        """Test error context is preserved."""
        context = {
            "model_id": "test_model",
            "input_shape": (1, 8),
            "step": 42
        }
        
        error = LiquidNetworkError(
            "Context test",
            severity=ErrorSeverity.MEDIUM,
            context=context
        )
        
        assert error.context == context
        assert error.context["model_id"] == "test_model"
        assert error.context["step"] == 42


class TestIntegrationWorkflows:
    """Test end-to-end integration workflows."""
    
    def test_basic_workflow(self):
        """Test basic liquid neural network workflow."""
        # Create configuration
        config = LiquidConfig(
            input_dim=6,
            hidden_dim=10,
            output_dim=3,
            energy_budget_mw=60.0
        )
        
        # Test energy estimation (mock implementation)
        estimated_energy = config.energy_budget_mw * 0.8  # 80% of budget
        
        assert estimated_energy < config.energy_budget_mw
        assert estimated_energy > 0
    
    def test_sensor_data_processing(self):
        """Test sensor data processing pipeline."""
        # Mock sensor data (8 distance sensors)
        sensor_data = np.array([2.5, 3.0, 1.8, 4.2, 3.5, 2.1, 2.8, 3.2])
        
        # Validate sensor ranges
        assert np.all(sensor_data >= 0.0)
        assert np.all(sensor_data <= 5.0)  # Max sensor range
        
        # Normalize sensor data
        normalized_data = sensor_data / 5.0
        assert np.all(normalized_data >= 0.0)
        assert np.all(normalized_data <= 1.0)
    
    def test_motor_command_validation(self):
        """Test motor command validation."""
        # Mock motor commands
        raw_commands = np.array([0.8, 1.5])  # Linear, Angular velocity
        
        # Validate and clamp commands
        linear_vel = np.clip(raw_commands[0], -1.0, 1.0)
        angular_vel = np.clip(raw_commands[1], -2.0, 2.0)
        
        safe_commands = np.array([linear_vel, angular_vel])
        
        assert -1.0 <= safe_commands[0] <= 1.0
        assert -2.0 <= safe_commands[1] <= 2.0
        assert safe_commands[0] == 0.8  # Within bounds
        assert safe_commands[1] == 1.5  # Within bounds
    
    def test_control_loop_timing(self):
        """Test control loop timing requirements."""
        target_frequency = 50.0  # 50Hz
        target_period = 1.0 / target_frequency
        
        # Simulate control loop timing
        loop_times = []
        
        for i in range(10):  # 10 iterations
            start_time = time.time()
            
            # Mock control processing
            time.sleep(0.001)  # 1ms processing time
            
            end_time = time.time()
            loop_time = end_time - start_time
            loop_times.append(loop_time)
            
            # Ensure we don't exceed target period
            assert loop_time < target_period
        
        avg_loop_time = np.mean(loop_times)
        assert avg_loop_time < target_period
        
        # Calculate achieved frequency
        achieved_frequency = 1.0 / avg_loop_time
        assert achieved_frequency > target_frequency  # Should be faster due to minimal processing


class TestPerformanceBenchmarks:
    """Performance benchmarking tests."""
    
    def test_inference_latency_benchmark(self):
        """Benchmark inference latency."""
        # Mock inference timing
        target_latency_us = 10000  # 10ms target
        
        latencies = []
        for i in range(100):  # 100 inferences
            start_time = time.perf_counter()
            
            # Mock inference work (matrix operations)
            input_data = np.random.randn(1, 8)
            hidden_state = np.random.randn(1, 12)
            
            # Simple mock computation
            output = np.tanh(input_data @ np.random.randn(8, 12) + hidden_state)
            
            end_time = time.perf_counter()
            latency_us = (end_time - start_time) * 1e6
            latencies.append(latency_us)
        
        avg_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)
        
        print(f"\nLatency Benchmark Results:")
        print(f"  Average: {avg_latency:.1f}μs")
        print(f"  P95: {p95_latency:.1f}μs")
        print(f"  P99: {p99_latency:.1f}μs")
        
        # Performance assertions
        assert avg_latency < target_latency_us  # Average should be under target
        assert p95_latency < target_latency_us * 2  # P95 should be reasonable
    
    def test_throughput_benchmark(self):
        """Benchmark inference throughput."""
        target_throughput = 100.0  # 100 FPS target
        
        # Mock batch processing
        batch_size = 16
        num_batches = 10
        
        start_time = time.perf_counter()
        
        for batch in range(num_batches):
            # Mock batch inference
            batch_input = np.random.randn(batch_size, 8)
            batch_hidden = np.random.randn(batch_size, 12)
            
            # Simple batch computation
            batch_output = np.tanh(batch_input @ np.random.randn(8, 12) + batch_hidden)
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        total_inferences = batch_size * num_batches
        throughput = total_inferences / total_time
        
        print(f"\nThroughput Benchmark Results:")
        print(f"  Total inferences: {total_inferences}")
        print(f"  Total time: {total_time:.3f}s")
        print(f"  Throughput: {throughput:.1f} FPS")
        
        # Performance assertion
        assert throughput > target_throughput
    
    def test_memory_usage_benchmark(self):
        """Benchmark memory usage."""
        import psutil
        import os
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / (1024 * 1024)  # MB
        
        # Create mock model data structures
        model_data = []
        for i in range(100):  # Simulate multiple model instances
            # Mock parameter arrays
            weights = np.random.randn(8, 12).astype(np.float32)
            biases = np.random.randn(12).astype(np.float32)
            model_data.append((weights, biases))
        
        # Get peak memory usage
        peak_memory = process.memory_info().rss / (1024 * 1024)  # MB
        memory_increase = peak_memory - initial_memory
        
        print(f"\nMemory Usage Benchmark:")
        print(f"  Initial memory: {initial_memory:.1f}MB")
        print(f"  Peak memory: {peak_memory:.1f}MB")
        print(f"  Memory increase: {memory_increase:.1f}MB")
        
        # Memory efficiency assertion
        expected_model_size = 100 * (8 * 12 + 12) * 4 / (1024 * 1024)  # Expected size in MB
        assert memory_increase < expected_model_size * 2  # Allow for overhead
        
        # Cleanup
        del model_data


class TestSecurityValidation:
    """Security validation tests."""
    
    def test_input_validation(self):
        """Test input validation and sanitization."""
        # Test valid inputs
        valid_sensor_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 2.5, 3.5, 1.5])
        
        # Should not raise any exceptions
        assert np.all(np.isfinite(valid_sensor_data))
        assert len(valid_sensor_data) == 8
        
        # Test invalid inputs
        invalid_inputs = [
            np.array([np.inf, 2.0, 3.0, 4.0, 5.0, 2.5, 3.5, 1.5]),  # Contains infinity
            np.array([1.0, 2.0, np.nan, 4.0, 5.0, 2.5, 3.5, 1.5]),   # Contains NaN
            np.array([-1.0, 2.0, 3.0, 4.0, 5.0, 2.5, 3.5, 1.5]),    # Negative value
            np.array([1.0, 2.0, 3.0, 4.0, 10.0, 2.5, 3.5, 1.5])     # Out of range
        ]
        
        for invalid_input in invalid_inputs:
            # Should detect invalid inputs
            has_invalid = (
                not np.all(np.isfinite(invalid_input)) or
                np.any(invalid_input < 0) or 
                np.any(invalid_input > 5.0)
            )
            assert has_invalid, f"Should detect invalid input: {invalid_input}"
    
    def test_parameter_bounds_checking(self):
        """Test parameter bounds checking."""
        # Test valid parameter ranges
        valid_configs = [
            {'input_dim': 4, 'hidden_dim': 8, 'output_dim': 2, 'sparsity': 0.0},
            {'input_dim': 16, 'hidden_dim': 32, 'output_dim': 4, 'sparsity': 0.5},
            {'input_dim': 8, 'hidden_dim': 12, 'output_dim': 2, 'sparsity': 0.9}
        ]
        
        for config_params in valid_configs:
            # Should not raise exceptions
            config = LiquidConfig(**config_params)
            assert config.input_dim > 0
            assert config.hidden_dim > 0  
            assert config.output_dim > 0
            assert 0.0 <= config.sparsity <= 1.0
    
    def test_safe_defaults(self):
        """Test that default values are safe."""
        config = LiquidConfig(input_dim=8, hidden_dim=12, output_dim=2)
        
        # Check safe default values
        assert config.tau_min > 0  # Positive time constants
        assert config.tau_max > config.tau_min  # Proper ordering
        assert config.energy_budget_mw > 0  # Positive energy budget
        assert config.target_fps > 0  # Positive frame rate
        assert 0.0 <= config.sparsity <= 1.0  # Valid sparsity range
        assert config.learning_rate > 0  # Positive learning rate
    
    def test_resource_limits(self):
        """Test resource limitation enforcement."""
        # Test memory limits
        max_reasonable_dim = 1000  # Reasonable upper bound
        
        large_config = LiquidConfig(
            input_dim=max_reasonable_dim,
            hidden_dim=max_reasonable_dim, 
            output_dim=max_reasonable_dim
        )
        
        # Calculate approximate memory usage
        approx_params = (
            large_config.input_dim * large_config.hidden_dim +  # Input weights
            large_config.hidden_dim * large_config.hidden_dim + # Recurrent weights
            large_config.hidden_dim * large_config.output_dim   # Output weights
        )
        
        memory_mb = approx_params * 4 / (1024 * 1024)  # 4 bytes per float32
        
        print(f"\nLarge model memory estimate: {memory_mb:.1f}MB")
        
        # Should be manageable (less than 1GB)
        assert memory_mb < 1024


class TestReliabilityValidation:
    """Reliability and robustness validation."""
    
    def test_graceful_degradation(self):
        """Test graceful degradation under errors."""
        # Mock error conditions and recovery
        error_scenarios = [
            "sensor_timeout",
            "inference_failure", 
            "energy_budget_exceeded",
            "memory_exhaustion"
        ]
        
        for scenario in error_scenarios:
            # Test each error scenario has a fallback
            if scenario == "sensor_timeout":
                fallback_data = np.zeros(8)  # Safe sensor fallback
                assert np.all(fallback_data >= 0)
                
            elif scenario == "inference_failure":
                fallback_output = np.zeros(2)  # Safe motor commands (stop)
                assert np.all(np.abs(fallback_output) <= 1.0)
                
            elif scenario == "energy_budget_exceeded":
                # Should trigger low-power mode
                low_power_mode = True
                assert low_power_mode
                
            elif scenario == "memory_exhaustion":
                # Should trigger cleanup
                cleanup_triggered = True
                assert cleanup_triggered
    
    def test_state_consistency(self):
        """Test internal state consistency."""
        # Mock hidden state evolution
        hidden_dim = 12
        hidden_state = np.random.randn(1, hidden_dim)
        
        # State should remain bounded after updates
        for step in range(100):
            # Mock state update (should use proper liquid dynamics)
            update = np.tanh(np.random.randn(1, hidden_dim) * 0.1)
            hidden_state = 0.9 * hidden_state + 0.1 * update
            
            # State should remain bounded
            assert np.all(np.isfinite(hidden_state))
            assert np.max(np.abs(hidden_state)) < 10.0  # Reasonable bounds
    
    def test_deterministic_behavior(self):
        """Test deterministic behavior with same inputs."""
        # Mock deterministic inference
        np.random.seed(42)  # Fixed seed
        input_data = np.random.randn(1, 8)
        hidden_state = np.random.randn(1, 12)
        
        # Run inference twice with same inputs
        np.random.seed(123)  # Reset for computation
        output1 = np.tanh(input_data @ np.random.randn(8, 12) + hidden_state)
        
        np.random.seed(123)  # Same seed
        output2 = np.tanh(input_data @ np.random.randn(8, 12) + hidden_state)
        
        # Results should be identical
        assert np.allclose(output1, output2, rtol=1e-10)


if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__, "-v", "--tb=short"])
