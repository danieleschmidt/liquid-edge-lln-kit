"""Ultra-fast liquid neural network training example."""

import jax
import jax.numpy as jnp
from liquid_edge import LiquidNN, LiquidConfig
import time
import numpy as np
from functools import partial


def create_ultra_fast_config():
    """Create configuration optimized for speed."""
    return LiquidConfig(
        input_dim=4,
        hidden_dim=8,
        output_dim=2,
        tau_min=10.0,
        tau_max=100.0,
        use_sparse=True,
        sparsity=0.5,  # Higher sparsity for speed
        dt=0.2,        # Larger time step for faster integration
        use_layer_norm=False,  # Disable layer norm for speed
        energy_budget_mw=100.0,
        target_fps=50
    )


def create_ultra_fast_train_step(model):
    """Create JIT-compiled training step for a specific model."""
    
    @jax.jit
    def train_step(params, optimizer_state, inputs, targets):
        """JIT-compiled ultra-fast training step."""
        
        def loss_fn(p):
            outputs, _ = model.apply(p, inputs, training=True)
            return jnp.mean((outputs - targets) ** 2)
        
        loss, grads = jax.value_and_grad(loss_fn)(params)
        
        # Simplified optimizer update (no momentum for speed)
        lr = 0.01
        params = jax.tree.map(lambda p, g: p - lr * g, params, grads)
        
        return params, optimizer_state, loss
    
    return train_step


def benchmark_ultra_fast_training():
    """Benchmark ultra-fast training implementation."""
    print("🚀 ULTRA-FAST LIQUID NEURAL NETWORK TRAINING")
    print("=" * 50)
    
    # Create ultra-fast configuration
    config = create_ultra_fast_config()
    model = LiquidNN(config)
    
    # Generate sample data
    key = jax.random.PRNGKey(42)
    inputs = jax.random.normal(key, (32, 4))  # 32 samples
    targets = jax.random.normal(jax.random.split(key)[1], (32, 2))
    
    # Initialize model
    params = model.init(key, inputs[:1])
    optimizer_state = {}  # Dummy state for simplified optimizer
    
    # Create JIT-compiled training function
    train_step = create_ultra_fast_train_step(model)
    
    # Warmup JIT compilation
    print("Warming up JIT compilation...")
    for _ in range(5):
        params, optimizer_state, loss = train_step(
            params, optimizer_state, inputs, targets
        )
    
    # Benchmark training speed
    print("Benchmarking training speed...")
    num_iterations = 1000
    
    start_time = time.perf_counter()
    
    for i in range(num_iterations):
        params, optimizer_state, loss = train_step(
            params, optimizer_state, inputs, targets
        )
    
    end_time = time.perf_counter()
    
    # Calculate performance metrics
    total_time_ms = (end_time - start_time) * 1000
    avg_time_ms = total_time_ms / num_iterations
    throughput = num_iterations / (end_time - start_time)
    
    print(f"✅ Training completed!")
    print(f"📊 PERFORMANCE RESULTS:")
    print(f"   • Total time: {total_time_ms:.2f}ms")
    print(f"   • Average per iteration: {avg_time_ms:.3f}ms")
    print(f"   • Throughput: {throughput:.0f} iterations/sec")
    print(f"   • Final loss: {float(loss):.6f}")
    
    # Performance targets for edge deployment
    target_time_ms = 5.0  # Target: <5ms per training step
    
    if avg_time_ms < target_time_ms:
        print(f"🎯 TARGET MET: {avg_time_ms:.3f}ms < {target_time_ms}ms")
    else:
        print(f"⚠️  TARGET MISSED: {avg_time_ms:.3f}ms > {target_time_ms}ms")
    
    return {
        'avg_time_ms': avg_time_ms,
        'throughput': throughput,
        'final_loss': float(loss)
    }


def create_ultra_fast_inference(model):
    """Create JIT-compiled inference function."""
    
    @jax.jit
    def inference_fn(params, inputs):
        outputs, hidden = model.apply(params, inputs, training=False)
        return outputs
    
    return inference_fn


def benchmark_inference_speed():
    """Benchmark inference speed for real-time applications."""
    print("\n🔥 ULTRA-FAST INFERENCE BENCHMARK")
    print("=" * 50)
    
    # Create model for inference
    config = create_ultra_fast_config()
    model = LiquidNN(config)
    
    key = jax.random.PRNGKey(42)
    sample_input = jax.random.normal(key, (1, 4))  # Single sample
    params = model.init(key, sample_input)
    
    # Create JIT-compiled inference function
    inference_fn = create_ultra_fast_inference(model)
    
    # Warmup
    for _ in range(10):
        _ = inference_fn(params, sample_input)
    
    # Benchmark inference
    num_inferences = 10000
    start_time = time.perf_counter()
    
    for _ in range(num_inferences):
        outputs = inference_fn(params, sample_input)
    
    end_time = time.perf_counter()
    
    # Calculate metrics
    avg_time_us = (end_time - start_time) * 1_000_000 / num_inferences
    throughput_fps = num_inferences / (end_time - start_time)
    
    print(f"📊 INFERENCE RESULTS:")
    print(f"   • Average latency: {avg_time_us:.1f}μs")
    print(f"   • Throughput: {throughput_fps:.0f} FPS")
    
    # Real-time targets
    target_latency_us = 200.0  # Target: <200μs (5kHz capable)
    target_fps = 1000  # Target: >1000 FPS
    
    if avg_time_us < target_latency_us:
        print(f"🎯 LATENCY TARGET MET: {avg_time_us:.1f}μs < {target_latency_us}μs")
    else:
        print(f"⚠️  LATENCY TARGET MISSED: {avg_time_us:.1f}μs > {target_latency_us}μs")
    
    if throughput_fps > target_fps:
        print(f"🎯 THROUGHPUT TARGET MET: {throughput_fps:.0f}fps > {target_fps}fps")
    else:
        print(f"⚠️  THROUGHPUT TARGET MISSED: {throughput_fps:.0f}fps < {target_fps}fps")
    
    return {
        'latency_us': avg_time_us,
        'throughput_fps': throughput_fps
    }


def memory_efficiency_analysis():
    """Analyze memory usage of optimized implementation."""
    print("\n🧠 MEMORY EFFICIENCY ANALYSIS")
    print("=" * 50)
    
    config = create_ultra_fast_config()
    model = LiquidNN(config)
    
    key = jax.random.PRNGKey(42)
    sample_input = jax.random.normal(key, (1, 4))
    params = model.init(key, sample_input)
    
    # Count parameters
    param_count = sum(
        param.size for param in jax.tree_util.tree_leaves(params)
    )
    
    # Estimate memory usage
    memory_bytes = param_count * 4  # float32 = 4 bytes
    memory_kb = memory_bytes / 1024
    
    print(f"📊 MEMORY ANALYSIS:")
    print(f"   • Parameter count: {param_count:,}")
    print(f"   • Memory usage: {memory_kb:.1f}KB")
    print(f"   • Memory per neuron: {memory_kb/config.hidden_dim:.1f}KB")
    
    # Edge deployment targets
    target_memory_kb = 64  # Target: <64KB for MCU deployment
    
    if memory_kb < target_memory_kb:
        print(f"🎯 MEMORY TARGET MET: {memory_kb:.1f}KB < {target_memory_kb}KB")
    else:
        print(f"⚠️  MEMORY TARGET MISSED: {memory_kb:.1f}KB > {target_memory_kb}KB")
    
    return {
        'param_count': param_count,
        'memory_kb': memory_kb
    }


if __name__ == "__main__":
    print("🌊 LIQUID EDGE LNN - ULTRA-FAST PERFORMANCE ANALYSIS")
    print("=" * 60)
    
    # Run all benchmarks
    training_results = benchmark_ultra_fast_training()
    inference_results = benchmark_inference_speed()
    memory_results = memory_efficiency_analysis()
    
    print(f"\n🏆 FINAL PERFORMANCE SUMMARY")
    print("=" * 60)
    print(f"Training: {training_results['avg_time_ms']:.3f}ms/iter @ {training_results['throughput']:.0f} iter/s")
    print(f"Inference: {inference_results['latency_us']:.1f}μs @ {inference_results['throughput_fps']:.0f} FPS")
    print(f"Memory: {memory_results['memory_kb']:.1f}KB ({memory_results['param_count']:,} params)")
    print("=" * 60)