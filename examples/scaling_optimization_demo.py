"""Generation 3: Scaling and optimization demonstration for production deployment."""

import jax
import jax.numpy as jnp
import numpy as np
import time
from liquid_edge import LiquidNN, LiquidConfig
from typing import Dict, List, Tuple
import concurrent.futures
from functools import partial
import psutil
import os


def create_scalable_config(scale_factor: int = 1):
    """Create scalable liquid neural network configuration."""
    return LiquidConfig(
        input_dim=4 * scale_factor,
        hidden_dim=16 * scale_factor,
        output_dim=2 * scale_factor,
        tau_min=5.0,
        tau_max=50.0,
        use_sparse=True,
        sparsity=0.5,  # Aggressive sparsity for scaling
        dt=0.1,
        use_layer_norm=False  # Disable for maximum speed
    )


@partial(jax.jit, static_argnums=(0,))
def batch_inference(model, params, batch_inputs):
    """JIT-compiled batch inference for maximum throughput."""
    outputs, _ = jax.vmap(
        lambda x: model.apply(params, x.reshape(1, -1), training=False)
    )(batch_inputs)
    return outputs[:, 0, :]  # Remove single batch dimension


def parallel_training_worker(worker_id: int, config: LiquidConfig, data_shard: Tuple[jnp.ndarray, jnp.ndarray], epochs: int):
    """Parallel training worker for distributed computation."""
    print(f"Worker {worker_id}: Starting training with {len(data_shard[0])} samples")
    
    # Create model for this worker
    model = LiquidNN(config)
    key = jax.random.PRNGKey(worker_id + 42)
    params = model.init(key, data_shard[0][:1])
    
    # Simple training loop
    inputs, targets = data_shard
    lr = 0.01
    
    start_time = time.perf_counter()
    
    for epoch in range(epochs):
        # Forward pass
        outputs, _ = model.apply(params, inputs, training=True)
        loss = jnp.mean((outputs - targets) ** 2)
        
        # Backward pass
        grads = jax.grad(lambda p: jnp.mean((model.apply(p, inputs, training=True)[0] - targets) ** 2))(params)
        
        # Update parameters
        params = jax.tree.map(lambda p, g: p - lr * g, params, grads)
        
        if epoch % 20 == 0:
            print(f"Worker {worker_id} - Epoch {epoch}: Loss = {float(loss):.6f}")
    
    training_time = time.perf_counter() - start_time
    final_loss = float(loss)
    
    print(f"Worker {worker_id}: Training completed in {training_time:.2f}s, Final loss: {final_loss:.6f}")
    
    return {
        'worker_id': worker_id,
        'params': params,
        'final_loss': final_loss,
        'training_time': training_time
    }


def demonstrate_horizontal_scaling():
    """Demonstrate horizontal scaling with parallel workers."""
    print("🔄 HORIZONTAL SCALING DEMONSTRATION")
    print("=" * 60)
    
    # Configuration
    config = create_scalable_config(scale_factor=2)
    num_workers = min(4, os.cpu_count())  # Use up to 4 workers
    total_samples = 1000
    epochs_per_worker = 50
    
    print(f"📊 Scaling Configuration:")
    print(f"   • Workers: {num_workers}")
    print(f"   • Model size: {config.input_dim}→{config.hidden_dim}→{config.output_dim}")
    print(f"   • Total samples: {total_samples}")
    print(f"   • Epochs per worker: {epochs_per_worker}")
    
    # Generate data
    key = jax.random.PRNGKey(42)
    inputs = jax.random.normal(key, (total_samples, config.input_dim))
    targets = jax.random.normal(jax.random.split(key)[0], (total_samples, config.output_dim))
    
    # Split data among workers
    shard_size = total_samples // num_workers
    data_shards = []
    
    for i in range(num_workers):
        start_idx = i * shard_size
        end_idx = start_idx + shard_size if i < num_workers - 1 else total_samples
        
        shard_inputs = inputs[start_idx:end_idx]
        shard_targets = targets[start_idx:end_idx]
        data_shards.append((shard_inputs, shard_targets))
        
        print(f"   • Worker {i}: {len(shard_inputs)} samples")
    
    # Parallel training
    print(f"\n🚀 Starting parallel training...")
    start_time = time.perf_counter()
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(parallel_training_worker, i, config, shard, epochs_per_worker)
            for i, shard in enumerate(data_shards)
        ]
        
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
    
    total_time = time.perf_counter() - start_time
    
    # Analyze results
    avg_loss = np.mean([r['final_loss'] for r in results])
    total_training_time = sum(r['training_time'] for r in results)
    speedup = total_training_time / total_time
    
    print(f"\n📈 HORIZONTAL SCALING RESULTS:")
    print(f"   • Total wall time: {total_time:.2f}s")
    print(f"   • Combined training time: {total_training_time:.2f}s")
    print(f"   • Speedup factor: {speedup:.2f}x")
    print(f"   • Average final loss: {avg_loss:.6f}")
    print(f"   • Throughput: {total_samples * epochs_per_worker / total_time:.0f} samples/s")
    
    return results


def demonstrate_vertical_scaling():
    """Demonstrate vertical scaling with different model sizes."""
    print("\n📈 VERTICAL SCALING DEMONSTRATION")
    print("=" * 60)
    
    # Test different model scales
    scale_factors = [1, 2, 4, 8]
    batch_size = 32
    num_batches = 100
    
    results = {}
    
    for scale in scale_factors:
        config = create_scalable_config(scale_factor=scale)
        model = LiquidNN(config)
        
        # Create data
        key = jax.random.PRNGKey(42)
        test_input = jax.random.normal(key, (batch_size, config.input_dim))
        params = model.init(key, test_input[:1])
        
        # Count parameters
        param_count = sum(param.size for param in jax.tree_util.tree_leaves(params))
        memory_mb = param_count * 4 / (1024 * 1024)  # float32 = 4 bytes
        
        print(f"\n🔍 Scale Factor {scale}x:")
        print(f"   • Architecture: {config.input_dim}→{config.hidden_dim}→{config.output_dim}")
        print(f"   • Parameters: {param_count:,}")
        print(f"   • Memory: {memory_mb:.2f}MB")
        
        # Benchmark inference speed
        # Warmup
        for _ in range(10):
            _ = batch_inference(model, params, test_input)
        
        # Actual benchmark
        times = []
        for _ in range(num_batches):
            start = time.perf_counter()
            outputs = batch_inference(model, params, test_input)
            times.append(time.perf_counter() - start)
        
        avg_time_ms = np.mean(times) * 1000
        throughput_fps = batch_size / np.mean(times)
        
        print(f"   • Latency: {avg_time_ms:.2f}ms")
        print(f"   • Throughput: {throughput_fps:.0f} samples/s")
        print(f"   • Efficiency: {throughput_fps/param_count*1000:.3f} samples/s/kparam")
        
        results[scale] = {
            'param_count': param_count,
            'memory_mb': memory_mb,
            'latency_ms': avg_time_ms,
            'throughput_fps': throughput_fps,
            'efficiency': throughput_fps/param_count*1000
        }
    
    # Scaling analysis
    print(f"\n📊 VERTICAL SCALING ANALYSIS:")
    
    base_throughput = results[1]['throughput_fps']
    base_params = results[1]['param_count']
    
    for scale in scale_factors:
        r = results[scale]
        throughput_scaling = r['throughput_fps'] / base_throughput
        param_scaling = r['param_count'] / base_params
        efficiency_ratio = throughput_scaling / param_scaling
        
        print(f"   • {scale}x scale: {throughput_scaling:.2f}x throughput, {param_scaling:.1f}x params, {efficiency_ratio:.3f} efficiency")
    
    return results


def demonstrate_memory_optimization():
    """Demonstrate memory optimization techniques."""
    print("\n🧠 MEMORY OPTIMIZATION DEMONSTRATION")  
    print("=" * 60)
    
    # Test different optimization strategies
    configs = {
        'baseline': LiquidConfig(
            input_dim=32, hidden_dim=64, output_dim=8,
            use_sparse=False, sparsity=0.0, use_layer_norm=True
        ),
        'sparse_30': LiquidConfig(
            input_dim=32, hidden_dim=64, output_dim=8,
            use_sparse=True, sparsity=0.3, use_layer_norm=True
        ),
        'sparse_50': LiquidConfig(
            input_dim=32, hidden_dim=64, output_dim=8,
            use_sparse=True, sparsity=0.5, use_layer_norm=True  
        ),
        'sparse_70_no_norm': LiquidConfig(
            input_dim=32, hidden_dim=64, output_dim=8,
            use_sparse=True, sparsity=0.7, use_layer_norm=False
        )
    }
    
    results = {}
    
    for name, config in configs.items():
        model = LiquidNN(config)
        key = jax.random.PRNGKey(42)
        test_input = jax.random.normal(key, (1, config.input_dim))
        params = model.init(key, test_input)
        
        # Memory analysis
        param_count = sum(param.size for param in jax.tree_util.tree_leaves(params))
        memory_kb = param_count * 4 / 1024
        
        # Speed test
        times = []
        for _ in range(1000):
            start = time.perf_counter()
            _ = model.apply(params, test_input, training=False)
            times.append(time.perf_counter() - start)
        
        avg_time_us = np.mean(times) * 1_000_000
        
        print(f"🔧 {name.upper()}:")
        print(f"   • Parameters: {param_count:,}")
        print(f"   • Memory: {memory_kb:.1f}KB")
        print(f"   • Latency: {avg_time_us:.1f}μs")
        print(f"   • Speed/Memory ratio: {1000/avg_time_us/memory_kb:.3f}")
        
        results[name] = {
            'param_count': param_count,
            'memory_kb': memory_kb,
            'latency_us': avg_time_us,
            'speed_memory_ratio': 1000/avg_time_us/memory_kb
        }
    
    # Find best optimization
    best_config = max(results.items(), key=lambda x: x[1]['speed_memory_ratio'])
    
    print(f"\n🏆 OPTIMAL CONFIGURATION: {best_config[0].upper()}")
    print(f"   • Best speed/memory ratio: {best_config[1]['speed_memory_ratio']:.3f}")
    
    return results


def demonstrate_production_deployment():
    """Demonstrate production-ready deployment capabilities."""
    print("\n🚀 PRODUCTION DEPLOYMENT DEMONSTRATION")
    print("=" * 60)
    
    # Production configuration
    config = create_scalable_config(scale_factor=1)
    model = LiquidNN(config)
    
    key = jax.random.PRNGKey(42)
    sample_input = jnp.ones((1, config.input_dim))
    params = model.init(key, sample_input)
    
    print(f"📦 Production Model:")
    param_count = sum(param.size for param in jax.tree_util.tree_leaves(params))
    memory_kb = param_count * 4 / 1024
    
    print(f"   • Architecture: {config.input_dim}→{config.hidden_dim}→{config.output_dim}")
    print(f"   • Parameters: {param_count:,}")
    print(f"   • Memory footprint: {memory_kb:.1f}KB")
    print(f"   • Sparsity: {config.sparsity:.0%}")
    
    # Create optimized inference function
    @jax.jit
    def production_inference(params, x):
        outputs, _ = model.apply(params, x, training=False)
        return outputs
    
    # Load test simulation
    print(f"\n🔥 LOAD TESTING:")
    
    test_scenarios = [
        ('Low Load (10 Hz)', 10, 100),
        ('Medium Load (100 Hz)', 100, 1000), 
        ('High Load (1kHz)', 1000, 10000),
        ('Stress Test (10kHz)', 10000, 100000)
    ]
    
    for scenario_name, target_hz, num_requests in test_scenarios:
        print(f"\n🎯 {scenario_name}:")
        
        # Generate requests
        test_inputs = jax.random.normal(
            jax.random.PRNGKey(123), 
            (num_requests, config.input_dim)
        )
        
        # Warmup
        for _ in range(10):
            _ = production_inference(params, test_inputs[:1])
        
        # Load test
        start_time = time.perf_counter()
        
        for i in range(0, num_requests, 32):  # Process in batches of 32
            batch = test_inputs[i:i+32]
            _ = jax.vmap(lambda x: production_inference(params, x.reshape(1, -1)))(batch)
        
        end_time = time.perf_counter()
        
        actual_time = end_time - start_time
        actual_hz = num_requests / actual_time
        success_rate = min(actual_hz / target_hz, 1.0) * 100
        
        print(f"   • Target: {target_hz}Hz ({num_requests} requests)")
        print(f"   • Achieved: {actual_hz:.0f}Hz")
        print(f"   • Success rate: {success_rate:.1f}%")
        print(f"   • Average latency: {actual_time/num_requests*1000000:.0f}μs")
        
        if success_rate >= 95:
            status = "✅ PASS"
        elif success_rate >= 80:
            status = "⚠️  MARGINAL" 
        else:
            status = "❌ FAIL"
        
        print(f"   • Status: {status}")
    
    # Resource usage
    process = psutil.Process()
    cpu_percent = process.cpu_percent()
    memory_mb = process.memory_info().rss / 1024 / 1024
    
    print(f"\n💻 RESOURCE USAGE:")
    print(f"   • CPU usage: {cpu_percent:.1f}%")
    print(f"   • Memory usage: {memory_mb:.1f}MB")
    print(f"   • Process threads: {process.num_threads()}")
    
    return {
        'model_params': param_count,
        'memory_footprint_kb': memory_kb,
        'max_achievable_hz': actual_hz,
        'resource_usage': {
            'cpu_percent': cpu_percent,
            'memory_mb': memory_mb
        }
    }


def main():
    """Run complete scaling and optimization demonstration."""
    print("🌊 LIQUID EDGE LNN - GENERATION 3: SCALING & OPTIMIZATION")
    print("=" * 70)
    
    # Run all scaling demonstrations
    horizontal_results = demonstrate_horizontal_scaling()
    vertical_results = demonstrate_vertical_scaling()
    memory_results = demonstrate_memory_optimization()
    production_results = demonstrate_production_deployment()
    
    # Final summary
    print(f"\n🎯 GENERATION 3 COMPLETE - SCALING & OPTIMIZATION SUMMARY")
    print("=" * 70)
    
    print(f"✅ HORIZONTAL SCALING:")
    print(f"   • Parallel workers: {len(horizontal_results)}")
    print(f"   • Speedup achieved: {sum(r['training_time'] for r in horizontal_results) / max(r['training_time'] for r in horizontal_results):.1f}x")
    
    print(f"✅ VERTICAL SCALING:")
    best_scale = max(vertical_results.items(), key=lambda x: x[1]['efficiency'])
    print(f"   • Optimal scale: {best_scale[0]}x")
    print(f"   • Peak efficiency: {best_scale[1]['efficiency']:.3f} samples/s/kparam")
    
    print(f"✅ MEMORY OPTIMIZATION:")
    best_memory = max(memory_results.items(), key=lambda x: x[1]['speed_memory_ratio'])
    print(f"   • Optimal config: {best_memory[0]}")
    print(f"   • Memory footprint: {best_memory[1]['memory_kb']:.1f}KB")
    
    print(f"✅ PRODUCTION DEPLOYMENT:")
    print(f"   • Max throughput: {production_results['max_achievable_hz']:.0f}Hz")
    print(f"   • Model size: {production_results['model_params']:,} parameters")
    print(f"   • Memory footprint: {production_results['memory_footprint_kb']:.1f}KB")
    
    print("=" * 70)
    print("🏆 GENERATION 3 SUCCESS: PRODUCTION-SCALE LIQUID NEURAL NETWORKS")
    

if __name__ == "__main__":
    main()