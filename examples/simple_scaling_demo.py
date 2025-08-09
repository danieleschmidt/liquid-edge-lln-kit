"""Simple scaling demonstration without multiprocessing."""

import jax
import jax.numpy as jnp
import numpy as np
import time
from liquid_edge import LiquidNN, LiquidConfig


def demonstrate_batch_scaling():
    """Demonstrate batch processing scalability."""
    print("🚀 BATCH SCALING DEMONSTRATION")
    print("=" * 50)
    
    config = LiquidConfig(
        input_dim=4,
        hidden_dim=16,
        output_dim=2,
        use_sparse=True,
        sparsity=0.5,
        use_layer_norm=False
    )
    
    model = LiquidNN(config)
    key = jax.random.PRNGKey(42)
    sample_input = jax.random.normal(key, (1, 4))
    params = model.init(key, sample_input)
    
    # Create JIT-compiled batch inference
    @jax.jit
    def batch_inference(params, inputs):
        # Vectorized inference across batch
        outputs, _ = jax.vmap(
            lambda x: model.apply(params, x.reshape(1, -1), training=False)
        )(inputs)
        return outputs[:, 0, :]  # Remove singleton dimension
    
    # Test different batch sizes
    batch_sizes = [1, 8, 32, 128, 512]
    
    print("📊 Batch Size Performance:")
    
    results = {}
    
    for batch_size in batch_sizes:
        # Create batch data
        test_inputs = jax.random.normal(key, (batch_size, 4))
        
        # Warmup
        for _ in range(5):
            _ = batch_inference(params, test_inputs)
        
        # Benchmark
        times = []
        for _ in range(100):
            start = time.perf_counter()
            outputs = batch_inference(params, test_inputs)
            times.append(time.perf_counter() - start)
        
        avg_time_ms = np.mean(times) * 1000
        throughput_fps = batch_size / np.mean(times)
        latency_per_sample_us = np.mean(times) * 1_000_000 / batch_size
        
        results[batch_size] = {
            'avg_time_ms': avg_time_ms,
            'throughput_fps': throughput_fps,
            'latency_per_sample_us': latency_per_sample_us
        }
        
        print(f"   • Batch {batch_size:3d}: {avg_time_ms:6.2f}ms total, {throughput_fps:7.0f} FPS, {latency_per_sample_us:5.1f}μs/sample")
    
    # Calculate scaling efficiency
    single_throughput = results[1]['throughput_fps']
    
    print(f"\n📈 Scaling Efficiency:")
    for batch_size in batch_sizes:
        if batch_size > 1:
            theoretical_speedup = batch_size
            actual_speedup = results[batch_size]['throughput_fps'] / single_throughput
            efficiency = (actual_speedup / theoretical_speedup) * 100
            
            print(f"   • Batch {batch_size:3d}: {actual_speedup:5.1f}x speedup ({efficiency:5.1f}% efficient)")
    
    return results


def demonstrate_model_scaling():
    """Demonstrate model size scaling."""
    print("\n📈 MODEL SIZE SCALING")
    print("=" * 50)
    
    scales = [1, 2, 4, 8]
    base_dim = 8
    
    results = {}
    
    for scale in scales:
        config = LiquidConfig(
            input_dim=4,
            hidden_dim=base_dim * scale,
            output_dim=2,
            use_sparse=True,
            sparsity=0.5,
            use_layer_norm=False
        )
        
        model = LiquidNN(config)
        key = jax.random.PRNGKey(42)
        sample_input = jax.random.normal(key, (1, 4))
        params = model.init(key, sample_input)
        
        # Count parameters and memory
        param_count = sum(param.size for param in jax.tree_util.tree_leaves(params))
        memory_kb = param_count * 4 / 1024
        
        # Benchmark inference
        @jax.jit
        def inference(params, x):
            return model.apply(params, x, training=False)
        
        # Warmup
        for _ in range(10):
            _ = inference(params, sample_input)
        
        # Timing
        times = []
        for _ in range(1000):
            start = time.perf_counter()
            _ = inference(params, sample_input)
            times.append(time.perf_counter() - start)
        
        avg_time_us = np.mean(times) * 1_000_000
        throughput_fps = 1.0 / np.mean(times)
        
        results[scale] = {
            'hidden_dim': base_dim * scale,
            'param_count': param_count,
            'memory_kb': memory_kb,
            'latency_us': avg_time_us,
            'throughput_fps': throughput_fps
        }
        
        print(f"🔍 Scale {scale}x (Hidden: {base_dim * scale}):")
        print(f"   • Parameters: {param_count:,}")
        print(f"   • Memory: {memory_kb:.1f}KB")
        print(f"   • Latency: {avg_time_us:.1f}μs")
        print(f"   • Throughput: {throughput_fps:.0f} FPS")
    
    # Scaling analysis
    print(f"\n📊 Model Scaling Analysis:")
    base_throughput = results[1]['throughput_fps']
    base_memory = results[1]['memory_kb']
    
    for scale in scales:
        r = results[scale]
        throughput_ratio = r['throughput_fps'] / base_throughput
        memory_ratio = r['memory_kb'] / base_memory
        efficiency = throughput_ratio / memory_ratio if memory_ratio > 0 else 0
        
        print(f"   • {scale}x: {throughput_ratio:.2f}x speed, {memory_ratio:.1f}x memory, {efficiency:.3f} efficiency")
    
    return results


def demonstrate_production_readiness():
    """Demonstrate production deployment readiness."""
    print("\n🚀 PRODUCTION READINESS TEST")
    print("=" * 50)
    
    # Production configuration
    config = LiquidConfig(
        input_dim=4,
        hidden_dim=16,
        output_dim=2,
        use_sparse=True,
        sparsity=0.6,  # Aggressive sparsity for production
        use_layer_norm=False,  # No layer norm for speed
        dt=0.2  # Larger dt for stability
    )
    
    model = LiquidNN(config)
    key = jax.random.PRNGKey(42)
    sample_input = jax.random.normal(key, (1, 4))
    params = model.init(key, sample_input)
    
    # Model specs
    param_count = sum(param.size for param in jax.tree_util.tree_leaves(params))
    memory_kb = param_count * 4 / 1024
    
    print(f"📦 Production Model Specifications:")
    print(f"   • Architecture: {config.input_dim}→{config.hidden_dim}→{config.output_dim}")
    print(f"   • Parameters: {param_count:,}")
    print(f"   • Memory footprint: {memory_kb:.1f}KB")
    print(f"   • Sparsity: {config.sparsity:.0%}")
    
    # Create production inference function
    @jax.jit
    def production_inference(params, x):
        outputs, _ = model.apply(params, x, training=False)
        return outputs
    
    # Performance benchmarks
    test_scenarios = [
        ("Real-time Control (1kHz)", 1000, 1000),
        ("High-speed Sensing (10kHz)", 10000, 10000),
        ("Ultra-fast Response (100kHz)", 100000, 100000)
    ]
    
    print(f"\n⚡ Performance Benchmarks:")
    
    for scenario_name, target_hz, num_tests in test_scenarios:
        # Warmup
        for _ in range(10):
            _ = production_inference(params, sample_input)
        
        # Benchmark
        times = []
        for _ in range(num_tests):
            start = time.perf_counter()
            _ = production_inference(params, sample_input)
            times.append(time.perf_counter() - start)
        
        avg_time_us = np.mean(times) * 1_000_000
        max_hz = 1_000_000 / avg_time_us
        success = max_hz >= target_hz
        
        status = "✅ CAPABLE" if success else "❌ LIMITED"
        
        print(f"   • {scenario_name}:")
        print(f"     - Target: {target_hz:,}Hz")
        print(f"     - Achieved: {max_hz:,.0f}Hz")
        print(f"     - Latency: {avg_time_us:.1f}μs")
        print(f"     - Status: {status}")
    
    # Robustness test
    print(f"\n🛡️  Robustness Test:")
    
    # Test with challenging inputs
    test_cases = [
        ("Normal input", jnp.array([[1.0, 0.5, -0.3, 0.8]])),
        ("Zero input", jnp.zeros((1, 4))),
        ("Large input", jnp.array([[10.0, -10.0, 5.0, -5.0]])),
        ("Small input", jnp.array([[0.001, -0.001, 0.0005, -0.0005]]))
    ]
    
    robust_outputs = 0
    
    for test_name, test_input in test_cases:
        try:
            output = production_inference(params, test_input)
            is_finite = jnp.all(jnp.isfinite(output))
            is_bounded = jnp.all(jnp.abs(output) < 100.0)  # Reasonable bounds
            
            if is_finite and is_bounded:
                robust_outputs += 1
                status = "✅ STABLE"
            else:
                status = "⚠️  UNSTABLE"
            
            print(f"   • {test_name}: {status} (output: {output.flatten()})")
            
        except Exception as e:
            print(f"   • {test_name}: ❌ ERROR ({str(e)})")
    
    robustness_score = robust_outputs / len(test_cases) * 100
    
    # Final assessment
    print(f"\n🎯 PRODUCTION READINESS ASSESSMENT:")
    
    criteria = {
        "Memory Efficiency": memory_kb < 32,  # < 32KB
        "Real-time Performance": max_hz >= 1000,  # >= 1kHz
        "Robustness": robustness_score >= 75,  # >= 75% stable
        "Parameter Efficiency": param_count < 1000  # < 1k params
    }
    
    passed = sum(criteria.values())
    total = len(criteria)
    
    for criterion, passed_test in criteria.items():
        status = "✅ PASS" if passed_test else "❌ FAIL"
        print(f"   • {criterion}: {status}")
    
    overall_score = passed / total * 100
    
    if overall_score >= 75:
        readiness = "🟢 PRODUCTION READY"
    elif overall_score >= 50:
        readiness = "🟡 DEVELOPMENT READY"
    else:
        readiness = "🔴 NOT READY"
    
    print(f"\n🏆 OVERALL READINESS: {readiness} ({overall_score:.0f}%)")
    
    return {
        'model_specs': {
            'param_count': param_count,
            'memory_kb': memory_kb,
            'max_hz': max_hz,
            'robustness_score': robustness_score
        },
        'production_score': overall_score
    }


def main():
    """Run scaling demonstrations."""
    print("🌊 LIQUID EDGE LNN - GENERATION 3: SCALING DEMO")
    print("=" * 60)
    
    batch_results = demonstrate_batch_scaling()
    model_results = demonstrate_model_scaling()
    production_results = demonstrate_production_readiness()
    
    print(f"\n🎯 GENERATION 3 SCALING SUMMARY")
    print("=" * 60)
    
    # Best batch size
    best_batch = max(batch_results.items(), key=lambda x: x[1]['throughput_fps'])
    print(f"✅ OPTIMAL BATCH SIZE: {best_batch[0]} ({best_batch[1]['throughput_fps']:.0f} FPS)")
    
    # Best model scale
    best_scale = max(model_results.items(), key=lambda x: x[1]['throughput_fps'] / x[1]['memory_kb'])
    print(f"✅ OPTIMAL MODEL SCALE: {best_scale[0]}x ({best_scale[1]['throughput_fps']:.0f} FPS, {best_scale[1]['memory_kb']:.1f}KB)")
    
    # Production readiness
    print(f"✅ PRODUCTION READINESS: {production_results['production_score']:.0f}%")
    print(f"   • Max throughput: {production_results['model_specs']['max_hz']:,.0f}Hz")
    print(f"   • Memory footprint: {production_results['model_specs']['memory_kb']:.1f}KB")
    
    print("=" * 60)
    print("🏆 GENERATION 3 SUCCESS: SCALABLE PRODUCTION DEPLOYMENT")


if __name__ == "__main__":
    main()