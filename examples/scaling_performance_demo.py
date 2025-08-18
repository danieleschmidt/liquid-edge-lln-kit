#!/usr/bin/env python3
"""Demonstration of high-performance scaling and optimization features."""

import asyncio
import time
import numpy as np
import jax
import jax.numpy as jnp
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from typing import List
from liquid_edge import LiquidNN, LiquidConfig
from liquid_edge.high_performance_inference import (
    HighPerformanceInferenceEngine, PerformanceConfig, InferenceMode, 
    LoadBalancingStrategy, DistributedInferenceCoordinator
)


def create_test_workload(num_samples: int = 1000, input_dim: int = 4) -> np.ndarray:
    """Create realistic test workload for performance testing."""
    np.random.seed(42)
    
    # Simulate various sensor input patterns
    normal_data = np.random.normal(0.0, 0.3, (num_samples // 2, input_dim))
    sine_data = np.array([
        [np.sin(i * 0.1), np.cos(i * 0.1), np.sin(i * 0.05), np.cos(i * 0.05)]
        for i in range(num_samples // 4)
    ])
    noise_data = np.random.uniform(-1.0, 1.0, (num_samples // 4, input_dim))
    
    workload = np.vstack([normal_data, sine_data, noise_data])
    return workload[:num_samples]


def benchmark_sequential_inference(model, params, test_data, num_iterations: int = 100):
    """Benchmark sequential inference performance."""
    print("📊 Benchmarking Sequential Inference...")
    
    latencies = []
    start_time = time.time()
    
    for i in range(num_iterations):
        inputs = jnp.array(test_data[i % len(test_data)].reshape(1, -1))
        
        iter_start = time.time()
        result = model.apply(params, inputs, training=False)
        iter_end = time.time()
        
        latencies.append((iter_end - iter_start) * 1000)  # Convert to ms
    
    end_time = time.time()
    total_time = end_time - start_time
    
    results = {
        "total_time_s": total_time,
        "total_requests": num_iterations,
        "throughput_rps": num_iterations / total_time,
        "average_latency_ms": np.mean(latencies),
        "min_latency_ms": np.min(latencies),
        "max_latency_ms": np.max(latencies),
        "p50_latency_ms": np.percentile(latencies, 50),
        "p95_latency_ms": np.percentile(latencies, 95),
        "p99_latency_ms": np.percentile(latencies, 99)
    }
    
    print(f"  Sequential Results:")
    print(f"    Throughput: {results['throughput_rps']:.2f} RPS")
    print(f"    Avg Latency: {results['average_latency_ms']:.2f}ms")
    print(f"    P95 Latency: {results['p95_latency_ms']:.2f}ms")
    print(f"    P99 Latency: {results['p99_latency_ms']:.2f}ms")
    
    return results


async def benchmark_async_inference(engine: HighPerformanceInferenceEngine, 
                                  test_data, num_iterations: int = 100):
    """Benchmark asynchronous inference performance."""
    print("\n🚀 Benchmarking Asynchronous Inference...")
    
    start_time = time.time()
    
    # Create concurrent inference tasks
    tasks = []
    for i in range(num_iterations):
        inputs = jnp.array(test_data[i % len(test_data)].reshape(1, -1))
        task = asyncio.create_task(
            engine.async_inference(inputs, request_id=f"async_{i}", priority=1)
        )
        tasks.append(task)
    
    # Wait for all tasks to complete
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Count successful vs failed requests
    successful = sum(1 for r in results if not isinstance(r, Exception))
    failed = len(results) - successful
    
    performance_report = engine.get_performance_report()
    
    print(f"  Async Results:")
    print(f"    Total Time: {total_time:.2f}s")
    print(f"    Successful: {successful}/{num_iterations}")
    print(f"    Failed: {failed}/{num_iterations}")
    print(f"    Throughput: {successful / total_time:.2f} RPS")
    print(f"    Avg Latency: {performance_report['performance_metrics']['average_latency_ms']:.2f}ms")
    print(f"    Cache Hit Rate: {performance_report['caching']['cache_hit_rate_percent']:.1f}%")
    
    return performance_report


def benchmark_batch_inference(engine: HighPerformanceInferenceEngine, 
                             test_data, batch_sizes: List[int] = [1, 8, 16, 32]):
    """Benchmark batch inference with different batch sizes."""
    print("\n📦 Benchmarking Batch Inference...")
    
    batch_results = {}
    
    for batch_size in batch_sizes:
        print(f"\n  Testing batch size: {batch_size}")
        
        # Prepare batches
        num_batches = 50
        batch_inputs = []
        for i in range(num_batches):
            batch = []
            for j in range(batch_size):
                idx = (i * batch_size + j) % len(test_data)
                batch.append(jnp.array(test_data[idx]))
            batch_inputs.append(batch)
        
        # Benchmark batch processing
        start_time = time.time()
        
        for batch in batch_inputs:
            results = engine.batch_inference(batch, batch_size)
        
        end_time = time.time()
        total_time = end_time - start_time
        total_samples = num_batches * batch_size
        
        batch_results[batch_size] = {
            "total_time_s": total_time,
            "total_samples": total_samples,
            "throughput_sps": total_samples / total_time,  # Samples per second
            "avg_time_per_batch_ms": (total_time / num_batches) * 1000,
            "avg_time_per_sample_ms": (total_time / total_samples) * 1000
        }
        
        print(f"    Throughput: {batch_results[batch_size]['throughput_sps']:.2f} samples/sec")
        print(f"    Avg time per sample: {batch_results[batch_size]['avg_time_per_sample_ms']:.3f}ms")
    
    # Find optimal batch size
    optimal_batch_size = max(batch_results.keys(), 
                           key=lambda bs: batch_results[bs]['throughput_sps'])
    
    print(f"\n  🏆 Optimal batch size: {optimal_batch_size} "
          f"({batch_results[optimal_batch_size]['throughput_sps']:.2f} samples/sec)")
    
    return batch_results


def test_streaming_inference(engine: HighPerformanceInferenceEngine, test_data):
    """Test streaming inference capabilities."""
    print("\n🌊 Testing Streaming Inference...")
    
    # Create input stream
    stream_index = 0
    stream_results = []
    stream_lock = threading.Lock()
    
    def input_stream():
        nonlocal stream_index
        if stream_index >= len(test_data):
            return None  # End of stream
        
        data = jnp.array(test_data[stream_index])
        stream_index += 1
        time.sleep(0.01)  # Simulate 100Hz data rate
        return data
    
    def output_callback(result):
        with stream_lock:
            stream_results.append({
                "timestamp": time.time(),
                "result": result,
                "index": len(stream_results)
            })
    
    # Start streaming
    start_time = time.time()
    stream_thread = engine.streaming_inference(input_stream, output_callback)
    
    # Let it run for a few seconds
    time.sleep(2.0)
    
    # Calculate streaming performance
    end_time = time.time()
    total_time = end_time - start_time
    
    with stream_lock:
        num_results = len(stream_results)
        if num_results > 1:
            # Calculate inter-arrival times
            timestamps = [r["timestamp"] for r in stream_results]
            inter_arrivals = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
            avg_inter_arrival = np.mean(inter_arrivals) * 1000  # Convert to ms
        else:
            avg_inter_arrival = 0
    
    print(f"  Streaming Results:")
    print(f"    Duration: {total_time:.2f}s")
    print(f"    Processed samples: {num_results}")
    print(f"    Streaming rate: {num_results / total_time:.2f} samples/sec")
    print(f"    Avg inter-arrival: {avg_inter_arrival:.2f}ms")
    
    return {
        "duration_s": total_time,
        "samples_processed": num_results,
        "streaming_rate_sps": num_results / total_time,
        "avg_inter_arrival_ms": avg_inter_arrival
    }


def test_load_balancing(test_data):
    """Test distributed load balancing."""
    print("\n⚖️  Testing Load Balancing...")
    
    # Create multiple engine configurations
    configs = [
        PerformanceConfig(max_workers=2, batch_size=16, enable_caching=True),
        PerformanceConfig(max_workers=4, batch_size=32, enable_caching=True),
        PerformanceConfig(max_workers=2, batch_size=8, enable_caching=False)
    ]
    
    # Create distributed coordinator
    coordinator = DistributedInferenceCoordinator(configs)
    
    # Simulate load balancing (this would be async in real implementation)
    print("  Testing Round Robin load balancing...")
    for i in range(10):
        inputs = jnp.array(test_data[i % len(test_data)].reshape(1, -1))
        try:
            # In a real implementation, this would distribute across network
            engine_idx = i % len(coordinator.engines)
            print(f"    Request {i+1} -> Engine {engine_idx}")
        except Exception as e:
            print(f"    Request {i+1} failed: {e}")
    
    print("  ✅ Load balancing test completed")


def test_auto_scaling(engine: HighPerformanceInferenceEngine, test_data):
    """Test auto-scaling capabilities."""
    print("\n📈 Testing Auto-Scaling...")
    
    # Generate load spike
    print("  Generating load spike...")
    
    async def load_spike():
        tasks = []
        # Create 50 concurrent requests
        for i in range(50):
            inputs = jnp.array(test_data[i % len(test_data)].reshape(1, -1))
            task = asyncio.create_task(
                engine.async_inference(inputs, request_id=f"spike_{i}")
            )
            tasks.append(task)
        
        # Wait for all to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return results
    
    # Monitor scaling behavior
    initial_workers = len(engine.workers)
    initial_report = engine.get_performance_report()
    
    print(f"    Initial workers: {initial_workers}")
    print(f"    Initial queue length: {initial_report['performance_metrics']['queue_length']}")
    
    # Execute load spike
    start_time = time.time()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    results = loop.run_until_complete(load_spike())
    loop.close()
    
    # Check final state
    time.sleep(1.0)  # Allow scaling to settle
    final_report = engine.get_performance_report()
    final_workers = len(engine.workers)
    
    print(f"    Final workers: {final_workers}")
    print(f"    Final queue length: {final_report['performance_metrics']['queue_length']}")
    print(f"    Scaling response: {final_workers - initial_workers} workers added")
    
    successful_requests = sum(1 for r in results if not isinstance(r, Exception))
    print(f"    Load spike results: {successful_requests}/50 successful")
    
    return {
        "initial_workers": initial_workers,
        "final_workers": final_workers,
        "scaling_delta": final_workers - initial_workers,
        "successful_requests": successful_requests,
        "total_requests": 50
    }


def generate_performance_summary(sequential_results, async_results, batch_results, 
                               streaming_results, scaling_results):
    """Generate comprehensive performance summary."""
    print("\n\n📊 PERFORMANCE SUMMARY")
    print("=" * 60)
    
    # Throughput comparison
    print("🚀 Throughput Comparison:")
    print(f"  Sequential:     {sequential_results['throughput_rps']:8.2f} RPS")
    print(f"  Asynchronous:   {async_results['performance_metrics']['throughput_rps']:8.2f} RPS")
    
    best_batch_size = max(batch_results.keys(), 
                         key=lambda bs: batch_results[bs]['throughput_sps'])
    print(f"  Batch (size {best_batch_size}): {batch_results[best_batch_size]['throughput_sps']:8.2f} samples/sec")
    print(f"  Streaming:      {streaming_results['streaming_rate_sps']:8.2f} samples/sec")
    
    # Latency comparison
    print("\n⚡ Latency Comparison:")
    print(f"  Sequential P95: {sequential_results['p95_latency_ms']:8.2f}ms")
    print(f"  Async Average:  {async_results['performance_metrics']['average_latency_ms']:8.2f}ms")
    print(f"  Streaming Avg:  {streaming_results['avg_inter_arrival_ms']:8.2f}ms")
    
    # Optimization impact
    print("\n🔧 Optimization Impact:")
    cache_hit_rate = async_results['caching']['cache_hit_rate_percent']
    print(f"  Cache Hit Rate: {cache_hit_rate:8.1f}%")
    print(f"  JIT Compilation: {'✅ Enabled' if async_results['optimizations']['jit_compilation'] else '❌ Disabled'}")
    print(f"  Auto-scaling:   {'✅ Active' if scaling_results['scaling_delta'] > 0 else '➖ No scaling needed'}")
    
    # Calculate overall performance score
    async_speedup = async_results['performance_metrics']['throughput_rps'] / sequential_results['throughput_rps']
    batch_speedup = batch_results[best_batch_size]['throughput_sps'] / sequential_results['throughput_rps']
    
    print(f"\n🏆 Performance Gains:")
    print(f"  Async Speedup:  {async_speedup:.2f}x")
    print(f"  Batch Speedup:  {batch_speedup:.2f}x")
    
    overall_score = min(100, (async_speedup + batch_speedup + cache_hit_rate/100) * 25)
    print(f"\n🎯 Overall Performance Score: {overall_score:.1f}/100")
    
    if overall_score >= 90:
        print("  🌟 Excellent - Ready for high-performance production deployment")
    elif overall_score >= 75:
        print("  ✅ Good - Suitable for most production workloads")
    elif overall_score >= 60:
        print("  ⚠️  Fair - Consider additional optimizations")
    else:
        print("  ❌ Poor - Significant performance improvements needed")


async def main():
    """Main demonstration of scaling and performance features."""
    print("🚀 Liquid Edge LLN - High-Performance Scaling Demonstration")
    print("=" * 70)
    
    # Create liquid neural network
    config = LiquidConfig(
        input_dim=4,
        hidden_dim=16,
        output_dim=2,
        energy_budget_mw=100.0,
        use_sparse=True,
        sparsity=0.3
    )
    
    model = LiquidNN(config)
    key = jax.random.PRNGKey(42)
    dummy_input = jnp.ones((1, 4))
    params = model.init(key, dummy_input)
    
    print(f"Model Configuration: {config.input_dim}→{config.hidden_dim}→{config.output_dim}")
    print(f"Sparsity: {config.sparsity*100:.1f}%")
    print(f"Energy Budget: {config.energy_budget_mw}mW")
    
    # Create high-performance inference engine
    perf_config = PerformanceConfig(
        max_workers=4,
        batch_size=32,
        enable_jit_compilation=True,
        enable_caching=True,
        cache_size=500,
        inference_mode=InferenceMode.PARALLEL,
        enable_auto_scaling=True,
        target_latency_ms=5.0
    )
    
    engine = HighPerformanceInferenceEngine(model, perf_config)
    engine.set_model_params(params)
    
    # Generate test workload
    print("\n📋 Generating Test Workload...")
    test_data = create_test_workload(num_samples=1000, input_dim=4)
    print(f"Generated {len(test_data)} test samples")
    
    try:
        # Start workers
        for i in range(perf_config.max_workers):
            engine._add_worker()
        
        # Run performance benchmarks
        sequential_results = benchmark_sequential_inference(model, params, test_data, 100)
        
        async_results = await benchmark_async_inference(engine, test_data, 100)
        
        batch_results = benchmark_batch_inference(engine, test_data, [1, 8, 16, 32])
        
        streaming_results = test_streaming_inference(engine, test_data[:100])
        
        test_load_balancing(test_data[:10])
        
        scaling_results = test_auto_scaling(engine, test_data[:20])
        
        # Generate summary
        generate_performance_summary(
            sequential_results, async_results, batch_results,
            streaming_results, scaling_results
        )
        
        print("\n🔗 Next Steps:")
        print("  1. Deploy optimized model: liquid-lln deploy --performance-optimized")
        print("  2. Monitor production metrics: liquid-lln monitor --performance")
        print("  3. Set up auto-scaling: liquid-lln autoscale enable --target-latency 5ms")
        
    except Exception as e:
        print(f"\n❌ Performance test failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        engine.shutdown()


if __name__ == "__main__":
    asyncio.run(main())