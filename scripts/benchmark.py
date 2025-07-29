"""Performance benchmarking and profiling tools."""

import json
import time
import psutil
import statistics
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import jax
import jax.numpy as jnp
from liquid_edge import LiquidNN, LiquidConfig


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    name: str
    duration_ms: float
    memory_mb: float
    throughput: Optional[float] = None
    energy_estimate_mw: Optional[float] = None
    metadata: Dict[str, Any] = None


class PerformanceProfiler:
    """Profile performance of liquid neural networks."""
    
    def __init__(self, output_dir: Path = Path("benchmarks")):
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)
        self.results: List[BenchmarkResult] = []
    
    def measure_training_performance(
        self, 
        config: LiquidConfig,
        epochs: int = 100,
        batch_size: int = 32
    ) -> BenchmarkResult:
        """Benchmark training performance."""
        print(f"Benchmarking training performance...")
        
        # Generate synthetic data
        key = jax.random.PRNGKey(42)
        input_data = jax.random.normal(key, (batch_size, config.input_dim))
        target_data = jax.random.normal(key, (batch_size, config.output_dim))
        
        # Create model
        model = LiquidNN(config)
        params = model.init(key, input_data[:1])
        
        # Loss function
        def loss_fn(params, x, y):
            pred = model.apply(params, x)
            return jnp.mean((pred - y) ** 2)
        
        # JIT compile
        loss_and_grad = jax.jit(jax.value_and_grad(loss_fn))
        
        # Warmup
        for _ in range(10):
            loss, grads = loss_and_grad(params, input_data, target_data)
        
        # Measure memory before training
        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024 / 1024
        
        # Benchmark training
        start_time = time.perf_counter()
        
        for epoch in range(epochs):
            loss, grads = loss_and_grad(params, input_data, target_data)
        
        end_time = time.perf_counter()
        mem_after = process.memory_info().rss / 1024 / 1024
        
        duration_ms = (end_time - start_time) * 1000
        throughput = epochs / (end_time - start_time)  # epochs/second
        memory_used = mem_after - mem_before
        
        # Estimate energy (simplified model)
        energy_estimate = self._estimate_energy(duration_ms, memory_used)
        
        return BenchmarkResult(
            name=f"training_{config.hidden_dim}h_{epochs}e",
            duration_ms=duration_ms,
            memory_mb=memory_used,
            throughput=throughput,
            energy_estimate_mw=energy_estimate,
            metadata={
                "epochs": epochs,
                "batch_size": batch_size,
                "hidden_dim": config.hidden_dim,
                "sparsity": config.sparsity if config.use_sparse else 0.0
            }
        )
    
    def measure_inference_performance(
        self,
        config: LiquidConfig,
        num_samples: int = 1000,
        batch_sizes: List[int] = [1, 8, 32, 128]
    ) -> List[BenchmarkResult]:
        """Benchmark inference performance across batch sizes."""
        print(f"Benchmarking inference performance...")
        
        results = []
        key = jax.random.PRNGKey(42)
        model = LiquidNN(config)
        
        for batch_size in batch_sizes:
            input_data = jax.random.normal(key, (batch_size, config.input_dim))
            params = model.init(key, input_data[:1])
            
            # JIT compile
            apply_fn = jax.jit(model.apply)
            
            # Warmup
            for _ in range(10):
                _ = apply_fn(params, input_data)
            
            # Measure memory
            process = psutil.Process()
            mem_before = process.memory_info().rss / 1024 / 1024
            
            # Benchmark
            durations = []
            for _ in range(num_samples):
                start_time = time.perf_counter()
                output = apply_fn(params, input_data)
                end_time = time.perf_counter()
                durations.append((end_time - start_time) * 1000)
            
            mem_after = process.memory_info().rss / 1024 / 1024
            
            avg_duration = statistics.mean(durations)
            throughput = batch_size / (avg_duration / 1000)  # samples/second
            memory_used = mem_after - mem_before
            energy_estimate = self._estimate_energy(avg_duration, memory_used)
            
            results.append(BenchmarkResult(
                name=f"inference_batch_{batch_size}",
                duration_ms=avg_duration,
                memory_mb=memory_used,  
                throughput=throughput,
                energy_estimate_mw=energy_estimate,
                metadata={
                    "batch_size": batch_size,
                    "samples_per_batch": batch_size,
                    "duration_std": statistics.stdev(durations),
                    "duration_p95": statistics.quantiles(durations, n=20)[18],  # 95th percentile
                }
            ))
        
        return results
    
    def measure_memory_scaling(
        self,
        input_dim: int = 4,
        output_dim: int = 2,
        hidden_dims: List[int] = [8, 16, 32, 64, 128]
    ) -> List[BenchmarkResult]:
        """Benchmark memory usage scaling with model size."""
        print(f"Benchmarking memory scaling...")
        
        results = []
        key = jax.random.PRNGKey(42)
        
        for hidden_dim in hidden_dims:
            config = LiquidConfig(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                use_sparse=True,
                sparsity=0.3
            )
            
            model = LiquidNN(config)
            input_data = jax.random.normal(key, (1, input_dim))
            
            # Measure memory before model creation
            process = psutil.Process()
            mem_before = process.memory_info().rss / 1024 / 1024
            
            # Initialize model
            start_time = time.perf_counter()
            params = model.init(key, input_data)
            end_time = time.perf_counter()
            
            mem_after = process.memory_info().rss / 1024 / 1024
            
            # Count parameters
            param_count = sum(
                param.size for param in jax.tree_util.tree_leaves(params)
            )
            
            # Estimate model size in MB (float32 = 4 bytes)
            model_size_mb = param_count * 4 / 1024 / 1024
            
            duration_ms = (end_time - start_time) * 1000
            memory_used = mem_after - mem_before
            
            results.append(BenchmarkResult(
                name=f"memory_scaling_{hidden_dim}h",
                duration_ms=duration_ms,
                memory_mb=memory_used,
                metadata={
                    "hidden_dim": hidden_dim,
                    "param_count": param_count,
                    "model_size_mb": model_size_mb,
                    "memory_efficiency": model_size_mb / memory_used if memory_used > 0 else 0
                }
            ))
        
        return results
    
    def measure_sparsity_impact(
        self,
        base_config: LiquidConfig,
        sparsity_levels: List[float] = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]
    ) -> List[BenchmarkResult]:
        """Benchmark impact of sparsity on performance."""
        print(f"Benchmarking sparsity impact...")
        
        results = []
        key = jax.random.PRNGKey(42)
        input_data = jax.random.normal(key, (32, base_config.input_dim))
        
        for sparsity in sparsity_levels:
            config = LiquidConfig(
                input_dim=base_config.input_dim,
                hidden_dim=base_config.hidden_dim,
                output_dim=base_config.output_dim,
                use_sparse=sparsity > 0,
                sparsity=sparsity
            )
            
            model = LiquidNN(config)
            params = model.init(key, input_data[:1])
            apply_fn = jax.jit(model.apply)
            
            # Warmup
            for _ in range(10):
                _ = apply_fn(params, input_data)
            
            # Benchmark
            durations = []
            for _ in range(100):
                start_time = time.perf_counter()
                output = apply_fn(params, input_data)
                end_time = time.perf_counter()
                durations.append((end_time - start_time) * 1000)
            
            avg_duration = statistics.mean(durations)
            throughput = len(input_data) / (avg_duration / 1000)
            
            # Estimate theoretical speedup from sparsity
            theoretical_speedup = 1 / (1 - sparsity) if sparsity > 0 else 1.0
            
            results.append(BenchmarkResult(
                name=f"sparsity_{int(sparsity*100)}pct",
                duration_ms=avg_duration,
                memory_mb=0,  # Not measuring memory for this benchmark
                throughput=throughput,
                metadata={
                    "sparsity": sparsity,
                    "theoretical_speedup": theoretical_speedup,
                    "actual_speedup": durations[0] / avg_duration if sparsity > 0 else 1.0,
                    "efficiency": (theoretical_speedup / (durations[0] / avg_duration)) if sparsity > 0 else 1.0
                }
            ))
        
        return results
    
    def _estimate_energy(self, duration_ms: float, memory_mb: float) -> float:
        """Estimate energy consumption in milliwatts (simplified model)."""
        # Simplified energy model:
        # Base power consumption + computational power + memory power
        base_power_mw = 50  # Baseline power
        compute_power_mw = duration_ms * 0.1  # Power proportional to time
        memory_power_mw = memory_mb * 0.5  # Power proportional to memory
        
        return base_power_mw + compute_power_mw + memory_power_mw
    
    def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """Run all benchmarks and return results."""
        print("Starting comprehensive performance benchmark...")
        
        # Base configuration
        base_config = LiquidConfig(
            input_dim=4,
            hidden_dim=16,
            output_dim=2,
            use_sparse=True,
            sparsity=0.3
        )
        
        # Run all benchmarks
        self.results.extend([self.measure_training_performance(base_config)])
        self.results.extend(self.measure_inference_performance(base_config))
        self.results.extend(self.measure_memory_scaling())
        self.results.extend(self.measure_sparsity_impact(base_config))
        
        # Generate summary
        summary = self._generate_summary()
        
        # Save results
        self._save_results()
        
        return summary
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate benchmark summary."""
        training_results = [r for r in self.results if "training" in r.name]
        inference_results = [r for r in self.results if "inference" in r.name]
        
        summary = {
            "total_benchmarks": len(self.results),
            "training": {
                "avg_duration_ms": statistics.mean([r.duration_ms for r in training_results]) if training_results else 0,
                "avg_throughput": statistics.mean([r.throughput for r in training_results if r.throughput]) if training_results else 0,
            },
            "inference": {
                "avg_duration_ms": statistics.mean([r.duration_ms for r in inference_results]) if inference_results else 0,
                "max_throughput": max([r.throughput for r in inference_results if r.throughput]) if inference_results else 0,
            },
            "memory": {
                "peak_usage_mb": max([r.memory_mb for r in self.results if r.memory_mb > 0]) if self.results else 0,
                "avg_usage_mb": statistics.mean([r.memory_mb for r in self.results if r.memory_mb > 0]) if self.results else 0,
            },
            "energy": {
                "avg_estimate_mw": statistics.mean([r.energy_estimate_mw for r in self.results if r.energy_estimate_mw]) if self.results else 0,
            }
        }
        
        return summary
    
    def _save_results(self):
        """Save benchmark results to files."""
        # Save detailed results
        results_data = {
            "timestamp": time.time(),
            "results": [
                {
                    "name": r.name,
                    "duration_ms": r.duration_ms,
                    "memory_mb": r.memory_mb,
                    "throughput": r.throughput,
                    "energy_estimate_mw": r.energy_estimate_mw,
                    "metadata": r.metadata
                }
                for r in self.results
            ],
            "summary": self._generate_summary()
        }
        
        with open(self.output_dir / "benchmark_results.json", "w") as f:
            json.dump(results_data, f, indent=2)
        
        # Save CSV for plotting
        import csv
        with open(self.output_dir / "benchmark_results.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "name", "duration_ms", "memory_mb", "throughput", 
                "energy_estimate_mw", "metadata"
            ])
            for r in self.results:
                writer.writerow([
                    r.name, r.duration_ms, r.memory_mb, r.throughput,
                    r.energy_estimate_mw, json.dumps(r.metadata)
                ])
        
        print(f"Results saved to {self.output_dir}/")


def main():
    """Main entry point for benchmarking."""
    profiler = PerformanceProfiler()
    summary = profiler.run_comprehensive_benchmark()
    
    print("\n" + "="*60) 
    print("PERFORMANCE BENCHMARK SUMMARY")
    print("="*60)
    print(f"Total Benchmarks: {summary['total_benchmarks']}")
    print(f"Training Avg Duration: {summary['training']['avg_duration_ms']:.2f}ms")
    print(f"Inference Max Throughput: {summary['inference']['max_throughput']:.0f} samples/s")
    print(f"Peak Memory Usage: {summary['memory']['peak_usage_mb']:.2f}MB")
    print(f"Avg Energy Estimate: {summary['energy']['avg_estimate_mw']:.2f}mW")
    print("="*60)
    print("✅ Benchmark completed successfully!")


if __name__ == "__main__":
    main()