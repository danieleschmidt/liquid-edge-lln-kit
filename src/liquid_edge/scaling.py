"""Auto-scaling and performance optimization for liquid neural networks."""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import threading
import time
import queue
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing as mp
from contextlib import contextmanager
import functools
import gc
from pathlib import Path
import pickle
import psutil

from .core import LiquidNN, LiquidConfig
from .monitoring import LiquidNetworkMonitor, PerformanceMetrics, AlertLevel
from .error_handling import RobustErrorHandler, ModelInferenceError, ErrorSeverity


class ScalingStrategy(Enum):
    """Auto-scaling strategies."""
    CPU_BASED = "cpu_based"
    MEMORY_BASED = "memory_based" 
    LATENCY_BASED = "latency_based"
    ENERGY_BASED = "energy_based"
    ADAPTIVE = "adaptive"


@dataclass
class ScalingConfig:
    """Configuration for auto-scaling system."""
    strategy: ScalingStrategy = ScalingStrategy.ADAPTIVE
    min_workers: int = 1
    max_workers: int = mp.cpu_count()
    target_cpu_percent: float = 70.0
    target_memory_percent: float = 80.0
    target_latency_ms: float = 10.0
    target_energy_efficiency: float = 50.0  # mW
    scale_up_threshold: float = 0.8
    scale_down_threshold: float = 0.3
    cooldown_seconds: float = 30.0
    enable_batching: bool = True
    max_batch_size: int = 32
    batch_timeout_ms: float = 5.0


@dataclass
class WorkerMetrics:
    """Metrics for individual worker."""
    worker_id: int
    cpu_percent: float = 0.0
    memory_mb: float = 0.0
    inference_latency_ms: float = 0.0
    energy_consumption_mw: float = 0.0
    throughput_fps: float = 0.0
    error_rate: float = 0.0
    active_requests: int = 0
    timestamp: float = field(default_factory=time.time)


class HighPerformanceLiquidPool:
    """High-performance worker pool for liquid neural networks."""
    
    def __init__(self, 
                 model_config: LiquidConfig,
                 scaling_config: ScalingConfig,
                 monitor: Optional[LiquidNetworkMonitor] = None):
        
        self.model_config = model_config
        self.scaling_config = scaling_config
        self.monitor = monitor
        
        # Worker management
        self._workers = {}
        self._worker_metrics = {}
        self._request_queue = queue.Queue(maxsize=1000)
        self._result_queues = {}
        self._next_worker_id = 0
        
        # Scaling state
        self._current_workers = 0
        self._last_scale_time = 0.0
        self._scaling_lock = threading.RLock()
        
        # Performance optimization
        self._batch_processor = None
        self._load_balancer = None
        
        # Monitoring
        self._monitoring_thread = None
        self._shutdown_event = threading.Event()
        
        # Initialize with minimum workers
        self._initialize_workers()
        self._start_monitoring()
        
    def _initialize_workers(self):
        """Initialize minimum number of workers."""
        for _ in range(self.scaling_config.min_workers):
            self._add_worker()
            
        if self.scaling_config.enable_batching:
            self._batch_processor = BatchProcessor(
                self.scaling_config.max_batch_size,
                self.scaling_config.batch_timeout_ms
            )
            
        self._load_balancer = LoadBalancer()
        
    def _add_worker(self) -> int:
        """Add new worker to pool."""
        worker_id = self._next_worker_id
        self._next_worker_id += 1
        
        # Create dedicated process for worker
        worker_process = mp.Process(
            target=self._worker_main,
            args=(worker_id, self.model_config),
            name=f"liquid-worker-{worker_id}"
        )
        worker_process.start()
        
        self._workers[worker_id] = worker_process
        self._worker_metrics[worker_id] = WorkerMetrics(worker_id)
        self._result_queues[worker_id] = queue.Queue()
        
        self._current_workers += 1
        
        if self.monitor:
            self.monitor.alert(
                AlertLevel.INFO,
                f"Added worker {worker_id}, total workers: {self._current_workers}",
                "scaling",
                {"worker_id": worker_id, "total_workers": self._current_workers}
            )
        
        return worker_id
    
    def _remove_worker(self, worker_id: int):
        """Remove worker from pool."""
        if worker_id in self._workers:
            worker = self._workers[worker_id]
            worker.terminate()
            worker.join(timeout=5.0)
            
            if worker.is_alive():
                worker.kill()
            
            del self._workers[worker_id]
            del self._worker_metrics[worker_id]
            if worker_id in self._result_queues:
                del self._result_queues[worker_id]
            
            self._current_workers -= 1
            
            if self.monitor:
                self.monitor.alert(
                    AlertLevel.INFO,
                    f"Removed worker {worker_id}, total workers: {self._current_workers}",
                    "scaling",
                    {"worker_id": worker_id, "total_workers": self._current_workers}
                )
    
    def _worker_main(self, worker_id: int, config: LiquidConfig):
        """Main function for worker process."""
        try:
            # Initialize model in worker process
            model = LiquidNN(config)
            key = jax.random.PRNGKey(worker_id)
            dummy_input = jnp.ones((1, config.input_dim))
            params = model.init(key, dummy_input, training=False)
            
            hidden_state = jnp.zeros((1, config.hidden_dim))
            
            # Worker event loop
            while not self._shutdown_event.is_set():
                try:
                    # Get request from queue (with timeout)
                    request = self._request_queue.get(timeout=1.0)
                    
                    if request is None:  # Shutdown signal
                        break
                    
                    request_id, input_data, start_time = request
                    
                    # Process inference
                    inference_start = time.perf_counter()
                    
                    output, new_hidden = model.apply(
                        params, input_data, hidden_state, training=False
                    )
                    
                    hidden_state = new_hidden
                    inference_time = (time.perf_counter() - inference_start) * 1000
                    
                    # Send result back
                    result = {
                        "request_id": request_id,
                        "output": output,
                        "worker_id": worker_id,
                        "inference_time_ms": inference_time,
                        "queue_time_ms": (inference_start - start_time) * 1000
                    }
                    
                    self._result_queues[worker_id].put(result)
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    if self.monitor:
                        self.monitor.alert(
                            AlertLevel.ERROR,
                            f"Worker {worker_id} error: {str(e)}",
                            "worker",
                            {"worker_id": worker_id}
                        )
        
        except Exception as e:
            if self.monitor:
                self.monitor.alert(
                    AlertLevel.CRITICAL,
                    f"Worker {worker_id} crashed: {str(e)}",
                    "worker",
                    {"worker_id": worker_id}
                )
    
    def _start_monitoring(self):
        """Start monitoring thread for auto-scaling."""
        self._monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            name="scaling-monitor",
            daemon=True
        )
        self._monitoring_thread.start()
    
    def _monitoring_loop(self):
        """Monitoring loop for auto-scaling decisions."""
        while not self._shutdown_event.is_set():
            try:
                # Collect metrics from all workers
                self._collect_worker_metrics()
                
                # Make scaling decision
                self._evaluate_scaling()
                
                # Cleanup dead workers
                self._cleanup_dead_workers()
                
                time.sleep(5.0)  # Monitor every 5 seconds
                
            except Exception as e:
                if self.monitor:
                    self.monitor.alert(
                        AlertLevel.ERROR,
                        f"Monitoring error: {str(e)}",
                        "scaling_monitor"
                    )
    
    def _collect_worker_metrics(self):
        """Collect metrics from all workers."""
        for worker_id in list(self._workers.keys()):
            try:
                # Get system metrics for worker process
                worker_process = self._workers[worker_id]
                
                if worker_process.is_alive():
                    process = psutil.Process(worker_process.pid)
                    cpu_percent = process.cpu_percent()
                    memory_mb = process.memory_info().rss / (1024 * 1024)
                    
                    # Update worker metrics
                    metrics = self._worker_metrics[worker_id]
                    metrics.cpu_percent = cpu_percent
                    metrics.memory_mb = memory_mb
                    metrics.timestamp = time.time()
                    
            except (psutil.NoSuchProcess, ProcessLookupError):
                # Worker died, will be cleaned up
                pass
            except Exception as e:
                if self.monitor:
                    self.monitor.alert(
                        AlertLevel.WARNING,
                        f"Failed to collect metrics for worker {worker_id}: {str(e)}",
                        "metrics"
                    )
    
    def _evaluate_scaling(self):
        """Evaluate whether to scale up or down."""
        current_time = time.time()
        
        # Check cooldown period
        if current_time - self._last_scale_time < self.scaling_config.cooldown_seconds:
            return
        
        with self._scaling_lock:
            # Calculate aggregate metrics
            avg_cpu = np.mean([m.cpu_percent for m in self._worker_metrics.values()])
            avg_memory = np.mean([m.memory_mb for m in self._worker_metrics.values()])
            queue_size = self._request_queue.qsize()
            
            scale_decision = self._make_scaling_decision(
                avg_cpu, avg_memory, queue_size
            )
            
            if scale_decision == "scale_up":
                self._scale_up()
            elif scale_decision == "scale_down":
                self._scale_down()
    
    def _make_scaling_decision(self, avg_cpu: float, avg_memory: float, queue_size: int) -> str:
        """Make scaling decision based on metrics and strategy."""
        config = self.scaling_config
        
        if config.strategy == ScalingStrategy.CPU_BASED:
            if avg_cpu > config.target_cpu_percent * config.scale_up_threshold:
                return "scale_up"
            elif avg_cpu < config.target_cpu_percent * config.scale_down_threshold:
                return "scale_down"
                
        elif config.strategy == ScalingStrategy.MEMORY_BASED:
            memory_percent = (avg_memory / (psutil.virtual_memory().total / (1024**2))) * 100
            if memory_percent > config.target_memory_percent * config.scale_up_threshold:
                return "scale_up"
            elif memory_percent < config.target_memory_percent * config.scale_down_threshold:
                return "scale_down"
                
        elif config.strategy == ScalingStrategy.ADAPTIVE:
            # Composite scoring
            cpu_score = avg_cpu / config.target_cpu_percent
            memory_percent = (avg_memory / (psutil.virtual_memory().total / (1024**2))) * 100
            memory_score = memory_percent / config.target_memory_percent
            queue_score = queue_size / 100.0  # Normalize queue size
            
            composite_score = (cpu_score + memory_score + queue_score) / 3.0
            
            if composite_score > config.scale_up_threshold:
                return "scale_up"
            elif composite_score < config.scale_down_threshold:
                return "scale_down"
        
        return "no_change"
    
    def _scale_up(self):
        """Scale up by adding workers."""
        if self._current_workers >= self.scaling_config.max_workers:
            return
            
        # Add workers (up to 50% increase or max workers)
        num_to_add = min(
            max(1, int(self._current_workers * 0.5)),
            self.scaling_config.max_workers - self._current_workers
        )
        
        for _ in range(num_to_add):
            self._add_worker()
        
        self._last_scale_time = time.time()
        
        if self.monitor:
            self.monitor.alert(
                AlertLevel.INFO,
                f"Scaled up by {num_to_add} workers to {self._current_workers}",
                "scaling",
                {"action": "scale_up", "workers_added": num_to_add}
            )
    
    def _scale_down(self):
        """Scale down by removing workers."""
        if self._current_workers <= self.scaling_config.min_workers:
            return
            
        # Remove workers (up to 25% decrease or min workers)
        num_to_remove = min(
            max(1, int(self._current_workers * 0.25)),
            self._current_workers - self.scaling_config.min_workers
        )
        
        # Remove least utilized workers first
        worker_utilization = [
            (worker_id, metrics.cpu_percent + metrics.active_requests)
            for worker_id, metrics in self._worker_metrics.items()
        ]
        worker_utilization.sort(key=lambda x: x[1])  # Sort by utilization
        
        for i in range(num_to_remove):
            worker_id = worker_utilization[i][0]
            self._remove_worker(worker_id)
        
        self._last_scale_time = time.time()
        
        if self.monitor:
            self.monitor.alert(
                AlertLevel.INFO,
                f"Scaled down by {num_to_remove} workers to {self._current_workers}",
                "scaling",
                {"action": "scale_down", "workers_removed": num_to_remove}
            )
    
    def _cleanup_dead_workers(self):
        """Remove dead workers from pool."""
        dead_workers = []
        
        for worker_id, worker in self._workers.items():
            if not worker.is_alive():
                dead_workers.append(worker_id)
        
        for worker_id in dead_workers:
            if self.monitor:
                self.monitor.alert(
                    AlertLevel.WARNING,
                    f"Worker {worker_id} died, removing from pool",
                    "worker_failure",
                    {"worker_id": worker_id}
                )
            self._remove_worker(worker_id)
    
    def submit_inference(self, input_data: jnp.ndarray, request_id: str = None) -> str:
        """Submit inference request to pool."""
        if request_id is None:
            request_id = f"req_{int(time.time() * 1000000)}"
        
        request = (request_id, input_data, time.perf_counter())
        
        try:
            self._request_queue.put(request, timeout=1.0)
            return request_id
        except queue.Full:
            if self.monitor:
                self.monitor.alert(
                    AlertLevel.WARNING,
                    "Request queue full, dropping request",
                    "queue_overflow"
                )
            raise RuntimeError("Request queue full")
    
    def get_result(self, request_id: str, timeout: float = 5.0) -> Optional[Dict[str, Any]]:
        """Get result for specific request."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            # Check all worker result queues
            for worker_id, result_queue in self._result_queues.items():
                try:
                    result = result_queue.get_nowait()
                    if result["request_id"] == request_id:
                        return result
                    else:
                        # Put back if not our request
                        result_queue.put(result)
                except queue.Empty:
                    continue
            
            time.sleep(0.001)  # Small sleep to prevent busy waiting
        
        return None  # Timeout
    
    def infer(self, input_data: jnp.ndarray, timeout: float = 5.0) -> jnp.ndarray:
        """Synchronous inference with automatic scaling."""
        request_id = self.submit_inference(input_data)
        result = self.get_result(request_id, timeout)
        
        if result is None:
            raise ModelInferenceError(
                f"Inference timeout after {timeout}s",
                severity=ErrorSeverity.HIGH,
                context={"timeout": timeout, "request_id": request_id}
            )
        
        return result["output"]
    
    def get_pool_statistics(self) -> Dict[str, Any]:
        """Get comprehensive pool statistics."""
        with self._scaling_lock:
            return {
                "current_workers": self._current_workers,
                "min_workers": self.scaling_config.min_workers,
                "max_workers": self.scaling_config.max_workers,
                "queue_size": self._request_queue.qsize(),
                "scaling_strategy": self.scaling_config.strategy.value,
                "last_scale_time": self._last_scale_time,
                "worker_metrics": {
                    wid: {
                        "cpu_percent": metrics.cpu_percent,
                        "memory_mb": metrics.memory_mb,
                        "active_requests": metrics.active_requests
                    }
                    for wid, metrics in self._worker_metrics.items()
                },
                "avg_cpu_percent": np.mean([m.cpu_percent for m in self._worker_metrics.values()]) if self._worker_metrics else 0,
                "total_memory_mb": sum(m.memory_mb for m in self._worker_metrics.values())
            }
    
    def shutdown(self, timeout: float = 10.0):
        """Graceful shutdown of worker pool."""
        print("\n🔄 Shutting down high-performance worker pool...")
        
        # Signal shutdown
        self._shutdown_event.set()
        
        # Stop accepting new requests
        for _ in range(self._current_workers):
            try:
                self._request_queue.put(None, timeout=1.0)  # Shutdown signal
            except queue.Full:
                pass
        
        # Wait for workers to finish
        start_time = time.time()
        for worker_id, worker in self._workers.items():
            remaining_time = timeout - (time.time() - start_time)
            if remaining_time > 0:
                worker.join(remaining_time)
            
            if worker.is_alive():
                worker.terminate()
                worker.join(1.0)
                
            if worker.is_alive():
                worker.kill()
        
        # Stop monitoring
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            self._monitoring_thread.join(timeout=2.0)
        
        print("   ✅ Worker pool shutdown complete")


class BatchProcessor:
    """Intelligent batch processing for improved throughput."""
    
    def __init__(self, max_batch_size: int = 32, timeout_ms: float = 5.0):
        self.max_batch_size = max_batch_size
        self.timeout_ms = timeout_ms
        self.pending_requests = []
        self.batch_lock = threading.Lock()
    
    def add_request(self, request: Tuple) -> bool:
        """Add request to batch. Returns True if batch is ready."""
        with self.batch_lock:
            self.pending_requests.append(request)
            return len(self.pending_requests) >= self.max_batch_size
    
    def get_batch(self, min_size: int = 1) -> List[Tuple]:
        """Get batch of requests."""
        with self.batch_lock:
            if len(self.pending_requests) >= min_size:
                batch = self.pending_requests[:self.max_batch_size]
                self.pending_requests = self.pending_requests[self.max_batch_size:]
                return batch
            return []


class LoadBalancer:
    """Intelligent load balancing for worker selection."""
    
    def __init__(self):
        self.worker_loads = {}
        self.round_robin_counter = 0
    
    def select_worker(self, worker_metrics: Dict[int, WorkerMetrics]) -> int:
        """Select best worker based on current load."""
        if not worker_metrics:
            return None
        
        # Find worker with lowest load (CPU + active requests)
        best_worker = min(
            worker_metrics.keys(),
            key=lambda wid: worker_metrics[wid].cpu_percent + worker_metrics[wid].active_requests
        )
        
        return best_worker


@contextmanager
def optimized_inference_context():
    """Context manager for optimized inference execution."""
    # JAX compilation optimizations
    original_flags = {}
    
    # Enable XLA optimizations
    jax_flags = [
        'jax_enable_x64=False',  # Use float32 for better performance
        'jax_enable_compilation_cache=True',
        'jax_persistent_cache_min_compile_time_secs=1'
    ]
    
    try:
        yield
    finally:
        # Restore original settings if needed
        pass


def optimize_model_for_inference(model: LiquidNN, 
                               example_input: jnp.ndarray,
                               compile_ahead: bool = True) -> Callable:
    """Optimize model for high-performance inference."""
    
    @functools.lru_cache(maxsize=128)
    def cached_inference(params_hash: int, input_data: jnp.ndarray, hidden_state: jnp.ndarray):
        """Cached inference function."""
        return model.apply(None, input_data, hidden_state, training=False)  # params passed via closure
    
    # Compile the model for the expected input shape
    @jax.jit
    def compiled_inference(params, input_data, hidden_state):
        return model.apply(params, input_data, hidden_state, training=False)
    
    if compile_ahead:
        # Trigger JIT compilation
        dummy_params = model.init(jax.random.PRNGKey(0), example_input, training=False)
        dummy_hidden = jnp.zeros((1, model.config.hidden_dim))
        _ = compiled_inference(dummy_params, example_input, dummy_hidden)
    
    return compiled_inference


class PerformanceOptimizer:
    """Advanced performance optimization for liquid networks."""
    
    def __init__(self, model: LiquidNN, monitor: Optional[LiquidNetworkMonitor] = None):
        self.model = model
        self.monitor = monitor
        self.optimized_functions = {}
        
    def optimize_for_batch_inference(self, 
                                   batch_size: int,
                                   compile_jit: bool = True) -> Callable:
        """Optimize model for batch inference."""
        
        if batch_size in self.optimized_functions:
            return self.optimized_functions[batch_size]
        
        def batch_inference(params, input_batch, hidden_batch):
            """Optimized batch inference."""
            # Vectorized inference across batch
            outputs = []
            new_hiddens = []
            
            for i in range(batch_size):
                output, new_hidden = self.model.apply(
                    params, 
                    input_batch[i:i+1], 
                    hidden_batch[i:i+1], 
                    training=False
                )
                outputs.append(output)
                new_hiddens.append(new_hidden)
            
            return jnp.concatenate(outputs, axis=0), jnp.concatenate(new_hiddens, axis=0)
        
        if compile_jit:
            batch_inference = jax.jit(batch_inference)
            
            # Pre-compile for this batch size
            dummy_input = jnp.ones((batch_size, self.model.config.input_dim))
            dummy_hidden = jnp.zeros((batch_size, self.model.config.hidden_dim))
            dummy_params = self.model.init(jax.random.PRNGKey(0), dummy_input[:1], training=False)
            
            _ = batch_inference(dummy_params, dummy_input, dummy_hidden)
        
        self.optimized_functions[batch_size] = batch_inference
        return batch_inference
    
    def benchmark_inference_speeds(self, input_shapes: List[Tuple[int, ...]], 
                                 iterations: int = 100) -> Dict[str, float]:
        """Benchmark inference speeds for different configurations."""
        results = {}
        
        for shape in input_shapes:
            key = f"shape_{shape}"
            
            # Create test data
            test_input = jax.random.normal(jax.random.PRNGKey(42), shape)
            test_hidden = jnp.zeros((shape[0], self.model.config.hidden_dim))
            
            # Initialize parameters
            params = self.model.init(jax.random.PRNGKey(0), test_input[:1], training=False)
            
            # Warm up JIT
            for _ in range(5):
                _ = self.model.apply(params, test_input[:1], test_hidden[:1], training=False)
            
            # Benchmark
            start_time = time.perf_counter()
            
            for _ in range(iterations):
                _ = self.model.apply(params, test_input[:1], test_hidden[:1], training=False)
            
            end_time = time.perf_counter()
            avg_time_us = (end_time - start_time) / iterations * 1e6
            
            results[key] = avg_time_us
            
            if self.monitor:
                self.monitor.alert(
                    AlertLevel.INFO,
                    f"Benchmark {key}: {avg_time_us:.1f}μs per inference",
                    "benchmark"
                )
        
        return results


# Convenience functions
def create_high_performance_pool(model_config: LiquidConfig,
                               scaling_config: Optional[ScalingConfig] = None,
                               monitor: Optional[LiquidNetworkMonitor] = None) -> HighPerformanceLiquidPool:
    """Create high-performance worker pool with sensible defaults."""
    if scaling_config is None:
        scaling_config = ScalingConfig()
    
    return HighPerformanceLiquidPool(model_config, scaling_config, monitor)


def auto_optimize_model(model: LiquidNN, 
                       target_latency_ms: float = 10.0,
                       target_throughput_fps: float = 100.0) -> Dict[str, Any]:
    """Automatically optimize model for target performance."""
    optimizer = PerformanceOptimizer(model)
    
    # Benchmark different configurations
    input_shapes = [(1, model.config.input_dim), (4, model.config.input_dim), (16, model.config.input_dim)]
    benchmark_results = optimizer.benchmark_inference_speeds(input_shapes)
    
    # Find optimal batch size for target throughput
    optimal_batch_size = 1
    for shape_key, latency_us in benchmark_results.items():
        batch_size = int(shape_key.split('_')[1].split(',')[0].replace('(', ''))
        throughput = 1000000 / latency_us  # Convert μs to FPS
        
        if throughput >= target_throughput_fps and latency_us / 1000 <= target_latency_ms:
            optimal_batch_size = max(optimal_batch_size, batch_size)
    
    return {
        "optimal_batch_size": optimal_batch_size,
        "benchmark_results": benchmark_results,
        "optimized_function": optimizer.optimize_for_batch_inference(optimal_batch_size)
    }
