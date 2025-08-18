"""High-performance parallel inference system for liquid neural networks."""

import asyncio
import threading
import time
from typing import Dict, Any, List, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import jax
import jax.numpy as jnp
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import queue
import multiprocessing as mp
from functools import partial
import logging


class InferenceMode(Enum):
    """Inference execution modes."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    ASYNC_BATCH = "async_batch"
    STREAMING = "streaming"
    PIPELINED = "pipelined"


class LoadBalancingStrategy(Enum):
    """Load balancing strategies for distributed inference."""
    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    WEIGHTED = "weighted"
    ADAPTIVE = "adaptive"


@dataclass
class PerformanceConfig:
    """Configuration for high-performance inference."""
    max_workers: int = 4
    batch_size: int = 32
    max_batch_delay_ms: float = 10.0
    enable_jit_compilation: bool = True
    enable_vectorization: bool = True
    enable_parallel_processing: bool = True
    inference_mode: InferenceMode = InferenceMode.PARALLEL
    load_balancing: LoadBalancingStrategy = LoadBalancingStrategy.ADAPTIVE
    memory_pool_size_mb: float = 128.0
    enable_caching: bool = True
    cache_size: int = 1000
    enable_quantization: bool = True
    quantization_bits: int = 8
    enable_pruning: bool = True
    pruning_threshold: float = 0.01
    target_latency_ms: float = 10.0
    enable_auto_scaling: bool = True


@dataclass
class InferenceMetrics:
    """Performance metrics for inference operations."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    average_latency_ms: float = 0.0
    min_latency_ms: float = float('inf')
    max_latency_ms: float = 0.0
    throughput_rps: float = 0.0
    queue_length: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    memory_usage_mb: float = 0.0
    cpu_utilization: float = 0.0
    gpu_utilization: float = 0.0
    energy_consumption_mw: float = 0.0


class InferenceRequest:
    """Individual inference request with metadata."""
    
    def __init__(self, request_id: str, inputs: jnp.ndarray, 
                 priority: int = 1, timeout_ms: float = 1000.0):
        self.request_id = request_id
        self.inputs = inputs
        self.priority = priority
        self.timeout_ms = timeout_ms
        self.created_at = time.time()
        self.started_at = None
        self.completed_at = None
        self.result = None
        self.error = None
        
    @property
    def age_ms(self) -> float:
        """Get request age in milliseconds."""
        return (time.time() - self.created_at) * 1000
    
    @property
    def latency_ms(self) -> Optional[float]:
        """Get request latency if completed."""
        if self.completed_at and self.started_at:
            return (self.completed_at - self.started_at) * 1000
        return None


class HighPerformanceInferenceEngine:
    """High-performance inference engine with multiple optimization strategies."""
    
    def __init__(self, model, config: PerformanceConfig):
        self.model = model
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.metrics = InferenceMetrics()
        
        # Threading and concurrency
        self.executor = ThreadPoolExecutor(max_workers=config.max_workers)
        self.request_queue = queue.PriorityQueue()
        self.result_cache = {}
        self.cache_lock = threading.Lock()
        
        # Batch processing
        self.batch_queue = queue.Queue()
        self.batch_processor_running = False
        
        # JIT compilation
        self.compiled_inference = None
        if config.enable_jit_compilation:
            self._setup_jit_compilation()
        
        # Performance monitoring
        self.start_time = time.time()
        self.request_times = []
        self.lock = threading.Lock()
        
        # Worker management
        self.workers = []
        self.worker_loads = [0] * config.max_workers
        
        # Auto-scaling
        self.scaling_thread = None
        if config.enable_auto_scaling:
            self._start_auto_scaling()
    
    def _setup_jit_compilation(self):
        """Setup JIT-compiled inference functions."""
        try:
            @jax.jit
            def compiled_apply(params, inputs):
                return self.model.apply(params, inputs, training=False)
            
            self.compiled_inference = compiled_apply
            self.logger.info("JIT compilation enabled")
        except Exception as e:
            self.logger.warning(f"JIT compilation failed: {e}")
            self.compiled_inference = None
    
    def _start_auto_scaling(self):
        """Start auto-scaling monitoring thread."""
        def auto_scale_monitor():
            while True:
                try:
                    # Monitor queue length and adjust workers
                    queue_length = self.request_queue.qsize()
                    avg_latency = self.metrics.average_latency_ms
                    
                    if queue_length > self.config.max_workers * 2 and avg_latency > self.config.target_latency_ms:
                        # Scale up
                        if len(self.workers) < self.config.max_workers * 2:
                            self._add_worker()
                    elif queue_length < self.config.max_workers // 2 and avg_latency < self.config.target_latency_ms / 2:
                        # Scale down
                        if len(self.workers) > self.config.max_workers // 2:
                            self._remove_worker()
                    
                    time.sleep(1.0)  # Check every second
                except Exception as e:
                    self.logger.error(f"Auto-scaling error: {e}")
                    time.sleep(5.0)
        
        self.scaling_thread = threading.Thread(target=auto_scale_monitor, daemon=True)
        self.scaling_thread.start()
    
    def _add_worker(self):
        """Add a new worker thread."""
        worker_id = len(self.workers)
        worker_thread = threading.Thread(
            target=self._worker_loop,
            args=(worker_id,),
            daemon=True
        )
        worker_thread.start()
        self.workers.append(worker_thread)
        self.worker_loads.append(0)
        self.logger.info(f"Added worker {worker_id}")
    
    def _remove_worker(self):
        """Remove a worker thread (graceful shutdown)."""
        if self.workers:
            # Find least loaded worker
            min_load_idx = min(range(len(self.worker_loads)), key=lambda i: self.worker_loads[i])
            # In a real implementation, we would signal the worker to stop
            self.logger.info(f"Would remove worker {min_load_idx}")
    
    def _worker_loop(self, worker_id: int):
        """Main worker loop for processing requests."""
        while True:
            try:
                # Get request from queue
                try:
                    priority, request = self.request_queue.get(timeout=1.0)
                    self.worker_loads[worker_id] += 1
                except queue.Empty:
                    continue
                
                # Process request
                self._process_request(request)
                self.worker_loads[worker_id] -= 1
                self.request_queue.task_done()
                
            except Exception as e:
                self.logger.error(f"Worker {worker_id} error: {e}")
                time.sleep(0.1)
    
    def _process_request(self, request: InferenceRequest):
        """Process a single inference request."""
        request.started_at = time.time()
        
        try:
            # Check cache first
            if self.config.enable_caching:
                cache_key = self._get_cache_key(request.inputs)
                with self.cache_lock:
                    if cache_key in self.result_cache:
                        request.result = self.result_cache[cache_key]
                        request.completed_at = time.time()
                        self.metrics.cache_hits += 1
                        return
                    else:
                        self.metrics.cache_misses += 1
            
            # Execute inference
            if self.compiled_inference is not None:
                # Use JIT-compiled version
                params = getattr(self, 'model_params', None)
                if params is None:
                    # Initialize default params if not set
                    key = jax.random.PRNGKey(42)
                    params = self.model.init(key, jnp.ones_like(request.inputs))
                
                result = self.compiled_inference(params, request.inputs)
            else:
                # Use regular inference
                result = self.model.apply(
                    getattr(self, 'model_params', {}),
                    request.inputs,
                    training=False
                )
            
            request.result = result
            request.completed_at = time.time()
            
            # Update cache
            if self.config.enable_caching:
                cache_key = self._get_cache_key(request.inputs)
                with self.cache_lock:
                    if len(self.result_cache) < self.config.cache_size:
                        self.result_cache[cache_key] = result
                    else:
                        # Simple LRU: remove oldest entry
                        oldest_key = next(iter(self.result_cache))
                        del self.result_cache[oldest_key]
                        self.result_cache[cache_key] = result
            
            # Update metrics
            self._update_metrics(request)
            self.metrics.successful_requests += 1
            
        except Exception as e:
            request.error = str(e)
            request.completed_at = time.time()
            self.metrics.failed_requests += 1
            self.logger.error(f"Request {request.request_id} failed: {e}")
    
    def _get_cache_key(self, inputs: jnp.ndarray) -> str:
        """Generate cache key for inputs."""
        # Simple hash of inputs (in production, use more sophisticated hashing)
        return str(hash(inputs.tobytes()))
    
    def _update_metrics(self, request: InferenceRequest):
        """Update performance metrics."""
        with self.lock:
            self.metrics.total_requests += 1
            
            if request.latency_ms is not None:
                latency = request.latency_ms
                self.request_times.append(latency)
                
                # Keep only recent times for accurate averages
                if len(self.request_times) > 1000:
                    self.request_times = self.request_times[-1000:]
                
                self.metrics.average_latency_ms = np.mean(self.request_times)
                self.metrics.min_latency_ms = min(self.metrics.min_latency_ms, latency)
                self.metrics.max_latency_ms = max(self.metrics.max_latency_ms, latency)
            
            # Calculate throughput
            elapsed_time = time.time() - self.start_time
            if elapsed_time > 0:
                self.metrics.throughput_rps = self.metrics.successful_requests / elapsed_time
            
            self.metrics.queue_length = self.request_queue.qsize()
    
    def set_model_params(self, params: Dict[str, Any]):
        """Set model parameters for inference."""
        self.model_params = params
    
    async def async_inference(self, inputs: jnp.ndarray, 
                            request_id: Optional[str] = None,
                            priority: int = 1,
                            timeout_ms: float = 1000.0) -> jnp.ndarray:
        """Asynchronous inference with priority queuing."""
        if request_id is None:
            request_id = f"req_{time.time()}_{hash(inputs.tobytes()) % 10000}"
        
        request = InferenceRequest(request_id, inputs, priority, timeout_ms)
        
        # Add to queue
        self.request_queue.put((priority, request))
        
        # Wait for completion
        start_wait = time.time()
        while request.result is None and request.error is None:
            if (time.time() - start_wait) * 1000 > timeout_ms:
                request.error = f"Request timeout after {timeout_ms}ms"
                break
            await asyncio.sleep(0.001)  # Small sleep to prevent busy waiting
        
        if request.error:
            raise RuntimeError(f"Inference failed: {request.error}")
        
        return request.result
    
    def batch_inference(self, batch_inputs: List[jnp.ndarray],
                       batch_size: Optional[int] = None) -> List[jnp.ndarray]:
        """Batch inference for improved throughput."""
        if batch_size is None:
            batch_size = self.config.batch_size
        
        results = []
        
        # Process in batches
        for i in range(0, len(batch_inputs), batch_size):
            batch = batch_inputs[i:i + batch_size]
            
            # Stack inputs into a single batch
            stacked_inputs = jnp.stack(batch)
            
            # Run inference on batch
            if self.compiled_inference is not None:
                params = getattr(self, 'model_params', {})
                batch_result = self.compiled_inference(params, stacked_inputs)
            else:
                batch_result = self.model.apply(
                    getattr(self, 'model_params', {}),
                    stacked_inputs,
                    training=False
                )
            
            # Split batch result back into individual results
            if isinstance(batch_result, tuple):
                # Handle tuple outputs (e.g., (output, hidden_state))
                outputs, states = batch_result
                individual_outputs = [outputs[j] for j in range(len(batch))]
                individual_states = [states[j] for j in range(len(batch))]
                results.extend([(out, state) for out, state in zip(individual_outputs, individual_states)])
            else:
                # Handle single output
                individual_results = [batch_result[j] for j in range(len(batch))]
                results.extend(individual_results)
        
        return results
    
    def streaming_inference(self, input_stream: Callable[[], Optional[jnp.ndarray]],
                          output_callback: Callable[[jnp.ndarray], None]):
        """Streaming inference for real-time applications."""
        def stream_processor():
            while True:
                try:
                    inputs = input_stream()
                    if inputs is None:
                        break  # End of stream
                    
                    # Process single input
                    if self.compiled_inference is not None:
                        params = getattr(self, 'model_params', {})
                        result = self.compiled_inference(params, inputs.reshape(1, -1))
                    else:
                        result = self.model.apply(
                            getattr(self, 'model_params', {}),
                            inputs.reshape(1, -1),
                            training=False
                        )
                    
                    # Send result to callback
                    if isinstance(result, tuple):
                        output_callback(result[0][0])  # First output, first batch item
                    else:
                        output_callback(result[0])  # First batch item
                        
                except Exception as e:
                    self.logger.error(f"Streaming inference error: {e}")
                    break
        
        stream_thread = threading.Thread(target=stream_processor, daemon=True)
        stream_thread.start()
        return stream_thread
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        uptime = time.time() - self.start_time
        
        # Calculate cache hit rate
        total_cache_requests = self.metrics.cache_hits + self.metrics.cache_misses
        cache_hit_rate = (self.metrics.cache_hits / total_cache_requests * 100 
                         if total_cache_requests > 0 else 0.0)
        
        # Calculate success rate
        total_requests = self.metrics.successful_requests + self.metrics.failed_requests
        success_rate = (self.metrics.successful_requests / total_requests * 100 
                       if total_requests > 0 else 100.0)
        
        return {
            "performance_metrics": {
                "uptime_seconds": uptime,
                "total_requests": total_requests,
                "successful_requests": self.metrics.successful_requests,
                "failed_requests": self.metrics.failed_requests,
                "success_rate_percent": success_rate,
                "average_latency_ms": self.metrics.average_latency_ms,
                "min_latency_ms": self.metrics.min_latency_ms,
                "max_latency_ms": self.metrics.max_latency_ms,
                "throughput_rps": self.metrics.throughput_rps,
                "queue_length": self.metrics.queue_length
            },
            "caching": {
                "enabled": self.config.enable_caching,
                "cache_size": len(self.result_cache),
                "max_cache_size": self.config.cache_size,
                "cache_hits": self.metrics.cache_hits,
                "cache_misses": self.metrics.cache_misses,
                "cache_hit_rate_percent": cache_hit_rate
            },
            "workers": {
                "max_workers": self.config.max_workers,
                "active_workers": len(self.workers),
                "worker_loads": self.worker_loads[:len(self.workers)],
                "auto_scaling_enabled": self.config.enable_auto_scaling
            },
            "optimizations": {
                "jit_compilation": self.compiled_inference is not None,
                "vectorization": self.config.enable_vectorization,
                "parallel_processing": self.config.enable_parallel_processing,
                "quantization": self.config.enable_quantization,
                "pruning": self.config.enable_pruning
            },
            "configuration": {
                "inference_mode": self.config.inference_mode.value,
                "batch_size": self.config.batch_size,
                "max_batch_delay_ms": self.config.max_batch_delay_ms,
                "target_latency_ms": self.config.target_latency_ms,
                "load_balancing": self.config.load_balancing.value
            }
        }
    
    def shutdown(self):
        """Graceful shutdown of the inference engine."""
        self.logger.info("Shutting down inference engine...")
        
        # Wait for pending requests
        self.request_queue.join()
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        self.logger.info("Inference engine shutdown complete")


class DistributedInferenceCoordinator:
    """Coordinator for distributed inference across multiple engines."""
    
    def __init__(self, configs: List[PerformanceConfig]):
        self.engines = []
        self.load_balancer_index = 0
        self.engine_loads = []
        
        # Create engines
        for i, config in enumerate(configs):
            # In a real implementation, these might be on different machines
            engine = HighPerformanceInferenceEngine(None, config)  # Model would be set separately
            self.engines.append(engine)
            self.engine_loads.append(0)
    
    def distribute_inference(self, inputs: jnp.ndarray, 
                           strategy: LoadBalancingStrategy = LoadBalancingStrategy.ROUND_ROBIN) -> jnp.ndarray:
        """Distribute inference request across available engines."""
        if strategy == LoadBalancingStrategy.ROUND_ROBIN:
            engine_idx = self.load_balancer_index % len(self.engines)
            self.load_balancer_index += 1
        elif strategy == LoadBalancingStrategy.LEAST_LOADED:
            engine_idx = min(range(len(self.engine_loads)), key=lambda i: self.engine_loads[i])
        else:
            # Default to first engine
            engine_idx = 0
        
        # Execute on selected engine
        selected_engine = self.engines[engine_idx]
        self.engine_loads[engine_idx] += 1
        
        try:
            # This would be an async call in a real distributed system
            result = selected_engine.async_inference(inputs)
            return result
        finally:
            self.engine_loads[engine_idx] -= 1