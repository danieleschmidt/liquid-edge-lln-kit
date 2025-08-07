"""Comprehensive monitoring and observability for liquid neural networks."""

import time
import json
import threading
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import logging
from queue import Queue, Empty
from datetime import datetime, timezone
import sys
import psutil
import traceback
from contextlib import contextmanager

# OpenTelemetry imports (optional)
try:
    from opentelemetry import trace, metrics
    from opentelemetry.exporter.prometheus import PrometheusMetricReader
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.resources import Resource
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False

# Prometheus client (optional)
try:
    from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, start_http_server
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class PerformanceMetrics:
    """Performance metrics collection."""
    inference_time_us: float = 0.0
    energy_consumption_mw: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    throughput_fps: float = 0.0
    accuracy: float = 0.0
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "inference_time_us": self.inference_time_us,
            "energy_consumption_mw": self.energy_consumption_mw,
            "memory_usage_mb": self.memory_usage_mb,
            "cpu_usage_percent": self.cpu_usage_percent,
            "throughput_fps": self.throughput_fps,
            "accuracy": self.accuracy,
            "timestamp": self.timestamp
        }


@dataclass
class Alert:
    """System alert."""
    level: AlertLevel
    message: str
    component: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "level": self.level.value,
            "message": self.message,
            "component": self.component,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }


class HealthStatus(Enum):
    """System health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class LiquidNetworkMonitor:
    """Production-grade monitoring system for liquid neural networks."""
    
    def __init__(self, 
                 name: str = "liquid_network",
                 enable_prometheus: bool = True,
                 enable_opentelemetry: bool = True,
                 prometheus_port: int = 8000,
                 metrics_retention_seconds: int = 3600):
        
        self.name = name
        self.enable_prometheus = enable_prometheus and PROMETHEUS_AVAILABLE
        self.enable_opentelemetry = enable_opentelemetry and OTEL_AVAILABLE
        self.prometheus_port = prometheus_port
        self.metrics_retention_seconds = metrics_retention_seconds
        
        # Internal state
        self._start_time = time.time()
        self._metrics_queue = Queue(maxsize=10000)
        self._alerts_queue = Queue(maxsize=1000)
        self._health_status = HealthStatus.UNKNOWN
        self._performance_metrics: List[PerformanceMetrics] = []
        self._alerts: List[Alert] = []
        
        # Thread safety
        self._lock = threading.RLock()
        self._monitoring_thread = None
        self._running = False
        
        # Alert callbacks
        self._alert_callbacks: List[Callable[[Alert], None]] = []
        
        # Setup monitoring backends
        self._setup_logging()
        self._setup_prometheus()
        self._setup_opentelemetry()
        
        # Start monitoring thread
        self.start_monitoring()
    
    def _setup_logging(self):
        """Configure structured logging."""
        self.logger = logging.getLogger(f"liquid_edge.{self.name}")
        self.logger.setLevel(logging.INFO)
        
        # Create formatter for structured logs
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        if not self.logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
    
    def _setup_prometheus(self):
        """Setup Prometheus metrics."""
        if not self.enable_prometheus:
            return
            
        try:
            # Create custom registry to avoid conflicts
            self.prometheus_registry = CollectorRegistry()
            
            # Define metrics
            self.prom_inference_time = Histogram(
                'liquid_nn_inference_time_seconds',
                'Inference time in seconds',
                ['model_name'],
                registry=self.prometheus_registry
            )
            
            self.prom_energy_consumption = Gauge(
                'liquid_nn_energy_consumption_mw',
                'Energy consumption in milliwatts',
                ['model_name'],
                registry=self.prometheus_registry
            )
            
            self.prom_throughput = Gauge(
                'liquid_nn_throughput_fps',
                'Throughput in frames per second',
                ['model_name'],
                registry=self.prometheus_registry
            )
            
            self.prom_accuracy = Gauge(
                'liquid_nn_accuracy',
                'Model accuracy',
                ['model_name'],
                registry=self.prometheus_registry
            )
            
            self.prom_alerts_total = Counter(
                'liquid_nn_alerts_total',
                'Total number of alerts',
                ['level', 'component'],
                registry=self.prometheus_registry
            )
            
            self.prom_health_status = Gauge(
                'liquid_nn_health_status',
                'Health status (1=healthy, 0.5=degraded, 0=unhealthy)',
                ['model_name'],
                registry=self.prometheus_registry
            )
            
            # Start Prometheus server
            start_http_server(self.prometheus_port, registry=self.prometheus_registry)
            self.logger.info(f"Prometheus metrics server started on port {self.prometheus_port}")
            
        except Exception as e:
            self.logger.error(f"Failed to setup Prometheus: {e}")
            self.enable_prometheus = False
    
    def _setup_opentelemetry(self):
        """Setup OpenTelemetry instrumentation."""
        if not self.enable_opentelemetry:
            return
            
        try:
            # Create resource
            resource = Resource.create({
                "service.name": f"liquid-edge-{self.name}",
                "service.version": "0.1.0"
            })
            
            # Setup metrics
            reader = PrometheusMetricReader()
            provider = MeterProvider(resource=resource, metric_readers=[reader])
            metrics.set_meter_provider(provider)
            
            self.otel_meter = metrics.get_meter("liquid_edge")
            
            # Create meters
            self.otel_inference_histogram = self.otel_meter.create_histogram(
                "inference_time",
                description="Inference time in microseconds"
            )
            
            self.otel_energy_gauge = self.otel_meter.create_up_down_counter(
                "energy_consumption",
                description="Energy consumption in milliwatts"
            )
            
            self.logger.info("OpenTelemetry instrumentation configured")
            
        except Exception as e:
            self.logger.error(f"Failed to setup OpenTelemetry: {e}")
            self.enable_opentelemetry = False
    
    def start_monitoring(self):
        """Start background monitoring thread."""
        if self._running:
            return
            
        self._running = True
        self._monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            name=f"liquid-monitor-{self.name}",
            daemon=True
        )
        self._monitoring_thread.start()
        self.logger.info("Monitoring system started")
    
    def stop_monitoring(self):
        """Stop monitoring system."""
        self._running = False
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            self._monitoring_thread.join(timeout=5.0)
        self.logger.info("Monitoring system stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self._running:
            try:
                # Process metrics queue
                self._process_metrics_queue()
                
                # Process alerts queue
                self._process_alerts_queue()
                
                # Update system metrics
                self._update_system_metrics()
                
                # Cleanup old metrics
                self._cleanup_old_metrics()
                
                # Health check
                self._update_health_status()
                
                time.sleep(1.0)  # 1 second monitoring cycle
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                traceback.print_exc()
    
    def _process_metrics_queue(self):
        """Process queued performance metrics."""
        processed = 0
        while processed < 100:  # Limit batch size
            try:
                metrics = self._metrics_queue.get_nowait()
                self._store_metrics(metrics)
                processed += 1
            except Empty:
                break
    
    def _process_alerts_queue(self):
        """Process queued alerts."""
        processed = 0
        while processed < 50:  # Limit batch size
            try:
                alert = self._alerts_queue.get_nowait()
                self._handle_alert(alert)
                processed += 1
            except Empty:
                break
    
    def _store_metrics(self, metrics: PerformanceMetrics):
        """Store performance metrics."""
        with self._lock:
            self._performance_metrics.append(metrics)
            
            # Update Prometheus metrics
            if self.enable_prometheus:
                try:
                    self.prom_inference_time.labels(model_name=self.name).observe(
                        metrics.inference_time_us / 1e6  # Convert to seconds
                    )
                    self.prom_energy_consumption.labels(model_name=self.name).set(
                        metrics.energy_consumption_mw
                    )
                    self.prom_throughput.labels(model_name=self.name).set(
                        metrics.throughput_fps
                    )
                    self.prom_accuracy.labels(model_name=self.name).set(
                        metrics.accuracy
                    )
                except Exception as e:
                    self.logger.error(f"Error updating Prometheus metrics: {e}")
            
            # Update OpenTelemetry metrics
            if self.enable_opentelemetry:
                try:
                    self.otel_inference_histogram.record(metrics.inference_time_us)
                    self.otel_energy_gauge.add(metrics.energy_consumption_mw)
                except Exception as e:
                    self.logger.error(f"Error updating OpenTelemetry metrics: {e}")
    
    def _handle_alert(self, alert: Alert):
        """Handle alert processing."""
        with self._lock:
            self._alerts.append(alert)
            
            # Log alert
            log_method = getattr(self.logger, alert.level.value)
            log_method(f"[{alert.component}] {alert.message}")
            
            # Update Prometheus counter
            if self.enable_prometheus:
                try:
                    self.prom_alerts_total.labels(
                        level=alert.level.value,
                        component=alert.component
                    ).inc()
                except Exception as e:
                    self.logger.error(f"Error updating alert counter: {e}")
            
            # Call alert callbacks
            for callback in self._alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    self.logger.error(f"Error in alert callback: {e}")
    
    def _update_system_metrics(self):
        """Update system-level metrics."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=None)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_mb = memory.used / (1024 * 1024)
            
            # Store as system metrics
            system_metrics = PerformanceMetrics(
                cpu_usage_percent=cpu_percent,
                memory_usage_mb=memory_mb
            )
            
            if not self._metrics_queue.full():
                self._metrics_queue.put(system_metrics)
                
        except Exception as e:
            self.logger.error(f"Error updating system metrics: {e}")
    
    def _cleanup_old_metrics(self):
        """Remove old metrics to prevent memory leaks."""
        try:
            current_time = time.time()
            cutoff_time = current_time - self.metrics_retention_seconds
            
            with self._lock:
                # Clean metrics
                self._performance_metrics = [
                    m for m in self._performance_metrics 
                    if m.timestamp > cutoff_time
                ]
                
                # Clean alerts (keep last 1000)
                if len(self._alerts) > 1000:
                    self._alerts = self._alerts[-1000:]
                    
        except Exception as e:
            self.logger.error(f"Error cleaning old metrics: {e}")
    
    def _update_health_status(self):
        """Update overall health status."""
        try:
            # Simple health check based on recent alerts
            current_time = time.time()
            recent_alerts = [
                alert for alert in self._alerts[-10:]  # Last 10 alerts
                if current_time - alert.timestamp < 300  # Last 5 minutes
            ]
            
            critical_count = sum(1 for a in recent_alerts if a.level == AlertLevel.CRITICAL)
            error_count = sum(1 for a in recent_alerts if a.level == AlertLevel.ERROR)
            
            if critical_count > 0:
                new_status = HealthStatus.UNHEALTHY
            elif error_count > 3:
                new_status = HealthStatus.DEGRADED
            else:
                new_status = HealthStatus.HEALTHY
            
            if new_status != self._health_status:
                self._health_status = new_status
                self.logger.info(f"Health status changed to: {new_status.value}")
                
                # Update Prometheus health metric
                if self.enable_prometheus:
                    health_value = {
                        HealthStatus.HEALTHY: 1.0,
                        HealthStatus.DEGRADED: 0.5,
                        HealthStatus.UNHEALTHY: 0.0,
                        HealthStatus.UNKNOWN: -1.0
                    }.get(new_status, -1.0)
                    
                    self.prom_health_status.labels(model_name=self.name).set(health_value)
                    
        except Exception as e:
            self.logger.error(f"Error updating health status: {e}")
    
    @contextmanager
    def track_inference(self, model_name: str = None):
        """Context manager to track inference performance."""
        model_name = model_name or self.name
        start_time = time.perf_counter()
        start_memory = self._get_memory_usage()
        
        try:
            yield
        except Exception as e:
            # Track errors
            self.alert(
                AlertLevel.ERROR,
                f"Inference error: {str(e)}",
                "inference",
                {"model_name": model_name}
            )
            raise
        finally:
            end_time = time.perf_counter()
            end_memory = self._get_memory_usage()
            
            # Calculate metrics
            inference_time_us = (end_time - start_time) * 1e6
            memory_usage_mb = end_memory
            
            # Store metrics
            metrics = PerformanceMetrics(
                inference_time_us=inference_time_us,
                memory_usage_mb=memory_usage_mb
            )
            
            self.record_metrics(metrics)
    
    def record_metrics(self, metrics: PerformanceMetrics):
        """Record performance metrics."""
        if not self._metrics_queue.full():
            self._metrics_queue.put(metrics)
        else:
            self.logger.warning("Metrics queue full, dropping metrics")
    
    def alert(self, level: AlertLevel, message: str, component: str, metadata: Dict[str, Any] = None):
        """Send an alert."""
        alert = Alert(
            level=level,
            message=message,
            component=component,
            metadata=metadata or {}
        )
        
        if not self._alerts_queue.full():
            self._alerts_queue.put(alert)
        else:
            # Critical alerts should always be processed immediately
            if level == AlertLevel.CRITICAL:
                self._handle_alert(alert)
    
    def add_alert_callback(self, callback: Callable[[Alert], None]):
        """Add callback function for alert processing."""
        self._alert_callbacks.append(callback)
    
    def get_health_status(self) -> HealthStatus:
        """Get current health status."""
        return self._health_status
    
    def get_recent_metrics(self, seconds: int = 60) -> List[PerformanceMetrics]:
        """Get recent performance metrics."""
        cutoff_time = time.time() - seconds
        with self._lock:
            return [
                m for m in self._performance_metrics
                if m.timestamp > cutoff_time
            ]
    
    def get_recent_alerts(self, seconds: int = 300) -> List[Alert]:
        """Get recent alerts."""
        cutoff_time = time.time() - seconds
        with self._lock:
            return [
                a for a in self._alerts
                if a.timestamp > cutoff_time
            ]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get monitoring statistics."""
        with self._lock:
            recent_metrics = self.get_recent_metrics(300)  # Last 5 minutes
            recent_alerts = self.get_recent_alerts(300)
            
            if recent_metrics:
                inference_times = [m.inference_time_us for m in recent_metrics if m.inference_time_us > 0]
                energy_values = [m.energy_consumption_mw for m in recent_metrics if m.energy_consumption_mw > 0]
                
                stats = {
                    "uptime_seconds": time.time() - self._start_time,
                    "health_status": self._health_status.value,
                    "total_metrics_collected": len(self._performance_metrics),
                    "total_alerts": len(self._alerts),
                    "recent_alerts_count": len(recent_alerts),
                    "avg_inference_time_us": sum(inference_times) / len(inference_times) if inference_times else 0,
                    "avg_energy_consumption_mw": sum(energy_values) / len(energy_values) if energy_values else 0,
                    "prometheus_enabled": self.enable_prometheus,
                    "opentelemetry_enabled": self.enable_opentelemetry
                }
            else:
                stats = {
                    "uptime_seconds": time.time() - self._start_time,
                    "health_status": self._health_status.value,
                    "total_metrics_collected": 0,
                    "total_alerts": len(self._alerts),
                    "recent_alerts_count": len(recent_alerts),
                    "prometheus_enabled": self.enable_prometheus,
                    "opentelemetry_enabled": self.enable_opentelemetry
                }
            
            return stats
    
    def export_metrics(self, filepath: str, format: str = "json"):
        """Export metrics to file."""
        try:
            with self._lock:
                data = {
                    "export_timestamp": time.time(),
                    "monitoring_statistics": self.get_statistics(),
                    "recent_metrics": [m.to_dict() for m in self.get_recent_metrics(3600)],
                    "recent_alerts": [a.to_dict() for a in self.get_recent_alerts(3600)]
                }
            
            if format.lower() == "json":
                with open(filepath, 'w') as f:
                    json.dump(data, f, indent=2)
            else:
                raise ValueError(f"Unsupported export format: {format}")
                
            self.logger.info(f"Metrics exported to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to export metrics: {e}")
            raise
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except Exception:
            return 0.0
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_monitoring()


class CircuitBreaker:
    """Circuit breaker pattern implementation for robustness."""
    
    def __init__(self, 
                 failure_threshold: int = 5,
                 timeout_seconds: float = 60.0,
                 expected_exception: type = Exception):
        self.failure_threshold = failure_threshold
        self.timeout_seconds = timeout_seconds
        self.expected_exception = expected_exception
        
        self._failure_count = 0
        self._last_failure_time = None
        self._state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self._lock = threading.Lock()
    
    def __call__(self, func):
        """Decorator for circuit breaker."""
        def wrapper(*args, **kwargs):
            return self._call_with_circuit_breaker(func, *args, **kwargs)
        return wrapper
    
    def _call_with_circuit_breaker(self, func, *args, **kwargs):
        """Execute function with circuit breaker logic."""
        with self._lock:
            if self._state == "OPEN":
                if self._should_attempt_reset():
                    self._state = "HALF_OPEN"
                else:
                    raise RuntimeError("Circuit breaker is OPEN")
            
            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
                
            except self.expected_exception as e:
                self._on_failure()
                raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset."""
        return (
            self._last_failure_time and
            time.time() - self._last_failure_time >= self.timeout_seconds
        )
    
    def _on_success(self):
        """Handle successful execution."""
        self._failure_count = 0
        self._state = "CLOSED"
    
    def _on_failure(self):
        """Handle failed execution."""
        self._failure_count += 1
        self._last_failure_time = time.time()
        
        if self._failure_count >= self.failure_threshold:
            self._state = "OPEN"
    
    @property
    def state(self) -> str:
        """Get current circuit breaker state."""
        return self._state


# Factory function for easy monitoring setup
def create_monitor(name: str = "liquid_network", **kwargs) -> LiquidNetworkMonitor:
    """Create a monitoring instance with sensible defaults."""
    return LiquidNetworkMonitor(name=name, **kwargs)
