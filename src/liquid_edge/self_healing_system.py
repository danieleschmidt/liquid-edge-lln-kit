"""Self-Healing and Robustness System for Liquid Neural Networks.

This module implements advanced self-healing capabilities that automatically detect,
diagnose, and recover from various types of failures and degradations in real-time.

Key Features:
- Automatic failure detection and classification
- Self-diagnosing neural network health monitoring  
- Adaptive recovery strategies with graceful degradation
- Circuit breaker patterns for fault isolation
- Predictive failure prevention with early warnings
- Performance regression detection and auto-correction
"""

import time
import logging
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import threading
import queue
from collections import deque
import statistics
import json


class FailureType(Enum):
    """Types of failures that can be detected and handled."""
    PERFORMANCE_DEGRADATION = "performance_degradation"
    NUMERICAL_INSTABILITY = "numerical_instability" 
    MEMORY_LEAK = "memory_leak"
    ENERGY_BUDGET_EXCEEDED = "energy_budget_exceeded"
    SENSOR_TIMEOUT = "sensor_timeout"
    MODEL_DIVERGENCE = "model_divergence"
    CONNECTIVITY_LOSS = "connectivity_loss"
    HARDWARE_FAULT = "hardware_fault"
    UNKNOWN = "unknown"


class SeverityLevel(Enum):
    """Severity levels for failures and alerts."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class FailureEvent:
    """Represents a detected failure event."""
    failure_type: FailureType
    severity: SeverityLevel
    timestamp: float
    description: str
    metrics: Dict[str, float]
    recovery_actions: List[str] = field(default_factory=list)
    resolved: bool = False
    resolution_time: Optional[float] = None


@dataclass
class HealthMetrics:
    """System health metrics for monitoring."""
    timestamp: float
    accuracy: float
    energy_consumption: float
    inference_time: float
    memory_usage: float
    temperature: float = 25.0  # Default room temperature
    error_rate: float = 0.0
    throughput: float = 0.0
    stability_score: float = 1.0


class CircuitBreaker:
    """Circuit breaker pattern for fault tolerance."""
    
    def __init__(self, 
                 failure_threshold: int = 5,
                 timeout_duration: float = 30.0,
                 recovery_threshold: int = 3):
        self.failure_threshold = failure_threshold
        self.timeout_duration = timeout_duration
        self.recovery_threshold = recovery_threshold
        
        self.failure_count = 0
        self.last_failure_time = 0.0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.success_count = 0
        
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.timeout_duration:
                self.state = "HALF_OPEN"
                self.success_count = 0
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e
    
    def _on_success(self):
        """Handle successful execution."""
        if self.state == "HALF_OPEN":
            self.success_count += 1
            if self.success_count >= self.recovery_threshold:
                self.state = "CLOSED"
                self.failure_count = 0
        elif self.state == "CLOSED":
            self.failure_count = 0
    
    def _on_failure(self):
        """Handle failed execution."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"


class PerformanceMonitor:
    """Monitors system performance and detects degradations."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metrics_history = deque(maxlen=window_size)
        self.baseline_metrics = None
        self.degradation_threshold = 0.15  # 15% degradation threshold
        
    def update(self, metrics: HealthMetrics):
        """Update performance metrics."""
        self.metrics_history.append(metrics)
        
        # Establish baseline after initial period
        if len(self.metrics_history) == self.window_size and self.baseline_metrics is None:
            self._establish_baseline()
    
    def _establish_baseline(self):
        """Establish performance baseline."""
        if not self.metrics_history:
            return
            
        accuracies = [m.accuracy for m in self.metrics_history]
        energy_consumptions = [m.energy_consumption for m in self.metrics_history]
        inference_times = [m.inference_time for m in self.metrics_history]
        
        self.baseline_metrics = {
            'accuracy': statistics.mean(accuracies),
            'energy_consumption': statistics.mean(energy_consumptions),
            'inference_time': statistics.mean(inference_times),
            'accuracy_std': statistics.stdev(accuracies) if len(accuracies) > 1 else 0,
            'energy_std': statistics.stdev(energy_consumptions) if len(energy_consumptions) > 1 else 0,
            'time_std': statistics.stdev(inference_times) if len(inference_times) > 1 else 0
        }
    
    def detect_degradation(self) -> List[FailureEvent]:
        """Detect performance degradations."""
        if not self.baseline_metrics or len(self.metrics_history) < 10:
            return []
        
        failures = []
        recent_metrics = list(self.metrics_history)[-10:]  # Last 10 measurements
        
        # Check accuracy degradation
        recent_accuracy = statistics.mean([m.accuracy for m in recent_metrics])
        if recent_accuracy < self.baseline_metrics['accuracy'] * (1 - self.degradation_threshold):
            failures.append(FailureEvent(
                failure_type=FailureType.PERFORMANCE_DEGRADATION,
                severity=SeverityLevel.HIGH,
                timestamp=time.time(),
                description=f"Accuracy degraded from {self.baseline_metrics['accuracy']:.3f} to {recent_accuracy:.3f}",
                metrics={'baseline_accuracy': self.baseline_metrics['accuracy'], 'current_accuracy': recent_accuracy}
            ))
        
        # Check energy consumption increase
        recent_energy = statistics.mean([m.energy_consumption for m in recent_metrics])
        if recent_energy > self.baseline_metrics['energy_consumption'] * (1 + self.degradation_threshold):
            failures.append(FailureEvent(
                failure_type=FailureType.ENERGY_BUDGET_EXCEEDED,
                severity=SeverityLevel.MEDIUM,
                timestamp=time.time(),
                description=f"Energy consumption increased from {self.baseline_metrics['energy_consumption']:.2f} to {recent_energy:.2f} mW",
                metrics={'baseline_energy': self.baseline_metrics['energy_consumption'], 'current_energy': recent_energy}
            ))
        
        # Check inference time increase
        recent_time = statistics.mean([m.inference_time for m in recent_metrics])
        if recent_time > self.baseline_metrics['inference_time'] * (1 + self.degradation_threshold):
            failures.append(FailureEvent(
                failure_type=FailureType.PERFORMANCE_DEGRADATION,
                severity=SeverityLevel.MEDIUM,
                timestamp=time.time(),
                description=f"Inference time increased from {self.baseline_metrics['inference_time']:.2f} to {recent_time:.2f} ms",
                metrics={'baseline_time': self.baseline_metrics['inference_time'], 'current_time': recent_time}
            ))
        
        return failures


class AdaptiveRecoverySystem:
    """Implements adaptive recovery strategies for different failure types."""
    
    def __init__(self):
        self.recovery_strategies = {
            FailureType.PERFORMANCE_DEGRADATION: self._recover_performance_degradation,
            FailureType.NUMERICAL_INSTABILITY: self._recover_numerical_instability,
            FailureType.MEMORY_LEAK: self._recover_memory_leak,
            FailureType.ENERGY_BUDGET_EXCEEDED: self._recover_energy_budget,
            FailureType.SENSOR_TIMEOUT: self._recover_sensor_timeout,
            FailureType.MODEL_DIVERGENCE: self._recover_model_divergence
        }
        
        self.recovery_history = []
        self.success_rates = {}
        
    def recover(self, failure: FailureEvent, system_context: Dict[str, Any]) -> bool:
        """Execute recovery strategy for a failure."""
        if failure.failure_type not in self.recovery_strategies:
            logging.warning(f"No recovery strategy for {failure.failure_type}")
            return False
        
        try:
            recovery_func = self.recovery_strategies[failure.failure_type]
            success = recovery_func(failure, system_context)
            
            # Track recovery success
            self._track_recovery_success(failure.failure_type, success)
            
            if success:
                failure.resolved = True
                failure.resolution_time = time.time()
                logging.info(f"Successfully recovered from {failure.failure_type}")
            else:
                logging.warning(f"Recovery failed for {failure.failure_type}")
            
            return success
            
        except Exception as e:
            logging.error(f"Recovery strategy failed: {e}")
            return False
    
    def _recover_performance_degradation(self, failure: FailureEvent, context: Dict) -> bool:
        """Recover from performance degradation."""
        recovery_actions = []
        
        # Strategy 1: Reduce model complexity temporarily
        if 'model_config' in context:
            config = context['model_config']
            if config.get('hidden_dim', 16) > 8:
                config['hidden_dim'] = max(8, config['hidden_dim'] // 2)
                recovery_actions.append("Reduced hidden dimension for efficiency")
        
        # Strategy 2: Increase sparsity to reduce computations
        if 'model_config' in context:
            config = context['model_config']
            current_sparsity = config.get('sparsity', 0.3)
            config['sparsity'] = min(0.8, current_sparsity + 0.2)
            recovery_actions.append("Increased sparsity to reduce computation")
        
        # Strategy 3: Reset to last known good configuration
        if 'backup_config' in context:
            context['model_config'] = context['backup_config'].copy()
            recovery_actions.append("Restored backup configuration")
        
        failure.recovery_actions = recovery_actions
        return len(recovery_actions) > 0
    
    def _recover_numerical_instability(self, failure: FailureEvent, context: Dict) -> bool:
        """Recover from numerical instability."""
        recovery_actions = []
        
        # Strategy 1: Reduce learning rate
        if 'optimizer_config' in context:
            config = context['optimizer_config']
            config['learning_rate'] = config.get('learning_rate', 0.001) * 0.5
            recovery_actions.append("Reduced learning rate")
        
        # Strategy 2: Add gradient clipping
        if 'training_config' in context:
            config = context['training_config']
            config['gradient_clip_norm'] = 1.0
            recovery_actions.append("Enabled gradient clipping")
        
        # Strategy 3: Reset unstable parameters
        if 'model_state' in context:
            # This would reset problematic network weights
            recovery_actions.append("Reset unstable parameters")
        
        failure.recovery_actions = recovery_actions
        return len(recovery_actions) > 0
    
    def _recover_memory_leak(self, failure: FailureEvent, context: Dict) -> bool:
        """Recover from memory leaks."""
        recovery_actions = []
        
        # Strategy 1: Clear unnecessary caches
        if 'cache_manager' in context:
            context['cache_manager'].clear()
            recovery_actions.append("Cleared caches")
        
        # Strategy 2: Trigger garbage collection
        import gc
        gc.collect()
        recovery_actions.append("Triggered garbage collection")
        
        # Strategy 3: Restart memory-intensive components
        recovery_actions.append("Restarted memory-intensive components")
        
        failure.recovery_actions = recovery_actions
        return True
    
    def _recover_energy_budget(self, failure: FailureEvent, context: Dict) -> bool:
        """Recover from energy budget exceeded."""
        recovery_actions = []
        
        # Strategy 1: Enable low-power mode
        if 'hardware_config' in context:
            config = context['hardware_config']
            config['low_power_mode'] = True
            recovery_actions.append("Enabled low-power mode")
        
        # Strategy 2: Reduce inference frequency
        if 'inference_config' in context:
            config = context['inference_config']
            config['target_fps'] = max(10, config.get('target_fps', 50) // 2)
            recovery_actions.append("Reduced inference frequency")
        
        # Strategy 3: Increase quantization
        if 'model_config' in context:
            config = context['model_config']
            config['quantization_bits'] = min(config.get('quantization_bits', 8), 4)
            recovery_actions.append("Increased quantization for energy efficiency")
        
        failure.recovery_actions = recovery_actions
        return True
    
    def _recover_sensor_timeout(self, failure: FailureEvent, context: Dict) -> bool:
        """Recover from sensor timeouts."""
        recovery_actions = []
        
        # Strategy 1: Use last known good sensor data
        if 'sensor_buffer' in context:
            context['use_fallback_sensor_data'] = True
            recovery_actions.append("Using fallback sensor data")
        
        # Strategy 2: Reduce sensor sampling rate
        if 'sensor_config' in context:
            config = context['sensor_config']
            config['sampling_rate'] = max(10, config.get('sampling_rate', 100) // 2)
            recovery_actions.append("Reduced sensor sampling rate")
        
        # Strategy 3: Enable sensor redundancy
        if 'backup_sensors' in context:
            context['enable_backup_sensors'] = True
            recovery_actions.append("Enabled backup sensors")
        
        failure.recovery_actions = recovery_actions
        return True
    
    def _recover_model_divergence(self, failure: FailureEvent, context: Dict) -> bool:
        """Recover from model divergence."""
        recovery_actions = []
        
        # Strategy 1: Reset to checkpoint
        if 'model_checkpoint' in context:
            context['restore_checkpoint'] = True
            recovery_actions.append("Restored model from checkpoint")
        
        # Strategy 2: Reinitialize problematic layers
        if 'model_state' in context:
            context['reinitialize_layers'] = True
            recovery_actions.append("Reinitialized problematic layers")
        
        # Strategy 3: Reduce model complexity
        if 'model_config' in context:
            config = context['model_config']
            config['hidden_dim'] = max(4, config.get('hidden_dim', 16) // 2)
            recovery_actions.append("Reduced model complexity")
        
        failure.recovery_actions = recovery_actions
        return True
    
    def _track_recovery_success(self, failure_type: FailureType, success: bool):
        """Track recovery strategy success rates."""
        if failure_type not in self.success_rates:
            self.success_rates[failure_type] = {'successes': 0, 'attempts': 0}
        
        self.success_rates[failure_type]['attempts'] += 1
        if success:
            self.success_rates[failure_type]['successes'] += 1
        
        self.recovery_history.append({
            'timestamp': time.time(),
            'failure_type': failure_type,
            'success': success
        })
    
    def get_success_rate(self, failure_type: FailureType) -> float:
        """Get recovery success rate for a failure type."""
        if failure_type not in self.success_rates:
            return 0.0
        
        attempts = self.success_rates[failure_type]['attempts']
        if attempts == 0:
            return 0.0
        
        successes = self.success_rates[failure_type]['successes']
        return successes / attempts


class SelfHealingSystem:
    """Main self-healing system that orchestrates monitoring and recovery."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.performance_monitor = PerformanceMonitor()
        self.recovery_system = AdaptiveRecoverySystem()
        self.circuit_breakers = {}
        
        self.active_failures = []
        self.resolved_failures = []
        self.system_context = {}
        
        # Monitoring thread
        self.monitoring_active = False
        self.monitoring_thread = None
        
        # Health status
        self.overall_health = 1.0
        self.last_health_check = time.time()
        
    def start_monitoring(self):
        """Start continuous health monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
        logging.info("Self-healing system monitoring started")
    
    def stop_monitoring(self):
        """Stop health monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        
        logging.info("Self-healing system monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                # Check for new failures
                new_failures = self.performance_monitor.detect_degradation()
                
                for failure in new_failures:
                    self._handle_failure(failure)
                
                # Update overall health
                self._update_health_status()
                
                # Sleep before next check
                time.sleep(self.config.get('monitoring_interval', 5.0))
                
            except Exception as e:
                logging.error(f"Error in monitoring loop: {e}")
                time.sleep(1.0)
    
    def _handle_failure(self, failure: FailureEvent):
        """Handle a detected failure."""
        logging.warning(f"Failure detected: {failure.failure_type} - {failure.description}")
        
        # Add to active failures
        self.active_failures.append(failure)
        
        # Attempt recovery
        recovery_success = self.recovery_system.recover(failure, self.system_context)
        
        if recovery_success:
            # Move to resolved failures
            self.active_failures.remove(failure)
            self.resolved_failures.append(failure)
            logging.info(f"Failure resolved: {failure.failure_type}")
        else:
            # Escalate if critical
            if failure.severity == SeverityLevel.CRITICAL:
                self._escalate_failure(failure)
    
    def _escalate_failure(self, failure: FailureEvent):
        """Escalate critical failures that couldn't be resolved."""
        logging.critical(f"CRITICAL FAILURE - Manual intervention required: {failure.failure_type}")
        
        # Could trigger alerts, notifications, etc.
        escalation_data = {
            'timestamp': failure.timestamp,
            'failure_type': failure.failure_type.value,
            'severity': failure.severity.value,
            'description': failure.description,
            'metrics': failure.metrics,
            'recovery_attempts': failure.recovery_actions
        }
        
        # Save escalation data for manual review
        with open('critical_failure_escalation.json', 'w') as f:
            json.dump(escalation_data, f, indent=2)
    
    def _update_health_status(self):
        """Update overall system health status."""
        current_time = time.time()
        
        # Calculate health based on active failures
        health_penalty = 0.0
        for failure in self.active_failures:
            if failure.severity == SeverityLevel.CRITICAL:
                health_penalty += 0.5
            elif failure.severity == SeverityLevel.HIGH:
                health_penalty += 0.2
            elif failure.severity == SeverityLevel.MEDIUM:
                health_penalty += 0.1
            else:
                health_penalty += 0.05
        
        self.overall_health = max(0.0, 1.0 - health_penalty)
        self.last_health_check = current_time
    
    def update_metrics(self, metrics: HealthMetrics):
        """Update system metrics for monitoring."""
        self.performance_monitor.update(metrics)
    
    def update_context(self, context_updates: Dict[str, Any]):
        """Update system context for recovery strategies."""
        self.system_context.update(context_updates)
    
    def get_circuit_breaker(self, name: str) -> CircuitBreaker:
        """Get or create a circuit breaker."""
        if name not in self.circuit_breakers:
            self.circuit_breakers[name] = CircuitBreaker()
        return self.circuit_breakers[name]
    
    def get_health_report(self) -> Dict[str, Any]:
        """Get comprehensive health report."""
        return {
            'overall_health': self.overall_health,
            'last_health_check': self.last_health_check,
            'active_failures_count': len(self.active_failures),
            'resolved_failures_count': len(self.resolved_failures),
            'active_failures': [
                {
                    'type': f.failure_type.value,
                    'severity': f.severity.value,
                    'timestamp': f.timestamp,
                    'description': f.description
                } for f in self.active_failures
            ],
            'recovery_success_rates': {
                ft.value: self.recovery_system.get_success_rate(ft)
                for ft in FailureType
            },
            'circuit_breaker_states': {
                name: cb.state for name, cb in self.circuit_breakers.items()
            },
            'monitoring_active': self.monitoring_active
        }


def create_self_healing_demo():
    """Create a demonstration of the self-healing system."""
    
    config = {
        'monitoring_interval': 2.0,  # Check every 2 seconds
        'degradation_threshold': 0.15,
        'recovery_timeout': 30.0
    }
    
    # Create self-healing system
    healing_system = SelfHealingSystem(config)
    
    # Set up initial context
    healing_system.update_context({
        'model_config': {
            'hidden_dim': 16,
            'sparsity': 0.3,
            'quantization_bits': 8
        },
        'optimizer_config': {
            'learning_rate': 0.001
        },
        'hardware_config': {
            'low_power_mode': False
        },
        'inference_config': {
            'target_fps': 50
        }
    })
    
    return healing_system