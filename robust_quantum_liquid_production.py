#!/usr/bin/env python3
"""
Robust Quantum-Liquid Neural Network Production System
Generation 2: MAKE IT ROBUST (Reliable Implementation)

This system adds comprehensive error handling, monitoring, logging,
security measures, and production-grade robustness to the quantum-liquid system.
"""

import time
import json
import math
import random
import hashlib
import threading
import queue
import signal
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional, Union
import logging
from dataclasses import dataclass, field
from enum import Enum
from contextlib import contextmanager
import warnings

# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('quantum_liquid_robust.log')
    ]
)
logger = logging.getLogger(__name__)

# Security and monitoring imports
try:
    import numpy as np
    NUMPY_AVAILABLE = True
    logger.info("NumPy available for enhanced performance")
except ImportError:
    NUMPY_AVAILABLE = False
    logger.warning("NumPy not available, using pure Python implementation")

class ErrorSeverity(Enum):
    """Error severity levels for robust error handling."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class SystemHealth(Enum):
    """System health status enumeration."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    FAILED = "failed"

class SecurityLevel(Enum):
    """Security level enumeration."""
    BASIC = "basic"
    ENHANCED = "enhanced"
    QUANTUM_SECURE = "quantum_secure"

@dataclass
class RobustQuantumLiquidConfig:
    """Enhanced configuration with robustness parameters."""
    
    # Core quantum-liquid parameters
    input_dim: int = 8
    quantum_dim: int = 16
    liquid_hidden_dim: int = 32
    output_dim: int = 4
    
    # Quantum parameters
    quantum_coherence_time: float = 100.0
    quantum_entanglement_strength: float = 0.7
    quantum_gate_fidelity: float = 0.99
    quantum_noise_level: float = 0.01
    
    # Liquid dynamics
    tau_min: float = 1.0
    tau_max: float = 50.0
    liquid_sparsity: float = 0.4
    
    # Robustness parameters
    max_inference_time_ms: float = 10.0
    max_memory_usage_mb: float = 100.0
    error_recovery_attempts: int = 3
    health_check_interval_s: float = 1.0
    
    # Security parameters
    security_level: SecurityLevel = SecurityLevel.ENHANCED
    enable_input_validation: bool = True
    enable_output_sanitization: bool = True
    enable_audit_logging: bool = True
    
    # Monitoring parameters
    enable_performance_monitoring: bool = True
    enable_resource_monitoring: bool = True
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'inference_time_ms': 5.0,
        'memory_usage_mb': 80.0,
        'error_rate': 0.05,
        'quantum_coherence_min': 0.6
    })
    
    # Production parameters
    enable_circuit_breaker: bool = True
    circuit_breaker_threshold: int = 5
    enable_graceful_degradation: bool = True
    enable_auto_recovery: bool = True
    backup_model_path: Optional[str] = None

class QuantumLiquidError(Exception):
    """Base exception for quantum-liquid system errors."""
    
    def __init__(self, message: str, severity: ErrorSeverity = ErrorSeverity.MEDIUM, 
                 error_code: str = "UNKNOWN", context: Dict[str, Any] = None):
        super().__init__(message)
        self.severity = severity
        self.error_code = error_code
        self.context = context or {}
        self.timestamp = datetime.now()

class InferenceTimeoutError(QuantumLiquidError):
    """Raised when inference takes too long."""
    
    def __init__(self, timeout_ms: float, actual_ms: float):
        super().__init__(
            f"Inference timeout: {actual_ms:.2f}ms > {timeout_ms:.2f}ms limit",
            ErrorSeverity.HIGH,
            "INFERENCE_TIMEOUT",
            {"timeout_ms": timeout_ms, "actual_ms": actual_ms}
        )

class QuantumCoherenceError(QuantumLiquidError):
    """Raised when quantum coherence drops below threshold."""
    
    def __init__(self, coherence: float, threshold: float):
        super().__init__(
            f"Quantum coherence too low: {coherence:.3f} < {threshold:.3f}",
            ErrorSeverity.MEDIUM,
            "QUANTUM_COHERENCE_LOW",
            {"coherence": coherence, "threshold": threshold}
        )

class SecurityViolationError(QuantumLiquidError):
    """Raised when security validation fails."""
    
    def __init__(self, violation_type: str, details: str):
        super().__init__(
            f"Security violation - {violation_type}: {details}",
            ErrorSeverity.CRITICAL,
            "SECURITY_VIOLATION",
            {"violation_type": violation_type, "details": details}
        )

class CircuitBreaker:
    """Circuit breaker pattern for fault tolerance."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self._lock = threading.Lock()
    
    def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        with self._lock:
            if self.state == "OPEN":
                if self._should_attempt_reset():
                    self.state = "HALF_OPEN"
                else:
                    raise QuantumLiquidError(
                        "Circuit breaker is OPEN", 
                        ErrorSeverity.HIGH,
                        "CIRCUIT_BREAKER_OPEN"
                    )
            
            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
            except Exception as e:
                self._on_failure()
                raise
    
    def _should_attempt_reset(self):
        """Check if enough time has passed to attempt reset."""
        if self.last_failure_time is None:
            return True
        return time.time() - self.last_failure_time > self.recovery_timeout
    
    def _on_success(self):
        """Handle successful execution."""
        if self.state == "HALF_OPEN":
            self.state = "CLOSED"
        self.failure_count = 0
    
    def _on_failure(self):
        """Handle failed execution."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"

class SecurityValidator:
    """Security validation for quantum-liquid inputs and outputs."""
    
    def __init__(self, security_level: SecurityLevel = SecurityLevel.ENHANCED):
        self.security_level = security_level
        self.max_input_magnitude = 10.0
        self.max_output_magnitude = 5.0
        self.suspicious_patterns = [
            lambda x: any(abs(xi) > self.max_input_magnitude for xi in x),
            lambda x: any(math.isnan(xi) or math.isinf(xi) for xi in x),
            lambda x: len(x) > 1000,  # Prevent DoS attacks
        ]
    
    def validate_input(self, input_data: List[float]) -> bool:
        """Validate input data for security issues."""
        try:
            # Basic type checking
            if not isinstance(input_data, (list, tuple)):
                raise SecurityViolationError("INVALID_TYPE", "Input must be list or tuple")
            
            # Check for suspicious patterns
            for i, pattern in enumerate(self.suspicious_patterns):
                if pattern(input_data):
                    raise SecurityViolationError(
                        "SUSPICIOUS_PATTERN", 
                        f"Pattern {i} detected in input"
                    )
            
            # Advanced security checks for enhanced mode
            if self.security_level in [SecurityLevel.ENHANCED, SecurityLevel.QUANTUM_SECURE]:
                self._advanced_input_validation(input_data)
            
            return True
            
        except SecurityViolationError:
            raise
        except Exception as e:
            raise SecurityViolationError("VALIDATION_ERROR", str(e))
    
    def _advanced_input_validation(self, input_data: List[float]):
        """Advanced security validation."""
        # Check for adversarial patterns
        variance = sum((x - sum(input_data)/len(input_data))**2 for x in input_data) / len(input_data)
        if variance > 100.0:
            raise SecurityViolationError("HIGH_VARIANCE", f"Input variance too high: {variance}")
        
        # Check for potential injection patterns
        if any(str(x).count('.') > 1 for x in input_data):
            raise SecurityViolationError("POTENTIAL_INJECTION", "Multiple decimal points detected")
    
    def sanitize_output(self, output_data: List[float]) -> List[float]:
        """Sanitize output data."""
        sanitized = []
        for value in output_data:
            if math.isnan(value) or math.isinf(value):
                sanitized.append(0.0)
            elif abs(value) > self.max_output_magnitude:
                sanitized.append(math.copysign(self.max_output_magnitude, value))
            else:
                sanitized.append(value)
        return sanitized

class PerformanceMonitor:
    """Real-time performance monitoring system."""
    
    def __init__(self, config: RobustQuantumLiquidConfig):
        self.config = config
        self.metrics = {
            'inference_times': [],
            'memory_usage': [],
            'error_counts': {},
            'quantum_coherence': [],
            'system_health': SystemHealth.HEALTHY
        }
        self.alerts = queue.Queue()
        self._start_time = time.time()
        
    def record_inference_time(self, time_ms: float):
        """Record inference time and check thresholds."""
        self.metrics['inference_times'].append(time_ms)
        
        # Keep only recent measurements
        if len(self.metrics['inference_times']) > 1000:
            self.metrics['inference_times'] = self.metrics['inference_times'][-500:]
        
        # Check alert threshold
        if time_ms > self.config.alert_thresholds['inference_time_ms']:
            self._trigger_alert(
                "INFERENCE_TIME_HIGH",
                f"Inference time {time_ms:.2f}ms exceeds threshold {self.config.alert_thresholds['inference_time_ms']}ms"
            )
    
    def record_error(self, error: Exception):
        """Record error occurrence."""
        error_type = type(error).__name__
        self.metrics['error_counts'][error_type] = self.metrics['error_counts'].get(error_type, 0) + 1
        
        # Calculate error rate
        total_inferences = len(self.metrics['inference_times'])
        total_errors = sum(self.metrics['error_counts'].values())
        error_rate = total_errors / max(total_inferences, 1)
        
        if error_rate > self.config.alert_thresholds['error_rate']:
            self._trigger_alert(
                "ERROR_RATE_HIGH",
                f"Error rate {error_rate:.3f} exceeds threshold {self.config.alert_thresholds['error_rate']}"
            )
    
    def record_quantum_coherence(self, coherence: float):
        """Record quantum coherence measurement."""
        self.metrics['quantum_coherence'].append(coherence)
        
        if coherence < self.config.alert_thresholds['quantum_coherence_min']:
            self._trigger_alert(
                "QUANTUM_COHERENCE_LOW",
                f"Quantum coherence {coherence:.3f} below threshold {self.config.alert_thresholds['quantum_coherence_min']}"
            )
    
    def _trigger_alert(self, alert_type: str, message: str):
        """Trigger system alert."""
        alert = {
            'type': alert_type,
            'message': message,
            'timestamp': datetime.now().isoformat(),
            'severity': 'WARNING'
        }
        self.alerts.put(alert)
        logger.warning(f"ALERT: {alert_type} - {message}")
    
    def get_health_status(self) -> SystemHealth:
        """Calculate overall system health."""
        if not self.metrics['inference_times']:
            return SystemHealth.HEALTHY
        
        # Check recent error rate
        recent_errors = sum(self.metrics['error_counts'].values())
        recent_inferences = len(self.metrics['inference_times'][-100:])
        error_rate = recent_errors / max(recent_inferences, 1)
        
        # Check recent performance
        recent_times = self.metrics['inference_times'][-10:] if self.metrics['inference_times'] else []
        avg_time = sum(recent_times) / len(recent_times) if recent_times else 0
        
        # Determine health status
        if error_rate > 0.2 or avg_time > self.config.alert_thresholds['inference_time_ms'] * 2:
            return SystemHealth.FAILED
        elif error_rate > 0.1 or avg_time > self.config.alert_thresholds['inference_time_ms']:
            return SystemHealth.CRITICAL
        elif error_rate > 0.05 or avg_time > self.config.alert_thresholds['inference_time_ms'] * 0.8:
            return SystemHealth.DEGRADED
        else:
            return SystemHealth.HEALTHY

@contextmanager
def timeout_context(timeout_seconds: float):
    """Context manager for operation timeout."""
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {timeout_seconds} seconds")
    
    # Set up the timeout
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(int(timeout_seconds))
    
    try:
        yield
    finally:
        # Clean up
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

class RobustQuantumLiquidSystem:
    """Production-grade robust quantum-liquid neural network system."""
    
    def __init__(self, config: RobustQuantumLiquidConfig):
        self.config = config
        self.security_validator = SecurityValidator(config.security_level)
        self.performance_monitor = PerformanceMonitor(config)
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=config.circuit_breaker_threshold,
            recovery_timeout=30.0
        ) if config.enable_circuit_breaker else None
        
        # Initialize the core quantum-liquid system
        self._initialize_core_system()
        
        # Health monitoring
        self._health_check_thread = None
        self._shutdown_event = threading.Event()
        
        if config.enable_performance_monitoring:
            self._start_health_monitoring()
        
        logger.info("RobustQuantumLiquidSystem initialized with enhanced security and monitoring")
    
    def _initialize_core_system(self):
        """Initialize the core quantum-liquid neural network."""
        # Import the pure Python implementation from Generation 1
        from pure_python_quantum_breakthrough import (
            PurePythonQuantumLiquidConfig,
            PurePythonQuantumLiquidNetwork
        )
        
        # Create core config
        core_config = PurePythonQuantumLiquidConfig()
        core_config.input_dim = self.config.input_dim
        core_config.quantum_dim = self.config.quantum_dim
        core_config.liquid_hidden_dim = self.config.liquid_hidden_dim
        core_config.output_dim = self.config.output_dim
        
        # Initialize core model
        self.core_model = PurePythonQuantumLiquidNetwork(core_config)
        
        # Initialize states
        self.quantum_state = [0.0] * core_config.quantum_dim
        self.liquid_state = [0.0] * core_config.liquid_hidden_dim
        
        logger.info("Core quantum-liquid system initialized")
    
    def _start_health_monitoring(self):
        """Start background health monitoring."""
        def health_monitor():
            while not self._shutdown_event.is_set():
                try:
                    health = self.performance_monitor.get_health_status()
                    self.performance_monitor.metrics['system_health'] = health
                    
                    if health in [SystemHealth.CRITICAL, SystemHealth.FAILED]:
                        logger.error(f"System health: {health.value}")
                        if self.config.enable_auto_recovery:
                            self._attempt_auto_recovery()
                    
                    time.sleep(self.config.health_check_interval_s)
                    
                except Exception as e:
                    logger.error(f"Health monitoring error: {e}")
                    time.sleep(self.config.health_check_interval_s)
        
        self._health_check_thread = threading.Thread(target=health_monitor, daemon=True)
        self._health_check_thread.start()
        logger.info("Health monitoring started")
    
    def _attempt_auto_recovery(self):
        """Attempt automatic system recovery."""
        logger.info("Attempting auto-recovery...")
        
        try:
            # Reset states
            self.quantum_state = [0.0] * self.config.quantum_dim
            self.liquid_state = [0.0] * self.config.liquid_hidden_dim
            
            # Reset circuit breaker
            if self.circuit_breaker:
                self.circuit_breaker.failure_count = 0
                self.circuit_breaker.state = "CLOSED"
            
            # Clear old metrics
            self.performance_monitor.metrics['inference_times'] = []
            
            logger.info("Auto-recovery completed")
            
        except Exception as e:
            logger.error(f"Auto-recovery failed: {e}")
    
    def robust_inference(self, input_data: List[float]) -> Tuple[List[float], Dict[str, Any]]:
        """Perform robust inference with comprehensive error handling."""
        inference_start = time.time()
        inference_metadata = {
            'timestamp': datetime.now().isoformat(),
            'input_hash': hashlib.md5(str(input_data).encode()).hexdigest()[:8],
            'security_level': self.config.security_level.value
        }
        
        try:
            # Security validation
            if self.config.enable_input_validation:
                self.security_validator.validate_input(input_data)
                inference_metadata['input_validated'] = True
            
            # Circuit breaker protection
            if self.circuit_breaker:
                output = self.circuit_breaker.call(self._core_inference, input_data)
            else:
                output = self._core_inference(input_data)
            
            # Output sanitization
            if self.config.enable_output_sanitization:
                output = self.security_validator.sanitize_output(output)
                inference_metadata['output_sanitized'] = True
            
            # Record successful inference
            inference_time_ms = (time.time() - inference_start) * 1000
            self.performance_monitor.record_inference_time(inference_time_ms)
            
            inference_metadata.update({
                'inference_time_ms': inference_time_ms,
                'success': True,
                'system_health': self.performance_monitor.get_health_status().value
            })
            
            # Audit logging
            if self.config.enable_audit_logging:
                self._audit_log("INFERENCE_SUCCESS", inference_metadata)
            
            return output, inference_metadata
            
        except Exception as e:
            # Record error
            self.performance_monitor.record_error(e)
            
            # Attempt graceful degradation
            if self.config.enable_graceful_degradation:
                output = self._graceful_degradation_response(input_data)
                inference_metadata.update({
                    'degraded_response': True,
                    'original_error': str(e),
                    'success': False
                })
                return output, inference_metadata
            else:
                inference_metadata.update({
                    'error': str(e),
                    'error_type': type(e).__name__,
                    'success': False
                })
                
                if self.config.enable_audit_logging:
                    self._audit_log("INFERENCE_ERROR", inference_metadata)
                
                raise
    
    def _core_inference(self, input_data: List[float]) -> List[float]:
        """Core inference with timeout protection."""
        timeout_s = self.config.max_inference_time_ms / 1000.0
        
        try:
            with timeout_context(timeout_s):
                output, (new_quantum_state, new_liquid_state) = self.core_model.forward(
                    input_data, self.quantum_state, self.liquid_state
                )
                
                # Update states
                self.quantum_state = new_quantum_state
                self.liquid_state = new_liquid_state
                
                # Record quantum coherence
                coherence = sum(abs(x) for x in self.quantum_state) / len(self.quantum_state)
                self.performance_monitor.record_quantum_coherence(coherence)
                
                # Check coherence threshold
                if coherence < self.config.alert_thresholds['quantum_coherence_min']:
                    raise QuantumCoherenceError(coherence, self.config.alert_thresholds['quantum_coherence_min'])
                
                return output
                
        except TimeoutError as e:
            actual_time_ms = self.config.max_inference_time_ms
            raise InferenceTimeoutError(self.config.max_inference_time_ms, actual_time_ms)
    
    def _graceful_degradation_response(self, input_data: List[float]) -> List[float]:
        """Provide fallback response during system degradation."""
        logger.warning("Providing graceful degradation response")
        
        # Simple fallback: return scaled input or zeros
        if len(input_data) >= self.config.output_dim:
            return input_data[:self.config.output_dim]
        else:
            return [0.0] * self.config.output_dim
    
    def _audit_log(self, event_type: str, metadata: Dict[str, Any]):
        """Write audit log entry."""
        audit_entry = {
            'event_type': event_type,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata
        }
        
        # Write to audit log file
        audit_file = Path("quantum_liquid_audit.log")
        with open(audit_file, "a") as f:
            f.write(json.dumps(audit_entry) + "\n")
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system metrics."""
        metrics = {
            'health_status': self.performance_monitor.get_health_status().value,
            'total_inferences': len(self.performance_monitor.metrics['inference_times']),
            'error_counts': self.performance_monitor.metrics['error_counts'].copy(),
            'circuit_breaker_state': self.circuit_breaker.state if self.circuit_breaker else "DISABLED",
            'uptime_seconds': time.time() - self.performance_monitor._start_time
        }
        
        # Calculate statistics
        if self.performance_monitor.metrics['inference_times']:
            times = self.performance_monitor.metrics['inference_times']
            metrics['inference_stats'] = {
                'avg_time_ms': sum(times) / len(times),
                'min_time_ms': min(times),
                'max_time_ms': max(times),
                'recent_avg_ms': sum(times[-10:]) / min(len(times), 10)
            }
        
        if self.performance_monitor.metrics['quantum_coherence']:
            coherence = self.performance_monitor.metrics['quantum_coherence']
            metrics['quantum_stats'] = {
                'avg_coherence': sum(coherence) / len(coherence),
                'min_coherence': min(coherence),
                'max_coherence': max(coherence),
                'recent_coherence': coherence[-1] if coherence else 0.0
            }
        
        return metrics
    
    def shutdown(self):
        """Gracefully shutdown the system."""
        logger.info("Shutting down RobustQuantumLiquidSystem...")
        
        # Signal shutdown
        self._shutdown_event.set()
        
        # Wait for health monitoring thread
        if self._health_check_thread and self._health_check_thread.is_alive():
            self._health_check_thread.join(timeout=5.0)
        
        logger.info("Shutdown complete")

def run_generation2_robust_demo():
    """Run Generation 2 robust quantum-liquid demonstration."""
    logger.info("🛡️ Starting Generation 2: MAKE IT ROBUST (Reliable) Demo")
    
    # Configure robust system
    config = RobustQuantumLiquidConfig(
        input_dim=8,
        quantum_dim=16,
        liquid_hidden_dim=32,
        output_dim=4,
        max_inference_time_ms=5.0,
        security_level=SecurityLevel.ENHANCED,
        enable_circuit_breaker=True,
        enable_graceful_degradation=True,
        enable_auto_recovery=True
    )
    
    # Create robust system
    system = RobustQuantumLiquidSystem(config)
    
    try:
        # Run robustness tests
        results = _run_robustness_tests(system)
        
        # Save results
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        with open(results_dir / "generation2_robust_quantum_liquid.json", "w") as f:
            json.dump(results, f, indent=2)
        
        logger.info("✅ Generation 2 robust quantum-liquid system completed!")
        logger.info(f"   Robustness Score: {results['robustness_score']:.2f}")
        logger.info(f"   Security Level: {results['security_level']}")
        logger.info(f"   Error Recovery Rate: {results['error_recovery_rate']:.1%}")
        
        return results
        
    finally:
        system.shutdown()

def _run_robustness_tests(system: RobustQuantumLiquidSystem) -> Dict[str, Any]:
    """Run comprehensive robustness tests."""
    logger.info("Running robustness tests...")
    
    test_results = {
        'successful_inferences': 0,
        'failed_inferences': 0,
        'security_violations': 0,
        'timeouts': 0,
        'auto_recoveries': 0,
        'graceful_degradations': 0
    }
    
    # Test normal operation
    for i in range(50):
        try:
            input_data = [random.uniform(-1, 1) for _ in range(8)]
            output, metadata = system.robust_inference(input_data)
            
            if metadata['success']:
                test_results['successful_inferences'] += 1
            else:
                test_results['failed_inferences'] += 1
                if metadata.get('degraded_response'):
                    test_results['graceful_degradations'] += 1
                    
        except Exception as e:
            test_results['failed_inferences'] += 1
    
    # Test with malicious inputs
    malicious_inputs = [
        [float('inf')] * 8,  # Infinity values
        [float('nan')] * 8,  # NaN values
        [1000.0] * 8,        # Extremely large values
        [],                  # Empty input
        [0] * 1000,          # Oversized input
    ]
    
    for malicious_input in malicious_inputs:
        try:
            output, metadata = system.robust_inference(malicious_input)
            if metadata.get('degraded_response'):
                test_results['graceful_degradations'] += 1
        except SecurityViolationError:
            test_results['security_violations'] += 1
        except Exception:
            test_results['failed_inferences'] += 1
    
    # Calculate metrics
    total_tests = test_results['successful_inferences'] + test_results['failed_inferences']
    error_recovery_rate = test_results['graceful_degradations'] / max(test_results['failed_inferences'], 1)
    robustness_score = (
        test_results['successful_inferences'] / max(total_tests, 1) * 50 +
        test_results['security_violations'] / len(malicious_inputs) * 30 +
        error_recovery_rate * 20
    )
    
    # Get final system metrics
    system_metrics = system.get_system_metrics()
    
    results = {
        'test_results': test_results,
        'robustness_score': robustness_score,
        'error_recovery_rate': error_recovery_rate,
        'security_level': system.config.security_level.value,
        'system_metrics': system_metrics,
        'timestamp': datetime.now().isoformat(),
        'generation': 2,
        'system_type': 'robust_quantum_liquid'
    }
    
    return results

if __name__ == "__main__":
    results = run_generation2_robust_demo()
    print(f"🛡️ Robust Quantum-Liquid System achieved robustness score: {results['robustness_score']:.2f}")