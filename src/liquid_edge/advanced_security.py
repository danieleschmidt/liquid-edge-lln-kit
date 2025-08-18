"""Advanced security and threat detection for liquid neural networks."""

import hashlib
import hmac
import time
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
import jax
import jax.numpy as jnp
import numpy as np
from functools import wraps
import logging


class ThreatLevel(Enum):
    """Security threat severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SecurityEvent(Enum):
    """Types of security events to monitor."""
    ADVERSARIAL_INPUT = "adversarial_input"
    MODEL_TAMPERING = "model_tampering"
    INFERENCE_ATTACK = "inference_attack"
    TIMING_ATTACK = "timing_attack"
    ENERGY_ANOMALY = "energy_anomaly"
    MEMORY_CORRUPTION = "memory_corruption"


@dataclass
class SecurityConfig:
    """Configuration for security monitoring."""
    enable_input_validation: bool = True
    enable_adversarial_detection: bool = True
    enable_model_integrity: bool = True
    enable_timing_protection: bool = True
    enable_energy_monitoring: bool = True
    max_inference_time_ms: float = 50.0
    max_energy_consumption_mw: float = 200.0
    adversarial_threshold: float = 0.1
    integrity_check_interval: int = 100
    log_security_events: bool = True


class SecurityMonitor:
    """Advanced security monitoring for liquid neural networks."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.threat_count = {level: 0 for level in ThreatLevel}
        self.security_events = []
        self.model_hash = None
        self.baseline_energy = None
        self.inference_count = 0
        
    def set_model_baseline(self, model_params: Dict[str, Any]) -> str:
        """Set baseline hash for model integrity checking."""
        # Serialize model parameters for hashing
        param_bytes = self._serialize_params(model_params)
        self.model_hash = hashlib.sha256(param_bytes).hexdigest()
        self.logger.info(f"Model integrity baseline set: {self.model_hash[:16]}...")
        return self.model_hash
    
    def check_model_integrity(self, model_params: Dict[str, Any]) -> bool:
        """Verify model has not been tampered with."""
        if not self.config.enable_model_integrity or self.model_hash is None:
            return True
            
        current_hash = hashlib.sha256(self._serialize_params(model_params)).hexdigest()
        
        if current_hash != self.model_hash:
            self._log_security_event(
                SecurityEvent.MODEL_TAMPERING,
                ThreatLevel.CRITICAL,
                f"Model hash mismatch: expected {self.model_hash[:16]}, got {current_hash[:16]}"
            )
            return False
        
        return True
    
    def validate_input(self, inputs: jnp.ndarray) -> Tuple[bool, Optional[str]]:
        """Validate input data for security threats."""
        if not self.config.enable_input_validation:
            return True, None
        
        # Check for NaN/Inf values
        if jnp.any(jnp.isnan(inputs)) or jnp.any(jnp.isinf(inputs)):
            self._log_security_event(
                SecurityEvent.ADVERSARIAL_INPUT,
                ThreatLevel.HIGH,
                "Invalid input: NaN or Inf values detected"
            )
            return False, "Invalid numerical values in input"
        
        # Check input range
        if jnp.any(jnp.abs(inputs) > 1000):
            self._log_security_event(
                SecurityEvent.ADVERSARIAL_INPUT,
                ThreatLevel.MEDIUM,
                f"Extreme input values detected: max={jnp.max(jnp.abs(inputs)):.2f}"
            )
            return False, "Input values exceed expected range"
        
        # Adversarial detection using input gradient analysis
        if self.config.enable_adversarial_detection:
            adversarial_score = self._detect_adversarial_input(inputs)
            if adversarial_score > self.config.adversarial_threshold:
                self._log_security_event(
                    SecurityEvent.ADVERSARIAL_INPUT,
                    ThreatLevel.HIGH,
                    f"Potential adversarial input detected: score={adversarial_score:.3f}"
                )
                return False, f"Adversarial input detected (score: {adversarial_score:.3f})"
        
        return True, None
    
    def monitor_inference_timing(self, inference_time_ms: float) -> bool:
        """Monitor inference timing for timing attacks."""
        if not self.config.enable_timing_protection:
            return True
        
        if inference_time_ms > self.config.max_inference_time_ms:
            self._log_security_event(
                SecurityEvent.TIMING_ATTACK,
                ThreatLevel.MEDIUM,
                f"Inference time exceeded threshold: {inference_time_ms:.2f}ms > {self.config.max_inference_time_ms}ms"
            )
            return False
        
        return True
    
    def monitor_energy_consumption(self, energy_mw: float) -> bool:
        """Monitor energy consumption for anomalies."""
        if not self.config.enable_energy_monitoring:
            return True
        
        if self.baseline_energy is None:
            self.baseline_energy = energy_mw
            return True
        
        # Check for excessive energy consumption
        if energy_mw > self.config.max_energy_consumption_mw:
            self._log_security_event(
                SecurityEvent.ENERGY_ANOMALY,
                ThreatLevel.HIGH,
                f"Energy consumption exceeded limit: {energy_mw:.1f}mW > {self.config.max_energy_consumption_mw}mW"
            )
            return False
        
        # Check for sudden energy spikes (DoS attack indicator)
        energy_spike_threshold = self.baseline_energy * 3.0
        if energy_mw > energy_spike_threshold:
            self._log_security_event(
                SecurityEvent.ENERGY_ANOMALY,
                ThreatLevel.MEDIUM,
                f"Energy spike detected: {energy_mw:.1f}mW (baseline: {self.baseline_energy:.1f}mW)"
            )
        
        return True
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get current security monitoring status."""
        return {
            "threat_counts": dict(self.threat_count),
            "total_events": len(self.security_events),
            "model_integrity": self.model_hash is not None,
            "inference_count": self.inference_count,
            "recent_events": self.security_events[-10:] if self.security_events else [],
            "security_score": self._calculate_security_score()
        }
    
    def _detect_adversarial_input(self, inputs: jnp.ndarray) -> float:
        """Detect potential adversarial inputs using statistical analysis."""
        # Simple adversarial detection based on input statistics
        input_std = jnp.std(inputs)
        input_mean = jnp.mean(jnp.abs(inputs))
        
        # High variance or extreme values may indicate adversarial crafting
        adversarial_score = float(input_std + input_mean / 10.0)
        
        return min(adversarial_score, 1.0)
    
    def _serialize_params(self, params: Dict[str, Any]) -> bytes:
        """Serialize model parameters for integrity checking."""
        # Flatten all parameters and convert to bytes
        flat_params = []
        
        def flatten_dict(d, prefix=""):
            for key, value in d.items():
                if isinstance(value, dict):
                    flatten_dict(value, f"{prefix}{key}.")
                else:
                    if hasattr(value, 'tobytes'):
                        flat_params.append(value.tobytes())
                    else:
                        flat_params.append(str(value).encode())
        
        flatten_dict(params)
        return b''.join(flat_params)
    
    def _log_security_event(self, event: SecurityEvent, level: ThreatLevel, message: str):
        """Log security event with timestamp."""
        event_data = {
            "timestamp": time.time(),
            "event": event.value,
            "level": level.value,
            "message": message,
            "inference_count": self.inference_count
        }
        
        self.security_events.append(event_data)
        self.threat_count[level] += 1
        
        if self.config.log_security_events:
            log_level = getattr(logging, level.value.upper(), logging.INFO)
            self.logger.log(log_level, f"Security event: {event.value} - {message}")
    
    def _calculate_security_score(self) -> float:
        """Calculate overall security score (0-100)."""
        if not self.security_events:
            return 100.0
        
        # Weight threats by severity
        threat_weights = {
            ThreatLevel.LOW: 1,
            ThreatLevel.MEDIUM: 3,
            ThreatLevel.HIGH: 7,
            ThreatLevel.CRITICAL: 15
        }
        
        total_threat_score = sum(
            self.threat_count[level] * weight
            for level, weight in threat_weights.items()
        )
        
        # Calculate score based on total inferences and threat score
        max_score = max(self.inference_count * 0.1, 1.0)  # Prevent division by zero
        security_score = max(0.0, 100.0 - (total_threat_score / max_score) * 10)
        
        return min(security_score, 100.0)


def secure_inference(security_monitor: SecurityMonitor):
    """Decorator for secure model inference."""
    def decorator(inference_func):
        @wraps(inference_func)
        def wrapper(model_params, inputs, *args, **kwargs):
            security_monitor.inference_count += 1
            
            # Pre-inference security checks
            if security_monitor.inference_count % security_monitor.config.integrity_check_interval == 0:
                if not security_monitor.check_model_integrity(model_params):
                    raise SecurityError("Model integrity check failed")
            
            # Input validation
            valid, error_msg = security_monitor.validate_input(inputs)
            if not valid:
                raise SecurityError(f"Input validation failed: {error_msg}")
            
            # Timing protection
            start_time = time.time()
            
            try:
                # Execute inference
                result = inference_func(model_params, inputs, *args, **kwargs)
                
                # Post-inference monitoring
                inference_time_ms = (time.time() - start_time) * 1000
                if not security_monitor.monitor_inference_timing(inference_time_ms):
                    security_monitor.logger.warning(f"Timing anomaly detected: {inference_time_ms:.2f}ms")
                
                return result
                
            except Exception as e:
                security_monitor._log_security_event(
                    SecurityEvent.INFERENCE_ATTACK,
                    ThreatLevel.HIGH,
                    f"Inference failed with error: {str(e)}"
                )
                raise
        
        return wrapper
    return decorator


class SecurityError(Exception):
    """Exception raised for security-related issues."""
    pass


class SecureLiquidInference:
    """Secure wrapper for liquid neural network inference."""
    
    def __init__(self, model, config: SecurityConfig):
        self.model = model
        self.security_monitor = SecurityMonitor(config)
        self.model_params = None
    
    def set_model_params(self, params: Dict[str, Any]):
        """Set model parameters and establish security baseline."""
        self.model_params = params
        self.security_monitor.set_model_baseline(params)
    
    def __call__(self, model_params, inputs, *args, **kwargs):
        """Secure inference execution."""
        # Manual security checks instead of decorator
        self.security_monitor.inference_count += 1
        
        # Pre-inference security checks
        if self.security_monitor.inference_count % self.security_monitor.config.integrity_check_interval == 0:
            if not self.security_monitor.check_model_integrity(model_params):
                raise SecurityError("Model integrity check failed")
        
        # Input validation
        valid, error_msg = self.security_monitor.validate_input(inputs)
        if not valid:
            raise SecurityError(f"Input validation failed: {error_msg}")
        
        # Timing protection
        start_time = time.time()
        
        try:
            # Execute inference
            result = self.model.apply(model_params, inputs, *args, **kwargs)
            
            # Post-inference monitoring
            inference_time_ms = (time.time() - start_time) * 1000
            if not self.security_monitor.monitor_inference_timing(inference_time_ms):
                self.security_monitor.logger.warning(f"Timing anomaly detected: {inference_time_ms:.2f}ms")
            
            return result
            
        except Exception as e:
            self.security_monitor._log_security_event(
                SecurityEvent.INFERENCE_ATTACK,
                ThreatLevel.HIGH,
                f"Inference failed with error: {str(e)}"
            )
            raise
    
    def get_security_report(self) -> Dict[str, Any]:
        """Generate comprehensive security report."""
        status = self.security_monitor.get_security_status()
        
        return {
            "security_status": status,
            "recommendations": self._generate_security_recommendations(status),
            "config": {
                "input_validation": self.security_monitor.config.enable_input_validation,
                "adversarial_detection": self.security_monitor.config.enable_adversarial_detection,
                "model_integrity": self.security_monitor.config.enable_model_integrity,
                "timing_protection": self.security_monitor.config.enable_timing_protection,
                "energy_monitoring": self.security_monitor.config.enable_energy_monitoring
            }
        }
    
    def _generate_security_recommendations(self, status: Dict[str, Any]) -> List[str]:
        """Generate security recommendations based on current status."""
        recommendations = []
        
        if status["security_score"] < 80:
            recommendations.append("Security score is below 80%. Increase monitoring sensitivity.")
        
        if status["threat_counts"].get(ThreatLevel.CRITICAL.value, 0) > 0:
            recommendations.append("Critical threats detected. Review model deployment immediately.")
        
        if status["threat_counts"].get(ThreatLevel.HIGH.value, 0) > 5:
            recommendations.append("Multiple high-severity threats. Consider implementing additional security layers.")
        
        if not status["model_integrity"]:
            recommendations.append("Model integrity checking is disabled. Enable for production deployment.")
        
        if len(status["recent_events"]) > 5:
            recommendations.append("High frequency of security events. Review input sources and system configuration.")
        
        if not recommendations:
            recommendations.append("Security status is good. Continue monitoring.")
        
        return recommendations