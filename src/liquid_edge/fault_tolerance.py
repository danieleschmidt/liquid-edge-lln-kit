"""Fault tolerance and resilience mechanisms for liquid neural networks."""

import time
import threading
from typing import Dict, Any, Optional, List, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import jax
import jax.numpy as jnp
import numpy as np
from functools import wraps
import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import queue


class FaultType(Enum):
    """Types of faults that can occur in edge systems."""
    SENSOR_FAILURE = "sensor_failure"
    COMMUNICATION_ERROR = "communication_error"
    MEMORY_ERROR = "memory_error"
    COMPUTATION_ERROR = "computation_error"
    TIMING_VIOLATION = "timing_violation"
    ENERGY_DEPLETION = "energy_depletion"
    HARDWARE_FAULT = "hardware_fault"


class RecoveryStrategy(Enum):
    """Recovery strategies for different fault types."""
    RETRY = "retry"
    FALLBACK = "fallback"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    SAFE_SHUTDOWN = "safe_shutdown"
    REDUNDANCY_SWITCH = "redundancy_switch"


@dataclass
class FaultToleranceConfig:
    """Configuration for fault tolerance mechanisms."""
    max_retries: int = 3
    retry_delay_ms: float = 100.0
    timeout_ms: float = 1000.0
    enable_redundancy: bool = True
    enable_graceful_degradation: bool = True
    enable_checkpointing: bool = True
    checkpoint_interval: int = 100
    memory_limit_mb: float = 64.0
    energy_reserve_percentage: float = 20.0
    enable_watchdog: bool = True
    watchdog_timeout_ms: float = 5000.0


class SystemState(Enum):
    """System operational states."""
    NORMAL = "normal"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    RECOVERY = "recovery"
    SHUTDOWN = "shutdown"


@dataclass
class FaultEvent:
    """Record of a fault occurrence."""
    timestamp: float
    fault_type: FaultType
    severity: str
    message: str
    recovery_action: Optional[RecoveryStrategy] = None
    recovery_successful: bool = False
    recovery_time_ms: float = 0.0


class FaultTolerantSystem:
    """Fault-tolerant wrapper for liquid neural network systems."""
    
    def __init__(self, config: FaultToleranceConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.system_state = SystemState.NORMAL
        self.fault_history: List[FaultEvent] = []
        self.checkpoints: Dict[str, Any] = {}
        self.redundant_models: List[Any] = []
        self.primary_model_index = 0
        self.inference_count = 0
        self.last_checkpoint = 0
        self.energy_level = 100.0  # Percentage
        self.watchdog_timer = None
        self.executor = ThreadPoolExecutor(max_workers=2)
        
    def register_model(self, model: Any, is_primary: bool = True) -> int:
        """Register a model for redundancy."""
        if is_primary:
            self.redundant_models.insert(0, model)
            self.primary_model_index = 0
        else:
            self.redundant_models.append(model)
        
        model_index = len(self.redundant_models) - 1
        self.logger.info(f"Model registered with index {model_index} (primary: {is_primary})")
        return model_index
    
    def create_checkpoint(self, name: str, data: Dict[str, Any]):
        """Create a system checkpoint for recovery."""
        if not self.config.enable_checkpointing:
            return
        
        checkpoint = {
            "timestamp": time.time(),
            "system_state": self.system_state.value,
            "inference_count": self.inference_count,
            "energy_level": self.energy_level,
            "data": data.copy()
        }
        
        self.checkpoints[name] = checkpoint
        self.last_checkpoint = self.inference_count
        self.logger.debug(f"Checkpoint '{name}' created")
    
    def restore_checkpoint(self, name: str) -> Optional[Dict[str, Any]]:
        """Restore system from a checkpoint."""
        if name not in self.checkpoints:
            self.logger.error(f"Checkpoint '{name}' not found")
            return None
        
        checkpoint = self.checkpoints[name]
        self.system_state = SystemState(checkpoint["system_state"])
        self.inference_count = checkpoint["inference_count"]
        self.energy_level = checkpoint["energy_level"]
        
        self.logger.info(f"System restored from checkpoint '{name}'")
        return checkpoint["data"]
    
    def fault_tolerant_inference(self, 
                                inference_func: Callable,
                                params: Dict[str, Any],
                                inputs: jnp.ndarray,
                                **kwargs) -> jnp.ndarray:
        """Execute inference with fault tolerance."""
        self.inference_count += 1
        
        # Automatic checkpointing
        if (self.config.enable_checkpointing and 
            self.inference_count - self.last_checkpoint >= self.config.checkpoint_interval):
            self.create_checkpoint(f"auto_{self.inference_count}", {
                "params": params,
                "last_inputs": inputs
            })
        
        # Energy monitoring
        if self.energy_level < self.config.energy_reserve_percentage:
            return self._handle_fault(
                FaultType.ENERGY_DEPLETION,
                "Energy level below reserve threshold",
                inference_func, params, inputs, **kwargs
            )
        
        # Start watchdog if enabled
        if self.config.enable_watchdog:
            self._start_watchdog()
        
        try:
            # Primary inference attempt
            result = self._execute_with_timeout(
                inference_func, params, inputs, **kwargs
            )
            
            self._stop_watchdog()
            self.energy_level = max(0.0, self.energy_level - 0.1)  # Simulate energy consumption
            return result
            
        except Exception as e:
            self._stop_watchdog()
            fault_type = self._classify_fault(e)
            return self._handle_fault(fault_type, str(e), inference_func, params, inputs, **kwargs)
    
    def _execute_with_timeout(self, 
                             inference_func: Callable,
                             params: Dict[str, Any],
                             inputs: jnp.ndarray,
                             **kwargs) -> jnp.ndarray:
        """Execute inference with timeout protection."""
        future = self.executor.submit(inference_func, params, inputs, **kwargs)
        
        try:
            result = future.result(timeout=self.config.timeout_ms / 1000.0)
            return result
        except TimeoutError:
            future.cancel()
            raise RuntimeError(f"Inference timeout after {self.config.timeout_ms}ms")
    
    def _handle_fault(self, 
                     fault_type: FaultType,
                     error_message: str,
                     inference_func: Callable,
                     params: Dict[str, Any],
                     inputs: jnp.ndarray,
                     **kwargs) -> jnp.ndarray:
        """Handle different types of faults with appropriate recovery strategies."""
        start_time = time.time()
        
        # Log fault occurrence
        fault_event = FaultEvent(
            timestamp=start_time,
            fault_type=fault_type,
            severity="high",
            message=error_message
        )
        
        self.fault_history.append(fault_event)
        self.logger.warning(f"Fault detected: {fault_type.value} - {error_message}")
        
        # Determine recovery strategy
        recovery_strategy = self._get_recovery_strategy(fault_type)
        fault_event.recovery_action = recovery_strategy
        
        try:
            if recovery_strategy == RecoveryStrategy.RETRY:
                result = self._retry_inference(inference_func, params, inputs, **kwargs)
            elif recovery_strategy == RecoveryStrategy.FALLBACK:
                result = self._fallback_inference(inference_func, params, inputs, **kwargs)
            elif recovery_strategy == RecoveryStrategy.GRACEFUL_DEGRADATION:
                result = self._degraded_inference(inputs)
            elif recovery_strategy == RecoveryStrategy.REDUNDANCY_SWITCH:
                result = self._switch_to_redundant_model(inference_func, params, inputs, **kwargs)
            elif recovery_strategy == RecoveryStrategy.SAFE_SHUTDOWN:
                result = self._safe_shutdown(inputs)
            else:
                raise RuntimeError(f"Unknown recovery strategy: {recovery_strategy}")
            
            # Record successful recovery
            fault_event.recovery_successful = True
            fault_event.recovery_time_ms = (time.time() - start_time) * 1000
            
            self.logger.info(f"Recovery successful using {recovery_strategy.value}")
            return result
            
        except Exception as recovery_error:
            fault_event.recovery_successful = False
            fault_event.recovery_time_ms = (time.time() - start_time) * 1000
            
            self.logger.error(f"Recovery failed: {recovery_error}")
            
            # Last resort: return safe default output
            return self._emergency_output(inputs)
    
    def _retry_inference(self, 
                        inference_func: Callable,
                        params: Dict[str, Any],
                        inputs: jnp.ndarray,
                        **kwargs) -> jnp.ndarray:
        """Retry inference with exponential backoff."""
        last_exception = None
        
        for attempt in range(self.config.max_retries):
            try:
                if attempt > 0:
                    delay = self.config.retry_delay_ms * (2 ** (attempt - 1)) / 1000.0
                    time.sleep(delay)
                
                result = self._execute_with_timeout(inference_func, params, inputs, **kwargs)
                return result
                
            except Exception as e:
                last_exception = e
                self.logger.warning(f"Retry attempt {attempt + 1} failed: {e}")
        
        raise last_exception
    
    def _fallback_inference(self, 
                           inference_func: Callable,
                           params: Dict[str, Any],
                           inputs: jnp.ndarray,
                           **kwargs) -> jnp.ndarray:
        """Use simplified fallback computation."""
        try:
            # Try with reduced precision
            simplified_inputs = jnp.round(inputs, decimals=2)
            return self._execute_with_timeout(inference_func, params, simplified_inputs, **kwargs)
        except:
            # Ultimate fallback: linear approximation
            return self._linear_approximation(inputs)
    
    def _degraded_inference(self, inputs: jnp.ndarray) -> jnp.ndarray:
        """Provide degraded but safe inference output."""
        self.system_state = SystemState.DEGRADED
        
        # Simple heuristic-based output
        batch_size = inputs.shape[0]
        output_dim = 2  # Assume 2D output for demonstration
        
        # Generate conservative output based on input statistics
        input_mean = jnp.mean(inputs, axis=-1, keepdims=True)
        degraded_output = jnp.clip(input_mean * 0.5, -1.0, 1.0)
        
        # Expand to correct output dimensions
        if degraded_output.shape[-1] < output_dim:
            degraded_output = jnp.tile(degraded_output, (1, output_dim))
        
        self.logger.warning("Operating in degraded mode")
        return degraded_output[:, :output_dim]
    
    def _switch_to_redundant_model(self, 
                                  inference_func: Callable,
                                  params: Dict[str, Any],
                                  inputs: jnp.ndarray,
                                  **kwargs) -> jnp.ndarray:
        """Switch to backup model if available."""
        if not self.config.enable_redundancy or len(self.redundant_models) <= 1:
            raise RuntimeError("No redundant models available")
        
        # Try next available model
        for i in range(1, len(self.redundant_models)):
            try:
                backup_index = (self.primary_model_index + i) % len(self.redundant_models)
                backup_model = self.redundant_models[backup_index]
                
                # Assuming the backup model has similar interface
                result = backup_model.apply(params, inputs, **kwargs)
                
                self.primary_model_index = backup_index
                self.logger.info(f"Switched to redundant model {backup_index}")
                return result
                
            except Exception as e:
                self.logger.warning(f"Backup model {backup_index} also failed: {e}")
        
        raise RuntimeError("All redundant models failed")
    
    def _safe_shutdown(self, inputs: jnp.ndarray) -> jnp.ndarray:
        """Initiate safe system shutdown."""
        self.system_state = SystemState.SHUTDOWN
        self.logger.critical("Initiating safe shutdown")
        
        # Return zero output for safety
        batch_size = inputs.shape[0]
        return jnp.zeros((batch_size, 2))  # Assume 2D output
    
    def _emergency_output(self, inputs: jnp.ndarray) -> jnp.ndarray:
        """Generate emergency safe output when all else fails."""
        self.system_state = SystemState.CRITICAL
        self.logger.critical("All recovery attempts failed, using emergency output")
        
        # Return minimal safe output
        batch_size = inputs.shape[0]
        return jnp.zeros((batch_size, 2))  # Assume 2D output
    
    def _linear_approximation(self, inputs: jnp.ndarray) -> jnp.ndarray:
        """Simple linear approximation as ultimate fallback."""
        # Simple linear transformation
        weights = jnp.array([[0.1, -0.05], [0.02, 0.08], [0.03, -0.02], [0.01, 0.04]])
        bias = jnp.array([0.0, 0.0])
        
        output = jnp.dot(inputs, weights) + bias
        return jnp.clip(output, -1.0, 1.0)
    
    def _classify_fault(self, exception: Exception) -> FaultType:
        """Classify exception into fault type."""
        error_message = str(exception).lower()
        
        if "timeout" in error_message:
            return FaultType.TIMING_VIOLATION
        elif "memory" in error_message or "allocation" in error_message:
            return FaultType.MEMORY_ERROR
        elif "nan" in error_message or "inf" in error_message:
            return FaultType.COMPUTATION_ERROR
        elif "connection" in error_message or "network" in error_message:
            return FaultType.COMMUNICATION_ERROR
        elif "sensor" in error_message:
            return FaultType.SENSOR_FAILURE
        else:
            return FaultType.HARDWARE_FAULT
    
    def _get_recovery_strategy(self, fault_type: FaultType) -> RecoveryStrategy:
        """Determine appropriate recovery strategy for fault type."""
        strategy_map = {
            FaultType.SENSOR_FAILURE: RecoveryStrategy.GRACEFUL_DEGRADATION,
            FaultType.COMMUNICATION_ERROR: RecoveryStrategy.RETRY,
            FaultType.MEMORY_ERROR: RecoveryStrategy.FALLBACK,
            FaultType.COMPUTATION_ERROR: RecoveryStrategy.RETRY,
            FaultType.TIMING_VIOLATION: RecoveryStrategy.FALLBACK,
            FaultType.ENERGY_DEPLETION: RecoveryStrategy.GRACEFUL_DEGRADATION,
            FaultType.HARDWARE_FAULT: RecoveryStrategy.REDUNDANCY_SWITCH
        }
        
        return strategy_map.get(fault_type, RecoveryStrategy.SAFE_SHUTDOWN)
    
    def _start_watchdog(self):
        """Start watchdog timer."""
        if self.watchdog_timer:
            self.watchdog_timer.cancel()
        
        self.watchdog_timer = threading.Timer(
            self.config.watchdog_timeout_ms / 1000.0,
            self._watchdog_timeout
        )
        self.watchdog_timer.start()
    
    def _stop_watchdog(self):
        """Stop watchdog timer."""
        if self.watchdog_timer:
            self.watchdog_timer.cancel()
            self.watchdog_timer = None
    
    def _watchdog_timeout(self):
        """Handle watchdog timeout."""
        self.logger.error("Watchdog timeout detected")
        self.system_state = SystemState.CRITICAL
        # In a real system, this might trigger a system reset
    
    def get_fault_report(self) -> Dict[str, Any]:
        """Generate comprehensive fault tolerance report."""
        recent_faults = self.fault_history[-10:] if self.fault_history else []
        
        fault_counts = {}
        for fault in self.fault_history:
            fault_type = fault.fault_type.value
            fault_counts[fault_type] = fault_counts.get(fault_type, 0) + 1
        
        recovery_success_rate = 0.0
        if self.fault_history:
            successful_recoveries = sum(1 for f in self.fault_history if f.recovery_successful)
            recovery_success_rate = successful_recoveries / len(self.fault_history) * 100
        
        return {
            "system_state": self.system_state.value,
            "total_inferences": self.inference_count,
            "total_faults": len(self.fault_history),
            "fault_counts": fault_counts,
            "recovery_success_rate": recovery_success_rate,
            "recent_faults": [
                {
                    "timestamp": f.timestamp,
                    "type": f.fault_type.value,
                    "message": f.message,
                    "recovery": f.recovery_action.value if f.recovery_action else None,
                    "successful": f.recovery_successful,
                    "recovery_time_ms": f.recovery_time_ms
                }
                for f in recent_faults
            ],
            "energy_level": self.energy_level,
            "active_model_index": self.primary_model_index,
            "available_models": len(self.redundant_models),
            "checkpoints": list(self.checkpoints.keys())
        }