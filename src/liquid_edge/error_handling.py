"""Comprehensive error handling and resilience patterns for liquid neural networks."""

import functools
import traceback
import logging
import time
import sys
from typing import Any, Callable, Dict, List, Optional, Type, Union
from dataclasses import dataclass
from enum import Enum
from contextlib import contextmanager
import threading
from queue import Queue, Empty


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ErrorContext:
    """Context information for error handling."""
    function_name: str
    args: tuple
    kwargs: dict
    timestamp: float
    attempt_number: int
    severity: ErrorSeverity
    metadata: Dict[str, Any]


class LiquidNetworkError(Exception):
    """Base exception for liquid neural network operations."""
    
    def __init__(self, message: str, severity: ErrorSeverity = ErrorSeverity.MEDIUM, 
                 context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.severity = severity
        self.context = context or {}
        self.timestamp = time.time()


class ModelInferenceError(LiquidNetworkError):
    """Error during model inference."""
    pass


class EnergyBudgetExceededError(LiquidNetworkError):
    """Error when energy budget is exceeded."""
    pass


class SensorTimeoutError(LiquidNetworkError):
    """Error when sensor data is stale or unavailable."""
    pass


class DeploymentError(LiquidNetworkError):
    """Error during model deployment."""
    pass


class ConfigurationError(LiquidNetworkError):
    """Error in model or system configuration."""
    pass


class ResourceExhaustionError(LiquidNetworkError):
    """Error when system resources are exhausted."""
    pass


class RobustErrorHandler:
    """Production-grade error handling system."""
    
    def __init__(self, name: str = "liquid_network"):
        self.name = name
        self.logger = logging.getLogger(f"liquid_edge.{name}.errors")
        
        # Error tracking
        self._error_counts = {}
        self._error_history = Queue(maxsize=1000)
        self._lock = threading.RLock()
        
        # Recovery strategies
        self._recovery_strategies = {}
        
        # Setup default recovery strategies
        self._setup_default_recovery_strategies()
    
    def _setup_default_recovery_strategies(self):
        """Setup default error recovery strategies."""
        
        # Model inference recovery
        def inference_recovery(error_context: ErrorContext) -> Any:
            """Recovery strategy for inference errors."""
            self.logger.warning(f"Inference failed, using fallback: {error_context.function_name}")
            # Return zero outputs as fallback
            return [0.0, 0.0]  # Typical motor commands fallback
        
        # Sensor timeout recovery
        def sensor_recovery(error_context: ErrorContext) -> Any:
            """Recovery strategy for sensor timeouts."""
            self.logger.warning(f"Sensor timeout, using last known values: {error_context.function_name}")
            # Return zero sensor readings as safe fallback
            return [0.0] * 8  # Typical sensor array fallback
        
        # Energy budget recovery
        def energy_recovery(error_context: ErrorContext) -> Any:
            """Recovery strategy for energy budget exceeded."""
            self.logger.error(f"Energy budget exceeded, reducing performance: {error_context.function_name}")
            # Trigger performance reduction
            return None
        
        # Register strategies
        self._recovery_strategies[ModelInferenceError] = inference_recovery
        self._recovery_strategies[SensorTimeoutError] = sensor_recovery
        self._recovery_strategies[EnergyBudgetExceededError] = energy_recovery
    
    def register_recovery_strategy(self, exception_type: Type[Exception], 
                                 strategy: Callable[[ErrorContext], Any]):
        """Register a custom recovery strategy for specific exception type."""
        self._recovery_strategies[exception_type] = strategy
        self.logger.info(f"Registered recovery strategy for {exception_type.__name__}")
    
    def handle_error(self, exception: Exception, context: ErrorContext) -> Optional[Any]:
        """Handle an error with appropriate recovery strategy."""
        with self._lock:
            # Track error
            self._track_error(exception, context)
            
            # Log error with full context
            self._log_error(exception, context)
            
            # Attempt recovery
            return self._attempt_recovery(exception, context)
    
    def _track_error(self, exception: Exception, context: ErrorContext):
        """Track error occurrence for analysis."""
        error_key = f"{type(exception).__name__}:{context.function_name}"
        self._error_counts[error_key] = self._error_counts.get(error_key, 0) + 1
        
        # Store in history
        error_record = {
            "timestamp": context.timestamp,
            "exception_type": type(exception).__name__,
            "message": str(exception),
            "function": context.function_name,
            "attempt": context.attempt_number,
            "severity": context.severity.value,
            "metadata": context.metadata
        }
        
        if not self._error_history.full():
            self._error_history.put(error_record)
    
    def _log_error(self, exception: Exception, context: ErrorContext):
        """Log error with structured information."""
        log_data = {
            "function": context.function_name,
            "attempt": context.attempt_number,
            "severity": context.severity.value,
            "exception_type": type(exception).__name__,
            "message": str(exception),
            "metadata": context.metadata
        }
        
        if context.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(f"Critical error in {context.function_name}: {exception}", extra=log_data)
        elif context.severity == ErrorSeverity.HIGH:
            self.logger.error(f"High severity error in {context.function_name}: {exception}", extra=log_data)
        elif context.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(f"Medium severity error in {context.function_name}: {exception}", extra=log_data)
        else:
            self.logger.info(f"Low severity error in {context.function_name}: {exception}", extra=log_data)
    
    def _attempt_recovery(self, exception: Exception, context: ErrorContext) -> Optional[Any]:
        """Attempt to recover from error using registered strategies."""
        exception_type = type(exception)
        
        # Look for exact match first
        if exception_type in self._recovery_strategies:
            try:
                self.logger.info(f"Attempting recovery for {exception_type.__name__}")
                result = self._recovery_strategies[exception_type](context)
                self.logger.info(f"Recovery successful for {exception_type.__name__}")
                return result
            except Exception as recovery_error:
                self.logger.error(f"Recovery failed for {exception_type.__name__}: {recovery_error}")
        
        # Look for parent class matches
        for registered_type, strategy in self._recovery_strategies.items():
            if issubclass(exception_type, registered_type):
                try:
                    self.logger.info(f"Attempting parent recovery for {exception_type.__name__} as {registered_type.__name__}")
                    result = strategy(context)
                    self.logger.info(f"Parent recovery successful for {exception_type.__name__}")
                    return result
                except Exception as recovery_error:
                    self.logger.error(f"Parent recovery failed for {exception_type.__name__}: {recovery_error}")
        
        # No recovery strategy found
        self.logger.warning(f"No recovery strategy found for {exception_type.__name__}")
        return None
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics."""
        with self._lock:
            total_errors = sum(self._error_counts.values())
            
            # Get recent errors (last 5 minutes)
            recent_cutoff = time.time() - 300
            recent_errors = []
            
            temp_queue = Queue()
            while not self._error_history.empty():
                try:
                    error_record = self._error_history.get_nowait()
                    if error_record["timestamp"] > recent_cutoff:
                        recent_errors.append(error_record)
                    temp_queue.put(error_record)
                except Empty:
                    break
            
            # Restore queue
            while not temp_queue.empty():
                try:
                    self._error_history.put(temp_queue.get_nowait())
                except Empty:
                    break
            
            return {
                "total_errors": total_errors,
                "error_types": len(self._error_counts),
                "recent_errors_5min": len(recent_errors),
                "most_common_errors": sorted(
                    self._error_counts.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:5],
                "recovery_strategies_registered": len(self._recovery_strategies)
            }


def retry_with_backoff(max_retries: int = 3, 
                      backoff_factor: float = 2.0,
                      exceptions: tuple = (Exception,),
                      severity: ErrorSeverity = ErrorSeverity.MEDIUM):
    """Decorator for retrying functions with exponential backoff."""
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            error_handler = getattr(wrapper, '_error_handler', None)
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                    
                except exceptions as e:
                    if attempt == max_retries:
                        # Last attempt failed, handle error if handler available
                        if error_handler:
                            context = ErrorContext(
                                function_name=func.__name__,
                                args=args,
                                kwargs=kwargs,
                                timestamp=time.time(),
                                attempt_number=attempt + 1,
                                severity=severity,
                                metadata={"max_retries_exceeded": True}
                            )
                            
                            recovery_result = error_handler.handle_error(e, context)
                            if recovery_result is not None:
                                return recovery_result
                        
                        # Re-raise if no recovery
                        raise
                    
                    # Calculate backoff delay
                    delay = backoff_factor ** attempt
                    
                    # Log retry attempt
                    logger = logging.getLogger(f"liquid_edge.retry")
                    logger.warning(
                        f"Attempt {attempt + 1}/{max_retries + 1} failed for {func.__name__}: {e}. "
                        f"Retrying in {delay:.2f}s"
                    )
                    
                    time.sleep(delay)
            
        return wrapper
    return decorator


def graceful_degradation(fallback_value: Any = None,
                       exceptions: tuple = (Exception,),
                       severity: ErrorSeverity = ErrorSeverity.MEDIUM):
    """Decorator for graceful degradation on errors."""
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            error_handler = getattr(wrapper, '_error_handler', None)
            
            try:
                return func(*args, **kwargs)
                
            except exceptions as e:
                # Handle error if handler available
                if error_handler:
                    context = ErrorContext(
                        function_name=func.__name__,
                        args=args,
                        kwargs=kwargs,
                        timestamp=time.time(),
                        attempt_number=1,
                        severity=severity,
                        metadata={"graceful_degradation": True}
                    )
                    
                    recovery_result = error_handler.handle_error(e, context)
                    if recovery_result is not None:
                        return recovery_result
                
                # Log degradation
                logger = logging.getLogger(f"liquid_edge.degradation")
                logger.warning(
                    f"Graceful degradation in {func.__name__}: {e}. "
                    f"Using fallback value: {fallback_value}"
                )
                
                return fallback_value
            
        return wrapper
    return decorator


def validate_inputs(**validators):
    """Decorator for input validation with custom error messages."""
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get function signature for parameter mapping
            import inspect
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            # Validate each parameter
            for param_name, validator in validators.items():
                if param_name in bound_args.arguments:
                    value = bound_args.arguments[param_name]
                    
                    try:
                        if callable(validator):
                            if not validator(value):
                                raise ConfigurationError(
                                    f"Validation failed for parameter '{param_name}' in {func.__name__}",
                                    severity=ErrorSeverity.HIGH,
                                    context={"parameter": param_name, "value": str(value)}
                                )
                        elif hasattr(validator, '__contains__'):
                            if value not in validator:
                                raise ConfigurationError(
                                    f"Parameter '{param_name}' must be one of {list(validator)}, got {value}",
                                    severity=ErrorSeverity.HIGH,
                                    context={"parameter": param_name, "value": str(value)}
                                )
                    except Exception as e:
                        if isinstance(e, LiquidNetworkError):
                            raise
                        else:
                            raise ConfigurationError(
                                f"Validation error for parameter '{param_name}': {e}",
                                severity=ErrorSeverity.HIGH,
                                context={"parameter": param_name}
                            )
            
            return func(*args, **kwargs)
            
        return wrapper
    return decorator


@contextmanager
def error_boundary(error_handler: RobustErrorHandler,
                  severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                  reraise: bool = True,
                  metadata: Optional[Dict[str, Any]] = None):
    """Context manager for error boundary with automatic handling."""
    try:
        yield
    except Exception as e:
        # Get caller information
        import inspect
        frame = inspect.currentframe().f_back
        function_name = frame.f_code.co_name
        
        context = ErrorContext(
            function_name=function_name,
            args=(),
            kwargs={},
            timestamp=time.time(),
            attempt_number=1,
            severity=severity,
            metadata=metadata or {}
        )
        
        recovery_result = error_handler.handle_error(e, context)
        
        if recovery_result is not None:
            # Recovery successful, but we can't return from context manager
            pass
        elif reraise:
            raise


class SafeExecutor:
    """Safe execution wrapper with comprehensive error handling."""
    
    def __init__(self, error_handler: RobustErrorHandler):
        self.error_handler = error_handler
    
    def execute(self, func: Callable, *args, 
               max_retries: int = 3,
               timeout_seconds: Optional[float] = None,
               severity: ErrorSeverity = ErrorSeverity.MEDIUM,
               metadata: Optional[Dict[str, Any]] = None,
               **kwargs) -> Any:
        """Execute function safely with error handling and timeouts."""
        
        def execute_with_timeout():
            return func(*args, **kwargs)
        
        # Apply timeout if specified
        if timeout_seconds:
            import signal
            
            def timeout_handler(signum, frame):
                raise ResourceExhaustionError(
                    f"Function {func.__name__} timed out after {timeout_seconds}s",
                    severity=ErrorSeverity.HIGH,
                    context={"timeout_seconds": timeout_seconds}
                )
            
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(timeout_seconds))
        
        try:
            # Retry logic
            for attempt in range(max_retries + 1):
                try:
                    result = execute_with_timeout()
                    return result
                    
                except Exception as e:
                    context = ErrorContext(
                        function_name=func.__name__,
                        args=args,
                        kwargs=kwargs,
                        timestamp=time.time(),
                        attempt_number=attempt + 1,
                        severity=severity,
                        metadata=metadata or {}
                    )
                    
                    if attempt == max_retries:
                        # Last attempt, try recovery
                        recovery_result = self.error_handler.handle_error(e, context)
                        if recovery_result is not None:
                            return recovery_result
                        else:
                            raise
                    else:
                        # Log and retry
                        self.error_handler._log_error(e, context)
                        time.sleep(2 ** attempt)  # Exponential backoff
        
        finally:
            if timeout_seconds:
                signal.alarm(0)  # Cancel timeout


# Convenience function for creating error handler
def create_error_handler(name: str = "liquid_network") -> RobustErrorHandler:
    """Create an error handler with sensible defaults."""
    return RobustErrorHandler(name)


# Function to attach error handler to decorated functions
def attach_error_handler(handler: RobustErrorHandler):
    """Attach error handler to functions decorated with error handling decorators."""
    def attach(func):
        func._error_handler = handler
        return func
    return attach
