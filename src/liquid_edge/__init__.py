"""Liquid Edge LLN Kit - Tiny liquid neural networks for edge robotics."""

__version__ = "0.1.0"
__author__ = "Daniel Schmidt"

# Core components
from .core import LiquidNN, LiquidConfig, EnergyAwareTrainer

# Advanced layers
from .layers import (
    LiquidCell, 
    LiquidRNN, 
    AdvancedLiquidCell,
    SparseLinear,
    MultiModalLiquidFusion,
    EnergyEfficientLiquidCell,
    QuantizedDense
)

# Deployment tools
from .deploy import MCUDeployer, TargetDevice, DeploymentConfig

# Profiling and optimization
from .profiling import EnergyProfiler, ProfilingConfig, ModelEnergyOptimizer

# Production robustness
from .monitoring import LiquidNetworkMonitor, PerformanceMetrics, AlertLevel, CircuitBreaker
from .error_handling import (
    RobustErrorHandler, LiquidNetworkError, ModelInferenceError,
    EnergyBudgetExceededError, SensorTimeoutError, ErrorSeverity,
    retry_with_backoff, graceful_degradation, validate_inputs
)

# ROS 2 integration (optional)
try:
    from .ros2_integration import LiquidController, ROS2Config, LiquidTurtleBot, LiquidNavigationNode
    _ros2_available = True
except ImportError:
    _ros2_available = False

__all__ = [
    # Core
    "LiquidNN", 
    "LiquidConfig",
    "EnergyAwareTrainer",
    
    # Layers
    "LiquidCell",
    "LiquidRNN",
    "AdvancedLiquidCell", 
    "SparseLinear",
    "MultiModalLiquidFusion",
    "EnergyEfficientLiquidCell",
    "QuantizedDense",
    
    # Deployment
    "MCUDeployer",
    "TargetDevice", 
    "DeploymentConfig",
    
    # Profiling
    "EnergyProfiler",
    "ProfilingConfig",
    "ModelEnergyOptimizer"
]

# Add robustness components
__all__.extend([
    "LiquidNetworkMonitor",
    "PerformanceMetrics", 
    "AlertLevel",
    "CircuitBreaker",
    "RobustErrorHandler",
    "LiquidNetworkError",
    "ModelInferenceError",
    "EnergyBudgetExceededError",
    "SensorTimeoutError",
    "ErrorSeverity",
    "retry_with_backoff",
    "graceful_degradation",
    "validate_inputs"
])

# Conditionally add ROS 2 components
if _ros2_available:
    __all__.extend([
        "LiquidController",
        "ROS2Config", 
        "LiquidTurtleBot",
        "LiquidNavigationNode"
    ])