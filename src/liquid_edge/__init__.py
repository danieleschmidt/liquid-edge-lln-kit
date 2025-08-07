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
from .profiling import EnergyProfiler, ProfilingConfig, ModelEnergyOptimizer\n\n# Production robustness\nfrom .monitoring import LiquidNetworkMonitor, PerformanceMetrics, AlertLevel, CircuitBreaker\nfrom .error_handling import (\n    RobustErrorHandler, LiquidNetworkError, ModelInferenceError,\n    EnergyBudgetExceededError, SensorTimeoutError, ErrorSeverity,\n    retry_with_backoff, graceful_degradation, validate_inputs\n)

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
]\n\n# Add robustness components\n__all__.extend([\n    \"LiquidNetworkMonitor\",\n    \"PerformanceMetrics\", \n    \"AlertLevel\",\n    \"CircuitBreaker\",\n    \"RobustErrorHandler\",\n    \"LiquidNetworkError\",\n    \"ModelInferenceError\",\n    \"EnergyBudgetExceededError\",\n    \"SensorTimeoutError\",\n    \"ErrorSeverity\",\n    \"retry_with_backoff\",\n    \"graceful_degradation\",\n    \"validate_inputs\"\n])\n\n# Conditionally add ROS 2 components\nif _ros2_available:\n    __all__.extend([\n        \"LiquidController\",\n        \"ROS2Config\", \n        \"LiquidTurtleBot\",\n        \"LiquidNavigationNode\"\n    ])