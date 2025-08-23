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

# Optimized high-performance layers
from .optimized_layers import (
    FastLiquidCell,
    LiquidNNOptimized,
    SparseLinearOptimized,
    EnergyEfficientLiquidCell as EnergyOptimizedCell
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
    
    # Optimized layers
    "FastLiquidCell",
    "LiquidNNOptimized", 
    "SparseLinearOptimized",
    "EnergyOptimizedCell",
    
    # Deployment
    "MCUDeployer",
    "TargetDevice", 
    "DeploymentConfig",
    
    # Profiling
    "EnergyProfiler",
    "ProfilingConfig",
    "ModelEnergyOptimizer"
]

# Advanced security and fault tolerance
from .advanced_security import (
    SecurityConfig, SecurityMonitor, SecureLiquidInference, 
    SecurityError, ThreatLevel, SecurityEvent, secure_inference
)
from .fault_tolerance import (
    FaultToleranceConfig, FaultTolerantSystem, FaultType, 
    RecoveryStrategy, SystemState, FaultEvent
)

# High-performance inference and scaling
from .high_performance_inference import (
    HighPerformanceInferenceEngine, PerformanceConfig, InferenceMode,
    LoadBalancingStrategy, InferenceRequest, InferenceMetrics,
    DistributedInferenceCoordinator
)

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
    "validate_inputs",
    # Advanced security
    "SecurityConfig",
    "SecurityMonitor", 
    "SecureLiquidInference",
    "SecurityError",
    "ThreatLevel",
    "SecurityEvent",
    "secure_inference",
    # Fault tolerance
    "FaultToleranceConfig",
    "FaultTolerantSystem",
    "FaultType",
    "RecoveryStrategy", 
    "SystemState",
    "FaultEvent",
    # High-performance inference
    "HighPerformanceInferenceEngine",
    "PerformanceConfig",
    "InferenceMode",
    "LoadBalancingStrategy",
    "InferenceRequest",
    "InferenceMetrics",
    "DistributedInferenceCoordinator"
])

# Autonomous Evolutionary SDLC (Breakthrough Innovation)
from .autonomous_evolutionary_sdlc import (
    AutonomousEvolutionarySDLC, SDLCGenome, EvolutionaryConfig,
    EvolutionaryPhase, OptimizationObjective, create_autonomous_evolutionary_sdlc
)

# Conditionally add ROS 2 components
if _ros2_available:
    __all__.extend([
        "LiquidController",
        "ROS2Config", 
        "LiquidTurtleBot",
        "LiquidNavigationNode"
    ])

# Internationalization (i18n) system
from .i18n import (
    Language, Region, LocaleConfig, TranslationManager,
    set_language, set_locale, translate, format_metric, format_datetime,
    get_compliance_requirements, get_localized_logger, LocalizedLogger,
    I18nConfig, get_i18n_config, get_translation_manager,
    integrate_with_error_handler, integrate_with_compliance
)

# Add evolutionary SDLC components  
__all__.extend([
    "AutonomousEvolutionarySDLC",
    "SDLCGenome", 
    "EvolutionaryConfig",
    "EvolutionaryPhase",
    "OptimizationObjective",
    "create_autonomous_evolutionary_sdlc"
])

# Add i18n components
__all__.extend([
    "Language",
    "Region", 
    "LocaleConfig",
    "TranslationManager",
    "set_language",
    "set_locale",
    "translate",
    "format_metric",
    "format_datetime",
    "get_compliance_requirements",
    "get_localized_logger",
    "LocalizedLogger",
    "I18nConfig",
    "get_i18n_config",
    "get_translation_manager",
    "integrate_with_error_handler",
    "integrate_with_compliance"
])