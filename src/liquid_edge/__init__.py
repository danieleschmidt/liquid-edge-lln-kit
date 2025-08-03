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