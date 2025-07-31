"""Core liquid neural network implementations."""

from typing import Dict, Any, Optional
import jax.numpy as jnp
from flax import linen as nn
from dataclasses import dataclass


@dataclass
class LiquidConfig:
    """Configuration for liquid neural networks."""
    
    input_size: int
    hidden_size: int
    output_size: int
    time_constant: float = 1.0
    sensory_sigma: float = 0.1
    sensory_mu: float = 0.3
    learning_rate: float = 0.001


class LiquidNN(nn.Module):
    """Liquid Neural Network for edge computing applications."""
    
    config: LiquidConfig
    
    def setup(self):
        """Initialize the liquid neural network layers."""
        from .layers import LiquidCell
        self.liquid_cell = LiquidCell(
            features=self.config.hidden_size,
            time_constant=self.config.time_constant
        )
        self.output_layer = nn.Dense(self.config.output_size)
    
    def __call__(self, x: jnp.ndarray, hidden: Optional[jnp.ndarray] = None) -> tuple:
        """Forward pass through the liquid neural network."""
        if hidden is None:
            hidden = jnp.zeros((x.shape[0], self.config.hidden_size))
            
        new_hidden = self.liquid_cell(x, hidden)
        output = self.output_layer(new_hidden)
        
        return output, new_hidden