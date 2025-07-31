"""Liquid neural network layer implementations."""

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Optional


class LiquidCell(nn.Module):
    """A single liquid neural network cell."""
    
    features: int
    time_constant: float = 1.0
    
    def setup(self):
        """Initialize the liquid cell parameters."""
        self.input_projection = nn.Dense(self.features)
        self.recurrent_projection = nn.Dense(self.features)
        
    def __call__(self, inputs: jnp.ndarray, hidden: jnp.ndarray) -> jnp.ndarray:
        """Forward pass through the liquid cell."""
        input_part = self.input_projection(inputs)
        recurrent_part = self.recurrent_projection(hidden)
        
        # Liquid dynamics: dx/dt = -x + f(Wx + Uh + b)
        activation = nn.tanh(input_part + recurrent_part)
        
        # Euler integration with time constant
        dt = 1.0 / self.time_constant
        new_hidden = hidden + dt * (-hidden + activation)
        
        return new_hidden


class LiquidRNN(nn.Module):
    """Recurrent liquid neural network for sequence processing."""
    
    features: int
    time_constant: float = 1.0
    
    def setup(self):
        """Initialize the liquid RNN."""
        self.liquid_cell = LiquidCell(
            features=self.features,
            time_constant=self.time_constant
        )
    
    def __call__(self, 
                 inputs: jnp.ndarray, 
                 initial_hidden: Optional[jnp.ndarray] = None) -> tuple:
        """Process a sequence through the liquid RNN."""
        batch_size = inputs.shape[0]
        seq_len = inputs.shape[1]
        
        if initial_hidden is None:
            hidden = jnp.zeros((batch_size, self.features))
        else:
            hidden = initial_hidden
            
        outputs = []
        for t in range(seq_len):
            hidden = self.liquid_cell(inputs[:, t], hidden)
            outputs.append(hidden)
            
        return jnp.stack(outputs, axis=1), hidden