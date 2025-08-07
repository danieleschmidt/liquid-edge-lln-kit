"""Core liquid neural network implementations."""

from typing import Dict, Any, Optional, Tuple, Union
import jax
import jax.numpy as jnp
from flax import linen as nn
import optax
from dataclasses import dataclass, field
import numpy as np


@dataclass
class LiquidConfig:
    """Configuration for liquid neural networks."""
    
    input_dim: int
    hidden_dim: int
    output_dim: int
    tau_min: float = 10.0
    tau_max: float = 100.0
    sensory_sigma: float = 0.1
    sensory_mu: float = 0.3
    learning_rate: float = 0.001
    use_sparse: bool = True
    sparsity: float = 0.3
    energy_budget_mw: float = 100.0
    target_fps: int = 50
    quantization: str = "int8"
    dt: float = 0.1
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if self.output_dim <= 0:
            raise ValueError("output_dim must be positive")
        if self.tau_min <= 0 or self.tau_max <= 0:
            raise ValueError("time constants must be positive")
        if self.tau_min >= self.tau_max:
            raise ValueError("tau_min must be less than tau_max")
        if not 0.0 <= self.sparsity <= 1.0:
            raise ValueError("sparsity must be between 0 and 1")
        if self.energy_budget_mw <= 0:
            raise ValueError("energy_budget_mw must be positive")


class LiquidNN(nn.Module):
    """Production-ready Liquid Neural Network for edge computing."""
    
    config: LiquidConfig
    
    def setup(self):
        """Initialize the liquid neural network layers."""
        from .layers import AdvancedLiquidCell
        
        self.liquid_cell = AdvancedLiquidCell(
            features=self.config.hidden_dim,
            tau_min=self.config.tau_min,
            tau_max=self.config.tau_max,
            sparsity=self.config.sparsity if self.config.use_sparse else 0.0,
            dt=self.config.dt
        )
        
        # Output projection with optional quantization awareness
        self.output_layer = nn.Dense(
            self.config.output_dim,
            kernel_init=nn.initializers.lecun_normal()
        )
        
        # Layer normalization for stable training
        self.layer_norm = nn.LayerNorm()
    
    def __call__(self, 
                 x: jnp.ndarray, 
                 hidden: Optional[jnp.ndarray] = None,
                 training: bool = False) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Forward pass through the liquid neural network."""
        batch_size = x.shape[0]
        
        if hidden is None:
            hidden = jnp.zeros((batch_size, self.config.hidden_dim))
            
        # Normalize input for stable dynamics
        x_norm = self.layer_norm(x)
        
        # Liquid dynamics
        new_hidden = self.liquid_cell(x_norm, hidden, training=training)
        
        # Output projection
        output = self.output_layer(new_hidden)
        
        return output, new_hidden
    
    def energy_estimate(self, sequence_length: int = 1) -> float:
        """Estimate energy consumption in milliwatts."""
        # Simplified energy model based on operations
        input_ops = self.config.input_dim * self.config.hidden_dim
        recurrent_ops = self.config.hidden_dim * self.config.hidden_dim
        output_ops = self.config.hidden_dim * self.config.output_dim
        
        # Apply sparsity reduction
        if self.config.use_sparse:
            recurrent_ops *= (1.0 - self.config.sparsity)
        
        total_ops = (input_ops + recurrent_ops + output_ops) * sequence_length
        
        # Energy per operation (empirical estimate for Cortex-M7 @ 400MHz)
        energy_per_op_nj = 0.5  # nanojoules per MAC operation
        
        # Convert to milliwatts at target FPS
        energy_mw = (total_ops * energy_per_op_nj * self.config.target_fps) / 1e6
        
        return energy_mw


class EnergyAwareTrainer:
    """Energy-constrained training for liquid neural networks."""
    
    def __init__(self, 
                 model: LiquidNN, 
                 config: LiquidConfig,
                 energy_penalty: float = 0.1):
        self.model = model
        self.config = config
        self.energy_penalty = energy_penalty
        self.optimizer = optax.adam(config.learning_rate)
        
    def create_train_state(self, rng_key: jax.random.PRNGKey, input_shape: Tuple[int, ...]):
        """Initialize training state."""
        dummy_input = jnp.ones((1, *input_shape))
        params = self.model.init(rng_key, dummy_input, training=True)
        opt_state = self.optimizer.init(params)
        
        return {
            'params': params,
            'opt_state': opt_state,
            'step': 0
        }
    
    def train_step(self, 
                   state: Dict[str, Any], 
                   batch: Tuple[jnp.ndarray, jnp.ndarray]) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """Single training step with energy awareness."""
        inputs, targets = batch
        
        def loss_fn(params):
            outputs, _ = self.model.apply(params, inputs, training=True)
            
            # Task loss (MSE)
            task_loss = jnp.mean((outputs - targets) ** 2)
            
            # Energy penalty
            estimated_energy = self.model.energy_estimate(inputs.shape[1])
            energy_penalty = jnp.maximum(0.0, estimated_energy - self.config.energy_budget_mw)
            
            total_loss = task_loss + self.energy_penalty * energy_penalty
            
            return total_loss, {
                'task_loss': task_loss,
                'energy_mw': estimated_energy,
                'energy_penalty': energy_penalty,
                'total_loss': total_loss
            }
        
        (loss_val, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state['params'])
        
        updates, new_opt_state = self.optimizer.update(grads, state['opt_state'], state['params'])
        new_params = optax.apply_updates(state['params'], updates)
        
        new_state = {
            'params': new_params,
            'opt_state': new_opt_state,
            'step': state['step'] + 1
        }
        
        return new_state, metrics
    
    def train(self, 
              train_data: jnp.ndarray, 
              targets: jnp.ndarray,
              epochs: int = 100,
              batch_size: int = 32) -> Dict[str, Any]:
        """Complete training loop."""
        rng_key = jax.random.PRNGKey(42)
        state = self.create_train_state(rng_key, train_data.shape[1:])
        
        dataset_size = train_data.shape[0]
        num_batches = dataset_size // batch_size
        
        history = {'loss': [], 'energy': [], 'accuracy': []}
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_energy = 0.0
            
            # Shuffle data
            perm = jax.random.permutation(rng_key, dataset_size)
            rng_key, _ = jax.random.split(rng_key)
            
            shuffled_data = train_data[perm]
            shuffled_targets = targets[perm]
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = start_idx + batch_size
                
                batch_data = shuffled_data[start_idx:end_idx]
                batch_targets = shuffled_targets[start_idx:end_idx]
                
                state, metrics = self.train_step(state, (batch_data, batch_targets))
                epoch_loss += metrics['total_loss']
                epoch_energy += metrics['energy_mw']
            
            avg_loss = epoch_loss / num_batches
            avg_energy = epoch_energy / num_batches
            
            history['loss'].append(float(avg_loss))
            history['energy'].append(float(avg_energy))
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch:3d}: Loss={avg_loss:.4f}, Energy={avg_energy:.1f}mW")
        
        return {
            'final_params': state['params'],
            'history': history,
            'final_energy_mw': float(avg_energy)
        }