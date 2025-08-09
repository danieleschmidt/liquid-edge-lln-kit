"""High-performance optimized liquid neural network layers."""

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Optional, Tuple
from functools import partial


class FastLiquidCell(nn.Module):
    """Ultra-fast liquid neural network cell optimized for real-time inference."""
    
    features: int
    tau_min: float = 10.0
    tau_max: float = 100.0
    sparsity: float = 0.0
    dt: float = 0.1
    use_fast_approx: bool = True
    
    def setup(self):
        """Initialize optimized liquid cell."""
        # Use smaller precision for edge deployment
        kernel_init = nn.initializers.lecun_normal()
        
        # Input projection (dense is often faster than sparse for small sizes)
        self.input_proj = nn.Dense(
            self.features,
            kernel_init=kernel_init,
            use_bias=False  # Remove bias for faster computation
        )
        
        # Optimized recurrent weights with optional sparsity
        if self.sparsity > 0.0:
            self.recurrent_proj = SparseLinearOptimized(
                self.features, 
                self.sparsity
            )
        else:
            self.recurrent_proj = nn.Dense(
                self.features,
                kernel_init=nn.initializers.orthogonal(),
                use_bias=False
            )
        
        # Learnable time constants (vectorized)
        self.log_tau = self.param(
            'log_tau',
            lambda key, shape: jax.random.uniform(
                key, shape, minval=jnp.log(self.tau_min), maxval=jnp.log(self.tau_max)
            ),
            (self.features,)
        )
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, h: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        """Ultra-fast forward pass."""
        # Fast input processing
        input_contrib = self.input_proj(x)
        recurrent_contrib = self.recurrent_proj(h)
        
        # Pre-compute time constants
        tau = jnp.exp(self.log_tau)
        alpha = self.dt / tau  # Pre-computed integration step
        
        # Fast activation (use faster approximations in inference mode)
        total_input = input_contrib + recurrent_contrib
        
        if self.use_fast_approx and not training:
            # Fast tanh approximation: tanh(x) ≈ x / (1 + |x|)
            activation = total_input / (1.0 + jnp.abs(total_input))
        else:
            activation = jnp.tanh(total_input)
        
        # Optimized liquid dynamics update
        # h_new = h + α * (-h + activation) = h(1-α) + α * activation
        h_new = h * (1.0 - alpha) + alpha * activation
        
        return h_new


class SparseLinearOptimized(nn.Module):
    """Optimized sparse linear layer with pre-computed indices."""
    
    features: int
    sparsity: float
    
    def setup(self):
        """Initialize optimized sparse connectivity."""
        # Create sparse mask as a parameter (deterministic)
        prob_keep = 1.0 - self.sparsity
        
        def init_sparse_mask(key, shape):
            return jax.random.bernoulli(key, prob_keep, shape).astype(jnp.float32)
        
        self.sparse_mask = self.param(
            'sparse_mask',
            init_sparse_mask,
            (self.features, self.features)
        )
        
        # Store weights
        self.weights = self.param(
            'weights',
            nn.initializers.orthogonal(),
            (self.features, self.features)
        )
    
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Fast sparse matrix multiplication."""
        # Apply mask directly (JAX optimizes this well)
        sparse_weights = self.sparse_mask * self.weights
        return x @ sparse_weights


class LiquidNNOptimized(nn.Module):
    """Highly optimized Liquid Neural Network for production deployment."""
    
    config: any  # LiquidConfig
    
    def setup(self):
        """Initialize optimized liquid network."""
        # Use fast liquid cell
        self.liquid_cell = FastLiquidCell(
            features=self.config.hidden_dim,
            tau_min=self.config.tau_min,
            tau_max=self.config.tau_max,
            sparsity=self.config.sparsity if self.config.use_sparse else 0.0,
            dt=self.config.dt,
            use_fast_approx=True  # Enable fast approximations
        )
        
        # Lightweight output projection
        self.output_proj = nn.Dense(
            self.config.output_dim,
            kernel_init=nn.initializers.lecun_normal(),
            use_bias=True  # Keep bias for output layer
        )
        
        # Optional layer norm (can be disabled for speed)
        if hasattr(self.config, 'use_layer_norm') and self.config.use_layer_norm:
            self.layer_norm = nn.LayerNorm()
        else:
            self.layer_norm = None
    
    @partial(jax.jit, static_argnums=(0,))  # JIT compile for maximum speed
    def __call__(self, 
                 x: jnp.ndarray, 
                 hidden: Optional[jnp.ndarray] = None,
                 training: bool = False) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Ultra-fast forward pass."""
        batch_size = x.shape[0]
        
        # Initialize hidden state if needed
        if hidden is None:
            hidden = jnp.zeros((batch_size, self.config.hidden_dim))
        
        # Optional input normalization (lightweight)
        if self.layer_norm is not None:
            x_norm = self.layer_norm(x)
        else:
            x_norm = x
        
        # Fast liquid dynamics
        new_hidden = self.liquid_cell(x_norm, hidden, training=training)
        
        # Output projection
        output = self.output_proj(new_hidden)
        
        return output, new_hidden


class EnergyEfficientLiquidCell(nn.Module):
    """Energy-optimized liquid cell for ultra-low power edge devices."""
    
    features: int
    tau_min: float = 10.0
    tau_max: float = 100.0
    quantization_bits: int = 8
    
    def setup(self):
        """Initialize energy-efficient cell."""
        # Quantization-aware weights
        self.input_proj = nn.Dense(
            self.features,
            kernel_init=nn.initializers.lecun_normal(),
            use_bias=False
        )
        
        # Minimal recurrent connections
        self.recurrent_proj = nn.Dense(
            self.features,
            kernel_init=nn.initializers.orthogonal(0.9),  # Slightly smaller init
            use_bias=False
        )
        
        # Fixed time constants to avoid exp() computation
        tau_values = jnp.linspace(self.tau_min, self.tau_max, self.features)
        self.alpha = 1.0 / tau_values  # Pre-computed integration steps
    
    def __call__(self, x: jnp.ndarray, h: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        """Energy-efficient forward pass."""
        # Quantize activations for lower power
        if not training:
            x = self._quantize_activation(x)
            h = self._quantize_activation(h)
        
        # Minimal computation liquid dynamics
        input_part = self.input_proj(x)
        recurrent_part = self.recurrent_proj(h)
        
        # Use ReLU instead of tanh (much faster on edge devices)
        activation = nn.relu(input_part + recurrent_part)
        
        # Simple integration (vectorized)
        h_new = h + self.alpha * (-h + activation)
        
        return h_new
    
    def _quantize_activation(self, x: jnp.ndarray) -> jnp.ndarray:
        """Quantize activations to reduce energy."""
        # Simple quantization to int8 range
        scale = 2.0 ** (self.quantization_bits - 1)
        x_quantized = jnp.round(jnp.clip(x * scale, -scale, scale - 1)) / scale
        return x_quantized


class QuantizedDense(nn.Module):
    """Quantized dense layer for deployment."""
    
    features: int
    bits: int = 8
    
    def setup(self):
        """Initialize quantized weights."""
        self.weights = self.param(
            'weights',
            nn.initializers.lecun_normal(),
            (self.features,)  # This will be expanded based on input
        )
        self.bias = self.param(
            'bias',
            nn.initializers.zeros,
            (self.features,)
        )
        
        # Quantization parameters
        self.scale = 2.0 ** (self.bits - 1)
    
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass with quantized weights."""
        # Dynamically determine weight shape
        if len(self.weights.shape) == 1:
            # Initialize weights properly on first call
            weights = self.param(
                'weights_full',
                nn.initializers.lecun_normal(),
                (x.shape[-1], self.features)
            )
        else:
            weights = self.weights
        
        # Quantize weights
        weights_q = jnp.round(jnp.clip(weights * self.scale, -self.scale, self.scale - 1)) / self.scale
        
        return x @ weights_q + self.bias