#!/usr/bin/env python3
"""Simple test to debug core functionality."""

import jax
import jax.numpy as jnp
from liquid_edge import LiquidNN, LiquidConfig

def main():
    print("🧠 Testing basic Liquid NN functionality...")
    
    # Simple configuration
    config = LiquidConfig(
        input_dim=4,
        hidden_dim=8,
        output_dim=2,
        tau_min=10.0,
        tau_max=100.0,
        use_sparse=False  # Disable sparsity for debugging
    )
    
    # Create model
    model = LiquidNN(config)
    
    # Test data
    key = jax.random.PRNGKey(42)
    x = jnp.ones((2, 4))  # batch of 2, 4 features
    
    print(f"Input shape: {x.shape}")
    
    # Initialize
    params = model.init(key, x, training=False)
    print("✓ Model initialized successfully")
    
    # Forward pass
    output, hidden = model.apply(params, x, training=False)
    print(f"✓ Forward pass successful - Output shape: {output.shape}, Hidden shape: {hidden.shape}")
    
    # Test energy estimation
    energy = model.energy_estimate()
    print(f"✓ Energy estimate: {energy:.2f}mW")
    
    print("\n🎉 Basic functionality working!")

if __name__ == "__main__":
    main()