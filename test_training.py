#!/usr/bin/env python3
"""Simple training test."""

import jax
import jax.numpy as jnp
import optax
from liquid_edge import LiquidNN, LiquidConfig

def main():
    print("🚀 Testing simplified training...")
    
    # Configuration
    config = LiquidConfig(
        input_dim=4,
        hidden_dim=8,  
        output_dim=2,
        tau_min=10.0,
        tau_max=100.0,
        use_sparse=False,
        learning_rate=0.001
    )
    
    # Model and optimizer
    model = LiquidNN(config)
    optimizer = optax.adam(config.learning_rate)
    
    # Synthetic data
    key = jax.random.PRNGKey(42)
    batch_size = 32
    
    inputs = jax.random.normal(key, (batch_size, 4))
    targets = jax.random.normal(key, (batch_size, 2))
    
    print(f"Data shapes - inputs: {inputs.shape}, targets: {targets.shape}")
    
    # Initialize
    params = model.init(key, inputs[:1], training=True)
    opt_state = optimizer.init(params)
    
    print("✓ Initialization successful")
    
    # Single training step
    def loss_fn(params):
        outputs, _ = model.apply(params, inputs, training=True)
        return jnp.mean((outputs - targets) ** 2)
    
    loss_val, grads = jax.value_and_grad(loss_fn)(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    
    print(f"✓ Training step successful - Loss: {loss_val:.4f}")
    
    # Test inference
    test_outputs, _ = model.apply(params, inputs[:5], training=False)
    print(f"✓ Inference successful - Output shape: {test_outputs.shape}")
    
    print("\n🎉 Training functionality working!")

if __name__ == "__main__":
    main()