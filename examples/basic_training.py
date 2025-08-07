#!/usr/bin/env python3
"""Basic liquid neural network training example."""

import jax
import jax.numpy as jnp
import numpy as np
from liquid_edge import LiquidNN, LiquidConfig, EnergyAwareTrainer
from liquid_edge.profiling import EnergyProfiler, ProfilingConfig

def generate_sensor_data(num_samples: int = 1000, sequence_length: int = 50):
    """Generate synthetic sensor data for training."""
    # Simulate IMU + proximity sensors
    key = jax.random.PRNGKey(42)
    
    # Create time series with some patterns
    t = np.linspace(0, 10, sequence_length)
    
    data = []
    targets = []
    
    for i in range(num_samples):
        # Generate synthetic sensor readings
        imu_accel = jnp.sin(t + i * 0.1) + 0.1 * jax.random.normal(key, (sequence_length,))
        imu_gyro = jnp.cos(t + i * 0.1) + 0.1 * jax.random.normal(key, (sequence_length,))
        proximity = jnp.exp(-((t - 5) ** 2)) + 0.05 * jax.random.normal(key, (sequence_length,))
        light = jnp.maximum(0, jnp.sin(2 * t + i * 0.05)) + 0.1 * jax.random.normal(key, (sequence_length,))
        
        # Stack sensors
        sample = jnp.stack([imu_accel, imu_gyro, proximity, light], axis=1)
        
        # Generate control targets (simple obstacle avoidance)
        linear_vel = jnp.where(proximity < 0.3, 0.1, 0.5)  # Slow down near obstacles
        angular_vel = jnp.where(proximity < 0.3, jnp.sign(imu_gyro) * 0.8, 0.0)  # Turn away
        
        target = jnp.stack([linear_vel, angular_vel], axis=1)
        
        data.append(sample[-1])  # Take last timestep
        targets.append(target[-1])
        
        key, _ = jax.random.split(key)
    
    return jnp.array(data), jnp.array(targets)

def main():
    """Train a liquid neural network for robot control."""
    print("🌊 Liquid Edge LLN - Basic Training Example")
    print("=" * 50)
    
    # Configuration
    config = LiquidConfig(
        input_dim=4,           # 4 sensors: accel, gyro, proximity, light
        hidden_dim=12,         # 12 liquid neurons
        output_dim=2,          # 2 motor commands: linear, angular velocity
        tau_min=10.0,
        tau_max=100.0,
        use_sparse=True,
        sparsity=0.3,          # 70% connections pruned
        energy_budget_mw=80.0,  # 80mW power budget
        target_fps=50
    )
    
    print(f"Model configuration: {config.input_dim}→{config.hidden_dim}→{config.output_dim}")
    print(f"Energy budget: {config.energy_budget_mw}mW @ {config.target_fps}Hz")
    print(f"Sparsity: {config.sparsity:.1%}")
    
    # Generate training data
    print("\n📊 Generating synthetic sensor data...")
    train_data, train_targets = generate_sensor_data(800, 50)
    test_data, test_targets = generate_sensor_data(200, 50)
    
    print(f"Training data: {train_data.shape}, targets: {train_targets.shape}")
    print(f"Test data: {test_data.shape}, targets: {test_targets.shape}")
    
    # Create model
    print("\n🧠 Creating liquid neural network...")
    model = LiquidNN(config)
    
    # Estimate energy consumption
    estimated_energy = model.energy_estimate()
    print(f"Estimated energy: {estimated_energy:.1f}mW (budget: {config.energy_budget_mw}mW)")
    
    if estimated_energy > config.energy_budget_mw:
        print("⚠️  Model exceeds energy budget - training will optimize for efficiency")
    
    # Setup energy profiling
    print("\n⚡ Setting up energy profiler...")
    profiler_config = ProfilingConfig(
        device="esp32s3",
        voltage=3.3,
        sampling_rate=1000
    )
    profiler = EnergyProfiler(profiler_config)
    
    # Energy-aware training
    print("\n🚀 Starting energy-aware training...")
    trainer = EnergyAwareTrainer(
        model=model,
        config=config,
        energy_penalty=0.15  # 15% penalty for exceeding energy budget
    )
    
    # Train with energy measurement
    with profiler.measure("liquid_training", estimated_operations=100000):
        results = trainer.train(
            train_data=train_data,
            targets=train_targets,
            epochs=50,
            batch_size=32
        )
    
    # Training results
    final_params = results['final_params']
    training_history = results['history']
    final_energy = results['final_energy_mw']
    
    print(f"\n✅ Training completed!")
    print(f"Final energy: {final_energy:.1f}mW (target: {config.energy_budget_mw}mW)")
    print(f"Final loss: {training_history['loss'][-1]:.4f}")
    
    energy_efficiency = (config.energy_budget_mw - final_energy) / config.energy_budget_mw * 100
    print(f"Energy efficiency: {energy_efficiency:.1f}% under budget")
    
    # Test inference
    print("\n🔬 Running inference tests...")
    with profiler.measure("liquid_inference", estimated_operations=1000):
        test_outputs, _ = model.apply(final_params, test_data[:10], training=False)
        
    print(f"Test inference shape: {test_outputs.shape}")
    print(f"Sample outputs: {test_outputs[0]}")
    print(f"Sample targets: {test_targets[0]}")
    
    # Calculate test accuracy (for motor control, use MSE)
    test_mse = float(jnp.mean((test_outputs - test_targets[:10]) ** 2))
    print(f"Test MSE: {test_mse:.4f}")
    
    # Energy analysis
    print("\n📈 Energy Analysis:")
    training_energy = profiler.get_energy_mj()
    avg_power = profiler.get_average_power_mw()
    
    print(f"Training energy consumed: {training_energy:.2f}mJ")
    print(f"Average power during training: {avg_power:.1f}mW")
    
    # Generate energy report
    profiler.export_report("results/energy_analysis.json")
    profiler.plot_comparison("results/energy_comparison.png")
    print("\n📊 Energy reports saved to results/")
    
    # Model summary
    print("\n🎯 Model Summary:")
    print(f"Architecture: Liquid Neural Network with adaptive time constants")
    print(f"Parameters: ~{config.input_dim * config.hidden_dim + config.hidden_dim * config.hidden_dim + config.hidden_dim * config.output_dim} (sparse)")
    print(f"Energy Target: {config.energy_budget_mw}mW @ {config.target_fps}Hz")
    print(f"Achieved Energy: {final_energy:.1f}mW")
    print(f"Performance: MSE = {test_mse:.4f}")
    
    savings_vs_dense = 100 * (1 - config.sparsity) - 100  # Rough estimate
    print(f"Estimated savings vs dense: {-savings_vs_dense:.0f}% energy reduction")
    
    print("\n🚀 Ready for MCU deployment!")
    print("Next steps:")
    print("  1. liquid-lln deploy --target esp32s3 --model liquid_robot.pkl")
    print("  2. liquid-lln flash --port /dev/ttyUSB0")
    print("  3. Monitor energy: liquid-lln monitor --device esp32s3")
    
if __name__ == "__main__":
    import os
    os.makedirs("results", exist_ok=True)
    main()
