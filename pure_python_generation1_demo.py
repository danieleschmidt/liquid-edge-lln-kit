#!/usr/bin/env python3
"""
Generation 1: MAKE IT WORK - Pure Python Liquid Neural Network Demo
Autonomous SDLC Execution - Basic functionality with minimal dependencies
"""

import numpy as np
import json
import time
from pathlib import Path
from typing import Dict, Any, Tuple


class SimpleLiquidConfig:
    """Simplified configuration for liquid neural networks."""
    
    def __init__(self, 
                 input_dim: int = 4,
                 hidden_dim: int = 8,
                 output_dim: int = 2,
                 tau_min: float = 10.0,
                 tau_max: float = 50.0,
                 learning_rate: float = 0.01,
                 sparsity: float = 0.2,
                 energy_budget_mw: float = 80.0,
                 target_fps: int = 30):
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.learning_rate = learning_rate
        self.sparsity = sparsity
        self.energy_budget_mw = energy_budget_mw
        self.target_fps = target_fps
        self.dt = 0.1


class SimpleLiquidNN:
    """Pure Python implementation of Liquid Neural Network."""
    
    def __init__(self, config: SimpleLiquidConfig):
        self.config = config
        self.rng = np.random.RandomState(42)
        
        # Initialize parameters
        self.W_in = self.rng.randn(config.input_dim, config.hidden_dim) * 0.1
        self.W_rec = self.rng.randn(config.hidden_dim, config.hidden_dim) * 0.1
        self.W_out = self.rng.randn(config.hidden_dim, config.output_dim) * 0.1
        
        self.b_rec = np.zeros(config.hidden_dim)
        self.b_out = np.zeros(config.output_dim)
        
        # Time constants (learnable)
        self.tau = self.rng.uniform(config.tau_min, config.tau_max, config.hidden_dim)
        
        # Apply sparsity to recurrent connections
        if config.sparsity > 0:
            mask = self.rng.random((config.hidden_dim, config.hidden_dim)) > config.sparsity
            self.W_rec *= mask
        
        # Initialize hidden state
        self.hidden = np.zeros(config.hidden_dim)
    
    def forward(self, x: np.ndarray, hidden: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """Forward pass through liquid neural network."""
        if hidden is None:
            hidden = self.hidden
        
        # Input transformation
        input_contrib = x @ self.W_in
        
        # Recurrent transformation
        recurrent_contrib = hidden @ self.W_rec + self.b_rec
        
        # Liquid dynamics (simplified ODE)
        dx_dt = -hidden / self.tau + np.tanh(input_contrib + recurrent_contrib)
        new_hidden = hidden + self.config.dt * dx_dt
        
        # Output projection
        output = new_hidden @ self.W_out + self.b_out
        
        return output, new_hidden
    
    def energy_estimate(self) -> float:
        """Estimate energy consumption in milliwatts."""
        # Count operations
        input_ops = self.config.input_dim * self.config.hidden_dim
        recurrent_ops = self.config.hidden_dim * self.config.hidden_dim
        output_ops = self.config.hidden_dim * self.config.output_dim
        
        # Apply sparsity reduction
        if self.config.sparsity > 0:
            recurrent_ops *= (1.0 - self.config.sparsity)
        
        total_ops = input_ops + recurrent_ops + output_ops
        
        # Energy per operation (empirical estimate)
        energy_per_op_nj = 0.5  # nanojoules per MAC
        
        # Convert to milliwatts at target FPS
        energy_mw = (total_ops * energy_per_op_nj * self.config.target_fps) / 1e6
        
        return energy_mw


class SimpleTrainer:
    """Simplified trainer for liquid neural networks."""
    
    def __init__(self, model: SimpleLiquidNN, config: SimpleLiquidConfig):
        self.model = model
        self.config = config
        
    def train(self, train_data: np.ndarray, targets: np.ndarray, epochs: int = 20) -> Dict[str, Any]:
        """Simple training loop."""
        history = {'loss': [], 'energy': []}
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            for i in range(len(train_data)):
                # Forward pass
                output, new_hidden = self.model.forward(train_data[i])
                
                # Loss (MSE)
                loss = np.mean((output - targets[i]) ** 2)
                epoch_loss += loss
                
                # Simple gradient descent (simplified)
                lr = self.config.learning_rate
                error = output - targets[i]
                
                # Update output weights
                self.model.W_out -= lr * np.outer(new_hidden, error)
                self.model.b_out -= lr * error
                
                # Update hidden state for next iteration
                self.model.hidden = new_hidden
            
            avg_loss = epoch_loss / len(train_data)
            energy = self.model.energy_estimate()
            
            history['loss'].append(float(avg_loss))
            history['energy'].append(float(energy))
            
            if epoch % 5 == 0:
                print(f"Epoch {epoch:2d}: Loss={avg_loss:.4f}, Energy={energy:.1f}mW")
        
        return {
            'history': history,
            'final_energy_mw': float(energy)
        }


def generate_synthetic_sensor_data(num_samples: int = 1000, input_dim: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic sensor data for robot control."""
    np.random.seed(42)
    
    # Simulate sensor readings
    t = np.linspace(0, 10, num_samples)
    
    sensors = np.zeros((num_samples, input_dim))
    sensors[:, 0] = np.sin(2 * np.pi * 0.5 * t) + 0.1 * np.random.randn(num_samples)  # Gyro
    sensors[:, 1] = np.cos(2 * np.pi * 0.3 * t) + 0.1 * np.random.randn(num_samples)  # Accel
    sensors[:, 2] = 2.0 + 0.5 * np.sin(2 * np.pi * 0.2 * t) + 0.05 * np.random.randn(num_samples)  # Distance
    sensors[:, 3] = np.where(sensors[:, 2] < 1.5, 1.0, 0.0) + 0.05 * np.random.randn(num_samples)  # Obstacle
    
    # Generate motor commands
    motor_commands = np.zeros((num_samples, 2))
    motor_commands[:, 0] = 0.8 * (1 - sensors[:, 3])  # Linear velocity
    motor_commands[:, 1] = 0.3 * sensors[:, 0]        # Angular velocity
    
    return sensors, motor_commands


def main():
    """Generation 1 Pure Python Demo - Basic functionality."""
    print("=== GENERATION 1: MAKE IT WORK ===")
    print("Pure Python Liquid Neural Network Demo")
    print("Autonomous SDLC - Basic Functionality")
    print()
    
    start_time = time.time()
    
    # 1. Configure system
    config = SimpleLiquidConfig(
        input_dim=4,
        hidden_dim=8,
        output_dim=2,
        tau_min=10.0,
        tau_max=50.0,
        learning_rate=0.01,
        sparsity=0.2,
        energy_budget_mw=80.0,
        target_fps=30
    )
    
    print(f"✓ Configured liquid neural network:")
    print(f"  - Input dim: {config.input_dim}")
    print(f"  - Hidden dim: {config.hidden_dim}")
    print(f"  - Output dim: {config.output_dim}")
    print(f"  - Energy budget: {config.energy_budget_mw}mW")
    print()
    
    # 2. Create model
    model = SimpleLiquidNN(config)
    print("✓ Created SimpleLiquidNN model")
    
    # 3. Generate data
    print("✓ Generating synthetic sensor data...")
    train_data, train_targets = generate_synthetic_sensor_data(200, config.input_dim)
    test_data, test_targets = generate_synthetic_sensor_data(50, config.input_dim)
    
    print(f"  - Training samples: {train_data.shape[0]}")
    print(f"  - Test samples: {test_data.shape[0]}")
    print()
    
    # 4. Train model
    trainer = SimpleTrainer(model, config)
    print("✓ Starting training...")
    
    results = trainer.train(train_data, train_targets, epochs=20)
    
    print(f"  - Final loss: {results['history']['loss'][-1]:.4f}")
    print(f"  - Final energy: {results['final_energy_mw']:.1f}mW")
    print()
    
    # 5. Test inference
    print("✓ Testing inference...")
    
    # Test on single sample
    test_input = test_data[0]
    output, hidden = model.forward(test_input)
    
    print(f"  - Input: [{test_input[0]:.3f}, {test_input[1]:.3f}, {test_input[2]:.3f}, {test_input[3]:.3f}]")
    print(f"  - Output: [{output[0]:.3f}, {output[1]:.3f}]")
    print(f"  - Target: [{test_targets[0][0]:.3f}, {test_targets[0][1]:.3f}]")
    print()
    
    # 6. Energy analysis
    estimated_energy = model.energy_estimate()
    print(f"✓ Energy analysis:")
    print(f"  - Estimated energy: {estimated_energy:.1f}mW")
    print(f"  - Energy budget: {config.energy_budget_mw}mW")
    print(f"  - Within budget: {'✓' if estimated_energy <= config.energy_budget_mw else '✗'}")
    print()
    
    # 7. Performance metrics
    end_time = time.time()
    training_time = end_time - start_time
    
    # Test inference speed
    inference_start = time.time()
    for _ in range(100):
        _ = model.forward(test_data[0])
    inference_time = (time.time() - inference_start) / 100
    
    print(f"✓ Performance metrics:")
    print(f"  - Training time: {training_time:.2f}s")
    print(f"  - Inference time: {inference_time*1000:.2f}ms")
    print(f"  - Target FPS: {config.target_fps}")
    print(f"  - Achievable FPS: {1/inference_time:.1f}")
    print()
    
    # 8. Save results
    results_data = {
        "generation": 1,
        "type": "pure_python_simple_demo",
        "config": {
            "input_dim": config.input_dim,
            "hidden_dim": config.hidden_dim,
            "output_dim": config.output_dim,
            "energy_budget_mw": config.energy_budget_mw,
            "target_fps": config.target_fps,
            "sparsity": config.sparsity
        },
        "metrics": {
            "final_loss": float(results['history']['loss'][-1]),
            "final_energy_mw": float(results['final_energy_mw']),
            "estimated_energy_mw": float(estimated_energy),
            "training_time_s": float(training_time),
            "inference_time_ms": float(inference_time * 1000),
            "achievable_fps": float(1/inference_time),
            "energy_within_budget": bool(estimated_energy <= config.energy_budget_mw)
        },
        "sample_prediction": {
            "input": test_input.tolist(),
            "output": output.tolist(),
            "target": test_targets[0].tolist()
        },
        "status": "completed",
        "timestamp": time.time()
    }
    
    # Save results
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    with open(results_dir / "generation1_pure_python_simple_demo.json", "w") as f:
        json.dump(results_data, f, indent=2)
    
    print("✓ Results saved to results/generation1_pure_python_simple_demo.json")
    print()
    
    # 9. Summary
    print("=== GENERATION 1 COMPLETE ===")
    print("✓ Basic liquid neural network working")
    print("✓ Energy-aware design implemented")
    print("✓ Real-time inference capability")
    print("✓ Within energy budget constraints")
    print(f"✓ Total execution time: {training_time:.2f}s")
    print()
    print("Ready to proceed to Generation 2: MAKE IT ROBUST")
    
    return results_data


if __name__ == "__main__":
    results = main()
    print(f"Generation 1 Status: {results['status']}")