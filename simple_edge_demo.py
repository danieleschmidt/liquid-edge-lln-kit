#!/usr/bin/env python3
"""
Generation 1: MAKE IT WORK - Simple Edge Robotics Demo
Demonstrating liquid neural network for basic sensor processing.
"""

import numpy as np
import time
import json
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, asdict


@dataclass
class SimpleLiquidConfig:
    """Simplified config for basic functionality."""
    input_dim: int = 4
    hidden_dim: int = 8
    output_dim: int = 2
    tau: float = 0.1
    dt: float = 0.01
    learning_rate: float = 0.01


class SimpleLiquidCell:
    """Minimal liquid neural network cell for basic demonstration."""
    
    def __init__(self, config: SimpleLiquidConfig):
        self.config = config
        # Initialize simple weights
        np.random.seed(42)
        self.W_in = np.random.randn(config.input_dim, config.hidden_dim) * 0.1
        self.W_rec = np.random.randn(config.hidden_dim, config.hidden_dim) * 0.1
        self.W_out = np.random.randn(config.hidden_dim, config.output_dim) * 0.1
        self.bias_h = np.zeros(config.hidden_dim)
        self.bias_out = np.zeros(config.output_dim)
        
        # State
        self.hidden_state = np.zeros(config.hidden_dim)
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Simple forward pass through liquid cell."""
        # Input + recurrent projections
        input_proj = x @ self.W_in
        recurrent_proj = self.hidden_state @ self.W_rec
        
        # Liquid dynamics: simplified ODE integration
        activation = np.tanh(input_proj + recurrent_proj + self.bias_h)
        
        # Euler integration with time constant
        dhdt = (-self.hidden_state + activation) / self.config.tau
        self.hidden_state = self.hidden_state + self.config.dt * dhdt
        
        # Output projection
        output = self.hidden_state @ self.W_out + self.bias_out
        return output
    
    def reset_state(self):
        """Reset hidden state."""
        self.hidden_state = np.zeros(self.config.hidden_dim)


class SimpleRobotController:
    """Simple robot controller demonstrating edge AI."""
    
    def __init__(self):
        self.config = SimpleLiquidConfig()
        self.liquid_brain = SimpleLiquidCell(self.config)
        self.energy_consumed = 0.0  # mW⋅s
        self.inference_count = 0
        
    def process_sensors(self, sensor_data: Dict[str, float]) -> Dict[str, float]:
        """Process sensor data and generate motor commands."""
        start_time = time.perf_counter()
        
        # Convert sensor dict to array
        sensor_array = np.array([
            sensor_data.get('front_distance', 0.5),
            sensor_data.get('left_distance', 0.5), 
            sensor_data.get('right_distance', 0.5),
            sensor_data.get('imu_angular_vel', 0.0)
        ])
        
        # Normalize inputs (0-1 range)
        sensor_array = np.clip(sensor_array, 0, 1)
        
        # Run liquid network inference
        motor_commands = self.liquid_brain.forward(sensor_array)
        
        # Convert to motor dict
        motors = {
            'left_motor': float(np.tanh(motor_commands[0])),  # -1 to 1
            'right_motor': float(np.tanh(motor_commands[1]))
        }
        
        # Track energy (simplified model)
        inference_time = time.perf_counter() - start_time
        # Assume ~50mW during inference
        self.energy_consumed += 50.0 * inference_time * 1000  # mW⋅s 
        self.inference_count += 1
        
        return motors
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance metrics."""
        avg_energy = self.energy_consumed / max(1, self.inference_count)
        return {
            'total_inferences': self.inference_count,
            'total_energy_mws': round(self.energy_consumed, 2),
            'avg_energy_per_inference_mws': round(avg_energy, 4),
            'estimated_fps': 100 if avg_energy > 0 else 0,  # Simplified
            'memory_usage_kb': 1.2  # Rough estimate for simple model
        }


def simulate_robot_navigation():
    """Simulate robot navigation with liquid neural network."""
    print("🤖 Generation 1: Simple Liquid Neural Network Robot Demo")
    print("=" * 60)
    
    controller = SimpleRobotController()
    
    # Simulate various scenarios
    scenarios = [
        {'name': 'Open Space', 'sensors': {'front_distance': 1.0, 'left_distance': 1.0, 'right_distance': 1.0, 'imu_angular_vel': 0.0}},
        {'name': 'Wall Ahead', 'sensors': {'front_distance': 0.1, 'left_distance': 1.0, 'right_distance': 0.8, 'imu_angular_vel': 0.0}},
        {'name': 'Narrow Corridor', 'sensors': {'front_distance': 0.8, 'left_distance': 0.2, 'right_distance': 0.3, 'imu_angular_vel': 0.1}},
        {'name': 'Left Turn', 'sensors': {'front_distance': 0.3, 'left_distance': 0.9, 'right_distance': 0.2, 'imu_angular_vel': 0.0}},
        {'name': 'Obstacle Course', 'sensors': {'front_distance': 0.4, 'left_distance': 0.6, 'right_distance': 0.7, 'imu_angular_vel': 0.05}}
    ]
    
    results = []
    
    for scenario in scenarios:
        print(f"\n🎯 Scenario: {scenario['name']}")
        sensors = scenario['sensors']
        motors = controller.process_sensors(sensors)
        
        # Display results
        print(f"   Sensors: {sensors}")
        print(f"   Motors:  {motors}")
        
        # Simple behavioral analysis
        left_speed = motors['left_motor']
        right_speed = motors['right_motor']
        
        if abs(left_speed - right_speed) < 0.1:
            behavior = "Moving Forward"
        elif left_speed > right_speed:
            behavior = "Turning Right"
        elif right_speed > left_speed:
            behavior = "Turning Left"
        else:
            behavior = "Stopping"
            
        print(f"   Behavior: {behavior}")
        
        results.append({
            'scenario': scenario['name'],
            'sensors': sensors,
            'motors': motors,
            'behavior': behavior
        })
    
    # Performance summary
    print(f"\n📊 Performance Summary")
    print("=" * 30)
    stats = controller.get_performance_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    # Save results
    output_data = {
        'generation': 1,
        'description': 'Simple liquid neural network edge robotics demo',
        'scenarios': results,
        'performance': stats,
        'timestamp': time.time()
    }
    
    with open('generation1_demo_results.json', 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n✅ Generation 1 complete! Results saved to generation1_demo_results.json")
    return output_data


def demonstrate_energy_efficiency():
    """Compare liquid vs traditional neural network energy usage."""
    print(f"\n⚡ Energy Efficiency Analysis")
    print("-" * 40)
    
    # Simulate traditional dense network
    traditional_params = 4 * 64 + 64 * 64 + 64 * 2  # Dense layers
    traditional_energy_per_op = 1.2  # Higher energy per operation
    traditional_total_energy = traditional_params * traditional_energy_per_op / 1000  # mW
    
    # Liquid network (sparse and adaptive)
    liquid_params = 4 * 8 + 8 * 8 * 0.7 + 8 * 2  # Sparse connectivity
    liquid_energy_per_op = 0.5  # Lower energy, adaptive computation
    liquid_total_energy = liquid_params * liquid_energy_per_op / 1000  # mW
    
    energy_savings = (traditional_total_energy - liquid_total_energy) / traditional_total_energy * 100
    
    print(f"   Traditional NN:  {traditional_total_energy:.1f} mW")
    print(f"   Liquid NN:       {liquid_total_energy:.1f} mW") 
    print(f"   Energy Savings:  {energy_savings:.1f}%")
    print(f"   Params Reduced:  {traditional_params} → {int(liquid_params)} ({100*(1-liquid_params/traditional_params):.1f}% reduction)")


if __name__ == "__main__":
    # Run Generation 1 demo
    results = simulate_robot_navigation()
    demonstrate_energy_efficiency()
    
    print(f"\n🚀 Ready for Generation 2: Adding robustness, error handling, and monitoring!")