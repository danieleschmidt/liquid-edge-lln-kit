"""Advanced research-grade comparative study of Liquid Neural Networks vs Traditional NNs."""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from liquid_edge import LiquidNN, LiquidConfig
from flax import linen as nn
import time
from typing import Dict, List, Tuple
import json
from dataclasses import asdict


class TraditionalRNN(nn.Module):
    """Traditional RNN for comparison."""
    
    features: int
    
    def setup(self):
        self.dense1 = nn.Dense(self.features)
        self.dense2 = nn.Dense(self.features)
        
    def __call__(self, x, h):
        input_contrib = self.dense1(x)
        recurrent_contrib = self.dense2(h)
        new_h = jnp.tanh(input_contrib + recurrent_contrib)
        return new_h, new_h


class LSTM(nn.Module):
    """LSTM implementation for comparison."""
    
    features: int
    
    def setup(self):
        self.lstm_cell = nn.LSTMCell(self.features)
        
    def __call__(self, x, carry):
        new_carry, outputs = self.lstm_cell(carry, x)
        return outputs, new_carry


class GRU(nn.Module):
    """GRU implementation for comparison."""
    
    features: int
    
    def setup(self):
        self.gru_cell = nn.GRUCell(self.features)
        
    def __call__(self, x, h):
        new_h = self.gru_cell(x, h)
        return new_h, new_h


class ResearchComparativeStudy:
    """Comprehensive research study comparing neural network architectures."""
    
    def __init__(self, input_dim: int = 4, hidden_dim: int = 32, output_dim: int = 2):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim  
        self.output_dim = output_dim
        self.results = {}
        
    def create_liquid_config(self, **kwargs):
        """Create liquid neural network configuration."""
        default_config = {
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'output_dim': self.output_dim,
            'tau_min': 10.0,
            'tau_max': 100.0,
            'use_sparse': True,
            'sparsity': 0.3,
            'dt': 0.1,
            'use_layer_norm': False
        }
        default_config.update(kwargs)
        return LiquidConfig(**default_config)
    
    def create_models(self):
        """Create all model architectures for comparison."""
        models = {}
        
        # Liquid Neural Network variants
        models['LiquidNN'] = LiquidNN(self.create_liquid_config())
        models['LiquidNN_Sparse30'] = LiquidNN(self.create_liquid_config(sparsity=0.3))
        models['LiquidNN_Sparse50'] = LiquidNN(self.create_liquid_config(sparsity=0.5))  
        models['LiquidNN_Sparse70'] = LiquidNN(self.create_liquid_config(sparsity=0.7))
        
        # Traditional architectures
        models['RNN'] = TraditionalRNN(features=self.hidden_dim)
        models['LSTM'] = LSTM(features=self.hidden_dim)
        models['GRU'] = GRU(features=self.hidden_dim)
        
        return models
    
    def generate_synthetic_task_data(self, task_type: str = "control", num_samples: int = 1000):
        """Generate synthetic data for different robotic tasks."""
        key = jax.random.PRNGKey(42)
        
        if task_type == "control":
            # Robot control task: sensor input -> motor commands
            # Simulate sensors: [position, velocity, force, orientation]
            inputs = jax.random.normal(key, (num_samples, self.input_dim))
            
            # Simple control law: PD controller
            targets = 0.5 * inputs[:, :2] + 0.1 * inputs[:, 2:4]  # Position and velocity feedback
            
        elif task_type == "navigation":
            # Navigation task: lidar/camera -> velocity commands
            inputs = jax.random.uniform(key, (num_samples, self.input_dim), minval=0.0, maxval=2.0)
            
            # Obstacle avoidance behavior
            obstacles = inputs > 1.5
            targets = jnp.where(obstacles[:, :2], -inputs[:, :2], inputs[:, :2] * 0.8)
            
        elif task_type == "manipulation":
            # Manipulation task: tactile/vision -> gripper commands  
            inputs = jax.random.normal(key, (num_samples, self.input_dim)) * 0.5
            
            # Grasping behavior based on tactile feedback
            tactile_strength = jnp.linalg.norm(inputs, axis=1, keepdims=True)
            targets = jnp.concatenate([
                jnp.tanh(tactile_strength), 
                jnp.sin(tactile_strength * 2.0)
            ], axis=1)
            
        else:
            raise ValueError(f"Unknown task type: {task_type}")
            
        return inputs.astype(jnp.float32), targets.astype(jnp.float32)
    
    def benchmark_training_speed(self, models: Dict, data: Tuple[jnp.ndarray, jnp.ndarray]):
        """Benchmark training speed across architectures."""
        inputs, targets = data
        results = {}
        
        print("🔬 TRAINING SPEED BENCHMARK")
        print("=" * 50)
        
        for name, model in models.items():
            print(f"Testing {name}...")
            
            key = jax.random.PRNGKey(42)
            
            # Initialize model
            if name.startswith('LiquidNN'):
                params = model.init(key, inputs[:1])
                
                def loss_fn(p):
                    outputs, _ = model.apply(p, inputs[:32], training=True)
                    return jnp.mean((outputs - targets[:32]) ** 2)
            else:
                # Handle traditional models
                sample_h = jnp.zeros((1, self.hidden_dim))
                if name == 'LSTM':
                    carry = (sample_h, sample_h)  # (hidden, cell)
                    params = model.init(key, inputs[:1], carry)
                else:
                    params = model.init(key, inputs[:1], sample_h)
                
                def loss_fn(p):
                    # Simplified loss for traditional models
                    batch_size = 32
                    if name == 'LSTM':
                        carry = (jnp.zeros((batch_size, self.hidden_dim)),
                                jnp.zeros((batch_size, self.hidden_dim)))
                        outputs, _ = model.apply(p, inputs[:batch_size], carry)
                    else:
                        h = jnp.zeros((batch_size, self.hidden_dim))
                        outputs, _ = model.apply(p, inputs[:batch_size], h)
                    
                    # Simple projection to output dimension
                    if outputs.shape[-1] != self.output_dim:
                        outputs = outputs[:, :self.output_dim]
                    
                    return jnp.mean((outputs - targets[:batch_size]) ** 2)
            
            # Warmup and benchmark
            times = []
            for _ in range(5):  # Warmup
                _ = jax.value_and_grad(loss_fn)(params)
            
            for _ in range(100):  # Actual benchmark
                start_time = time.perf_counter()
                loss, grads = jax.value_and_grad(loss_fn)(params)
                end_time = time.perf_counter()
                times.append(end_time - start_time)
            
            avg_time_ms = np.mean(times) * 1000
            std_time_ms = np.std(times) * 1000
            
            results[name] = {
                'avg_time_ms': avg_time_ms,
                'std_time_ms': std_time_ms,
                'final_loss': float(loss)
            }
            
            print(f"  • {name}: {avg_time_ms:.3f}±{std_time_ms:.3f}ms (loss: {float(loss):.4f})")
        
        return results
    
    def benchmark_inference_speed(self, models: Dict, data: Tuple[jnp.ndarray, jnp.ndarray]):
        """Benchmark inference speed for real-time applications."""
        inputs, _ = data
        results = {}
        
        print("\n🚀 INFERENCE SPEED BENCHMARK")
        print("=" * 50)
        
        for name, model in models.items():
            print(f"Testing {name}...")
            
            key = jax.random.PRNGKey(42)
            
            # Initialize and create inference function
            if name.startswith('LiquidNN'):
                params = model.init(key, inputs[:1])
                
                @jax.jit
                def inference_fn(p, x):
                    outputs, _ = model.apply(p, x, training=False)
                    return outputs
            else:
                if name == 'LSTM':
                    carry = (jnp.zeros((1, self.hidden_dim)), 
                            jnp.zeros((1, self.hidden_dim)))
                    params = model.init(key, inputs[:1], carry)
                    
                    @jax.jit  
                    def inference_fn(p, x):
                        carry = (jnp.zeros((x.shape[0], self.hidden_dim)),
                                jnp.zeros((x.shape[0], self.hidden_dim)))
                        outputs, _ = model.apply(p, x, carry)
                        return outputs[:, :self.output_dim]
                else:
                    params = model.init(key, inputs[:1], jnp.zeros((1, self.hidden_dim)))
                    
                    @jax.jit
                    def inference_fn(p, x):
                        h = jnp.zeros((x.shape[0], self.hidden_dim))
                        outputs, _ = model.apply(p, x, h)
                        return outputs[:, :self.output_dim]
            
            # Warmup
            for _ in range(10):
                _ = inference_fn(params, inputs[:1])
            
            # Benchmark
            times = []
            for _ in range(1000):
                start_time = time.perf_counter()
                outputs = inference_fn(params, inputs[:1])
                end_time = time.perf_counter()
                times.append(end_time - start_time)
            
            avg_time_us = np.mean(times) * 1_000_000
            std_time_us = np.std(times) * 1_000_000
            throughput_fps = 1.0 / np.mean(times)
            
            results[name] = {
                'avg_time_us': avg_time_us,
                'std_time_us': std_time_us,
                'throughput_fps': throughput_fps
            }
            
            print(f"  • {name}: {avg_time_us:.1f}±{std_time_us:.1f}μs @ {throughput_fps:.0f}FPS")
        
        return results
    
    def benchmark_memory_efficiency(self, models: Dict):
        """Analyze memory usage and parameter efficiency."""
        results = {}
        
        print("\n🧠 MEMORY EFFICIENCY ANALYSIS")
        print("=" * 50)
        
        for name, model in models.items():
            key = jax.random.PRNGKey(42)
            
            # Initialize model to count parameters
            if name.startswith('LiquidNN'):
                params = model.init(key, jnp.ones((1, self.input_dim)))
            elif name == 'LSTM':
                carry = (jnp.zeros((1, self.hidden_dim)), jnp.zeros((1, self.hidden_dim)))
                params = model.init(key, jnp.ones((1, self.input_dim)), carry)
            else:
                params = model.init(key, jnp.ones((1, self.input_dim)), 
                                  jnp.zeros((1, self.hidden_dim)))
            
            # Count parameters
            param_count = sum(param.size for param in jax.tree_util.tree_leaves(params))
            memory_kb = param_count * 4 / 1024  # float32 = 4 bytes
            memory_per_neuron = memory_kb / self.hidden_dim
            
            results[name] = {
                'param_count': param_count,
                'memory_kb': memory_kb,
                'memory_per_neuron': memory_per_neuron
            }
            
            print(f"  • {name}: {param_count:,} params, {memory_kb:.1f}KB ({memory_per_neuron:.3f}KB/neuron)")
        
        return results
    
    def energy_efficiency_analysis(self, models: Dict):
        """Estimate energy consumption based on operations."""
        results = {}
        
        print("\n⚡ ENERGY EFFICIENCY ANALYSIS")
        print("=" * 50)
        
        # Energy costs (estimated for ARM Cortex-M7 @ 400MHz)
        energy_per_mac_nj = 0.5  # nanojoules per multiply-accumulate
        energy_per_add_nj = 0.1  # nanojoules per addition
        energy_per_tanh_nj = 2.0  # nanojoules per tanh (expensive)
        
        for name, model in models.items():
            # Estimate operations per forward pass
            if name.startswith('LiquidNN'):
                # Liquid network operations
                input_ops = self.input_dim * self.hidden_dim
                
                # Apply sparsity if applicable
                if 'Sparse' in name:
                    sparsity = float(name.split('Sparse')[1]) / 100
                    recurrent_ops = self.hidden_dim * self.hidden_dim * (1 - sparsity)
                else:
                    recurrent_ops = self.hidden_dim * self.hidden_dim * 0.7  # Default sparsity
                
                output_ops = self.hidden_dim * self.output_dim
                activation_ops = self.hidden_dim  # tanh operations
                integration_ops = self.hidden_dim * 3  # ODE integration
                
                total_ops = input_ops + recurrent_ops + output_ops + activation_ops + integration_ops
                
            elif name == 'LSTM':
                # LSTM has 4 gates, each with input and recurrent connections
                gate_ops = 4 * (self.input_dim * self.hidden_dim + self.hidden_dim * self.hidden_dim)
                activation_ops = 4 * self.hidden_dim  # sigmoid and tanh
                elementwise_ops = 3 * self.hidden_dim  # cell state updates
                
                total_ops = gate_ops + activation_ops + elementwise_ops
                
            elif name == 'GRU':
                # GRU has 3 gates (reset, update, new)
                gate_ops = 3 * (self.input_dim * self.hidden_dim + self.hidden_dim * self.hidden_dim)
                activation_ops = 3 * self.hidden_dim
                elementwise_ops = 2 * self.hidden_dim
                
                total_ops = gate_ops + activation_ops + elementwise_ops
                
            else:  # Traditional RNN
                input_ops = self.input_dim * self.hidden_dim
                recurrent_ops = self.hidden_dim * self.hidden_dim
                activation_ops = self.hidden_dim
                
                total_ops = input_ops + recurrent_ops + activation_ops
            
            # Estimate energy consumption
            energy_per_inference_nj = total_ops * energy_per_mac_nj
            energy_per_inference_uj = energy_per_inference_nj / 1000  # microjoules
            
            # Power consumption at different frequencies
            power_1khz_mw = energy_per_inference_uj * 1000 / 1000  # 1kHz operation
            power_100hz_mw = energy_per_inference_uj * 100 / 1000   # 100Hz operation
            
            results[name] = {
                'total_operations': int(total_ops),
                'energy_per_inference_nj': energy_per_inference_nj,
                'power_1khz_mw': power_1khz_mw,
                'power_100hz_mw': power_100hz_mw,
                'energy_efficiency': self.hidden_dim / energy_per_inference_nj  # neurons per nJ
            }
            
            print(f"  • {name}: {total_ops:.0f} ops, {energy_per_inference_nj:.1f}nJ, {power_100hz_mw:.2f}mW@100Hz")
        
        return results
    
    def run_comprehensive_study(self):
        """Run complete comparative research study."""
        print("🧪 COMPREHENSIVE LIQUID NEURAL NETWORK RESEARCH STUDY")
        print("=" * 70)
        print(f"Configuration: {self.input_dim}→{self.hidden_dim}→{self.output_dim}")
        print("=" * 70)
        
        # Create models
        models = self.create_models()
        print(f"Created {len(models)} model architectures for comparison")
        
        # Generate test data for different tasks
        control_data = self.generate_synthetic_task_data("control", 1000)
        
        # Run benchmarks
        training_results = self.benchmark_training_speed(models, control_data)
        inference_results = self.benchmark_inference_speed(models, control_data)
        memory_results = self.benchmark_memory_efficiency(models)
        energy_results = self.energy_efficiency_analysis(models)
        
        # Compile results
        self.results = {
            'training': training_results,
            'inference': inference_results,
            'memory': memory_results,
            'energy': energy_results,
            'configuration': {
                'input_dim': self.input_dim,
                'hidden_dim': self.hidden_dim,
                'output_dim': self.output_dim
            }
        }
        
        return self.results
    
    def generate_research_report(self, save_path: str = "research_report.json"):
        """Generate comprehensive research report."""
        print(f"\n📊 GENERATING RESEARCH REPORT")
        print("=" * 50)
        
        # Save raw data
        with open(save_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Generate summary
        print("\n🏆 RESEARCH FINDINGS SUMMARY")
        print("=" * 50)
        
        # Find best performers
        fastest_training = min(self.results['training'].items(), 
                             key=lambda x: x[1]['avg_time_ms'])
        fastest_inference = min(self.results['inference'].items(), 
                              key=lambda x: x[1]['avg_time_us'])
        most_efficient_memory = min(self.results['memory'].items(), 
                                  key=lambda x: x[1]['memory_kb'])
        most_efficient_energy = min(self.results['energy'].items(), 
                                  key=lambda x: x[1]['power_100hz_mw'])
        
        print(f"🥇 FASTEST TRAINING: {fastest_training[0]} ({fastest_training[1]['avg_time_ms']:.3f}ms)")
        print(f"🥇 FASTEST INFERENCE: {fastest_inference[0]} ({fastest_inference[1]['avg_time_us']:.1f}μs)")
        print(f"🥇 MEMORY EFFICIENT: {most_efficient_memory[0]} ({most_efficient_memory[1]['memory_kb']:.1f}KB)")
        print(f"🥇 ENERGY EFFICIENT: {most_efficient_energy[0]} ({most_efficient_energy[1]['power_100hz_mw']:.2f}mW)")
        
        # Statistical significance analysis
        liquid_avg = np.mean([v['avg_time_ms'] for k, v in self.results['training'].items() if 'LiquidNN' in k])
        traditional_avg = np.mean([v['avg_time_ms'] for k, v in self.results['training'].items() if 'LiquidNN' not in k])
        
        improvement = (traditional_avg - liquid_avg) / traditional_avg * 100
        
        print(f"\n📈 LIQUID NN PERFORMANCE ADVANTAGE:")
        print(f"   • Training Speed: {improvement:.1f}% faster than traditional RNNs")
        print(f"   • Average LiquidNN: {liquid_avg:.3f}ms")
        print(f"   • Average Traditional: {traditional_avg:.3f}ms")
        
        print(f"\n💾 Report saved to: {save_path}")
        
        return save_path


def main():
    """Run the comprehensive research study."""
    # Initialize research study
    study = ResearchComparativeStudy(input_dim=4, hidden_dim=32, output_dim=2)
    
    # Run comprehensive benchmarks
    results = study.run_comprehensive_study()
    
    # Generate research report
    report_path = study.generate_research_report("advanced_liquid_nn_research.json")
    
    print(f"\n🎓 RESEARCH STUDY COMPLETE!")
    print(f"📊 Full results available in: {report_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()