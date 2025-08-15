"""Autonomous Evolution System for Self-Improving Liquid Neural Networks.

This module implements breakthrough autonomous evolution capabilities that allow
liquid neural networks to continuously improve themselves through:
- Self-modifying architecture adaptation
- Autonomous hyperparameter optimization  
- Dynamic sparsity evolution
- Energy-aware morphological changes
- Experience-driven neural plasticity
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from dataclasses import dataclass, field
from functools import partial
import time


@dataclass 
class EvolutionConfig:
    """Configuration for autonomous evolution system."""
    
    # Evolution parameters
    mutation_rate: float = 0.05
    selection_pressure: float = 0.3
    adaptation_speed: float = 0.01
    population_size: int = 8
    
    # Architecture evolution
    max_hidden_dim: int = 128
    min_hidden_dim: int = 8
    architecture_mutation_prob: float = 0.1
    
    # Performance thresholds  
    energy_efficiency_target: float = 0.8
    accuracy_threshold: float = 0.90
    convergence_patience: int = 50
    
    # Self-modification limits
    max_mutations_per_cycle: int = 3
    stability_check_window: int = 100


class EvolutionaryMemory:
    """Memory system for tracking evolution history and learning."""
    
    def __init__(self, memory_size: int = 1000):
        self.memory_size = memory_size
        self.experiences = []
        self.performance_history = []
        self.successful_mutations = []
        self.failed_mutations = []
        
    def store_experience(self, 
                        architecture_config: Dict,
                        performance_metrics: Dict,
                        mutation_info: Optional[Dict] = None):
        """Store evolutionary experience for learning."""
        experience = {
            'timestamp': time.time(),
            'architecture': architecture_config.copy(),
            'performance': performance_metrics.copy(),
            'mutation': mutation_info
        }
        
        self.experiences.append(experience)
        
        # Maintain memory size limit
        if len(self.experiences) > self.memory_size:
            self.experiences.pop(0)
            
        # Track performance trends
        self.performance_history.append(performance_metrics['overall_score'])
        if len(self.performance_history) > self.memory_size:
            self.performance_history.pop(0)
    
    def get_successful_patterns(self) -> List[Dict]:
        """Extract patterns from successful mutations."""
        if not self.experiences:
            return []
            
        # Find top performing configurations
        sorted_experiences = sorted(
            self.experiences, 
            key=lambda x: x['performance']['overall_score'],
            reverse=True
        )
        
        top_20_percent = max(1, len(sorted_experiences) // 5)
        return sorted_experiences[:top_20_percent]
    
    def predict_mutation_success(self, proposed_mutation: Dict) -> float:
        """Predict likelihood of mutation success based on history."""
        if not self.successful_mutations:
            return 0.5  # Neutral prediction
            
        # Simple pattern matching for mutation types
        mutation_type = proposed_mutation.get('type', 'unknown')
        
        similar_mutations = [
            m for m in self.successful_mutations 
            if m.get('type') == mutation_type
        ]
        
        if similar_mutations:
            success_rate = len(similar_mutations) / (
                len(similar_mutations) + len([
                    m for m in self.failed_mutations 
                    if m.get('type') == mutation_type
                ])
            )
            return success_rate
        
        return 0.5


class ArchitectureMutator:
    """Handles autonomous architecture mutations."""
    
    def __init__(self, config: EvolutionConfig):
        self.config = config
        self.mutation_strategies = [
            self._mutate_hidden_dimension,
            self._mutate_sparsity,
            self._mutate_time_constants,
            self._mutate_connectivity,
            self._mutate_activation_functions
        ]
    
    def generate_mutation(self, 
                         current_config: Dict,
                         memory: EvolutionaryMemory) -> Tuple[Dict, Dict]:
        """Generate intelligent mutation based on history."""
        # Select mutation strategy based on memory
        successful_patterns = memory.get_successful_patterns()
        
        if successful_patterns and np.random.random() < 0.7:
            # Bias towards historically successful mutations
            mutation_strategy = self._select_successful_strategy(successful_patterns)
        else:
            # Random exploration
            mutation_strategy = np.random.choice(self.mutation_strategies)
        
        # Apply mutation
        mutated_config, mutation_info = mutation_strategy(current_config)
        
        return mutated_config, mutation_info
    
    def _mutate_hidden_dimension(self, config: Dict) -> Tuple[Dict, Dict]:
        """Mutate the hidden dimension size."""
        current_dim = config.get('hidden_dim', 16)
        
        # Intelligent sizing based on current performance
        if np.random.random() < 0.5:
            # Increase for more capacity
            new_dim = min(self.config.max_hidden_dim, int(current_dim * 1.2))
        else:
            # Decrease for efficiency
            new_dim = max(self.config.min_hidden_dim, int(current_dim * 0.8))
        
        new_config = config.copy()
        new_config['hidden_dim'] = new_dim
        
        mutation_info = {
            'type': 'hidden_dimension',
            'old_value': current_dim,
            'new_value': new_dim,
            'change_ratio': new_dim / current_dim
        }
        
        return new_config, mutation_info
    
    def _mutate_sparsity(self, config: Dict) -> Tuple[Dict, Dict]:
        """Mutate network sparsity for efficiency."""
        current_sparsity = config.get('sparsity', 0.3)
        
        # Adaptive sparsity mutation
        delta = np.random.normal(0, 0.1)
        new_sparsity = np.clip(current_sparsity + delta, 0.0, 0.9)
        
        new_config = config.copy()
        new_config['sparsity'] = new_sparsity
        
        mutation_info = {
            'type': 'sparsity',
            'old_value': current_sparsity,
            'new_value': new_sparsity,
            'delta': delta
        }
        
        return new_config, mutation_info
    
    def _mutate_time_constants(self, config: Dict) -> Tuple[Dict, Dict]:
        """Mutate liquid time constants."""
        tau_min = config.get('tau_min', 10.0)
        tau_max = config.get('tau_max', 100.0)
        
        # Mutate both bounds
        tau_min_new = tau_min * np.random.uniform(0.5, 2.0)
        tau_max_new = tau_max * np.random.uniform(0.5, 2.0)
        
        # Ensure ordering
        if tau_min_new > tau_max_new:
            tau_min_new, tau_max_new = tau_max_new, tau_min_new
        
        new_config = config.copy()
        new_config['tau_min'] = tau_min_new
        new_config['tau_max'] = tau_max_new
        
        mutation_info = {
            'type': 'time_constants',
            'old_tau_min': tau_min,
            'old_tau_max': tau_max,
            'new_tau_min': tau_min_new,
            'new_tau_max': tau_max_new
        }
        
        return new_config, mutation_info
    
    def _mutate_connectivity(self, config: Dict) -> Tuple[Dict, Dict]:
        """Mutate connectivity patterns."""
        # This would modify recurrent connection structure
        connectivity_type = config.get('connectivity_type', 'dense')
        
        connectivity_options = ['dense', 'sparse', 'small_world', 'scale_free']
        new_connectivity = np.random.choice(
            [c for c in connectivity_options if c != connectivity_type]
        )
        
        new_config = config.copy()
        new_config['connectivity_type'] = new_connectivity
        
        mutation_info = {
            'type': 'connectivity',
            'old_value': connectivity_type,
            'new_value': new_connectivity
        }
        
        return new_config, mutation_info
    
    def _mutate_activation_functions(self, config: Dict) -> Tuple[Dict, Dict]:
        """Mutate activation function choices."""
        current_activation = config.get('activation', 'tanh')
        
        activation_options = ['tanh', 'relu', 'gelu', 'swish', 'sigmoid']
        new_activation = np.random.choice(
            [a for a in activation_options if a != current_activation]
        )
        
        new_config = config.copy()
        new_config['activation'] = new_activation
        
        mutation_info = {
            'type': 'activation',
            'old_value': current_activation,
            'new_value': new_activation
        }
        
        return new_config, mutation_info
    
    def _select_successful_strategy(self, successful_patterns: List[Dict]):
        """Select mutation strategy based on successful patterns."""
        # Analyze which mutation types were most successful
        mutation_type_scores = {}
        
        for pattern in successful_patterns:
            if pattern['mutation']:
                mut_type = pattern['mutation'].get('type', 'unknown')
                score = pattern['performance']['overall_score']
                
                if mut_type not in mutation_type_scores:
                    mutation_type_scores[mut_type] = []
                mutation_type_scores[mut_type].append(score)
        
        # Select based on average performance
        if mutation_type_scores:
            best_type = max(
                mutation_type_scores.keys(),
                key=lambda t: np.mean(mutation_type_scores[t])
            )
            
            # Map type to strategy
            type_to_strategy = {
                'hidden_dimension': self._mutate_hidden_dimension,
                'sparsity': self._mutate_sparsity,
                'time_constants': self._mutate_time_constants,
                'connectivity': self._mutate_connectivity,
                'activation': self._mutate_activation_functions
            }
            
            return type_to_strategy.get(best_type, self._mutate_hidden_dimension)
        
        return np.random.choice(self.mutation_strategies)


class PerformanceEvaluator:
    """Evaluates network performance across multiple metrics."""
    
    def __init__(self, config: EvolutionConfig):
        self.config = config
        
    def evaluate_comprehensive(self, 
                             model,
                             test_data: jnp.ndarray,
                             test_targets: jnp.ndarray,
                             energy_budget: float) -> Dict[str, float]:
        """Comprehensive performance evaluation."""
        metrics = {}
        
        # Accuracy evaluation
        predictions, _ = model.apply(model.params, test_data)
        accuracy = self._compute_accuracy(predictions, test_targets)
        metrics['accuracy'] = accuracy
        
        # Energy efficiency
        estimated_energy = model.energy_estimate()
        energy_efficiency = min(1.0, energy_budget / max(estimated_energy, 0.1))
        metrics['energy_efficiency'] = energy_efficiency
        
        # Inference speed (simulated)
        inference_time = self._measure_inference_speed(model, test_data)
        speed_score = max(0.0, 1.0 - inference_time / 100.0)  # Normalize to 100ms baseline
        metrics['speed_score'] = speed_score
        
        # Memory efficiency
        param_count = sum(x.size for x in jax.tree_leaves(model.params))
        memory_efficiency = max(0.0, 1.0 - param_count / 100000)  # Normalize to 100k params
        metrics['memory_efficiency'] = memory_efficiency
        
        # Stability score
        stability = self._evaluate_stability(model, test_data)
        metrics['stability'] = stability
        
        # Overall composite score
        weights = {
            'accuracy': 0.4,
            'energy_efficiency': 0.3,
            'speed_score': 0.15,
            'memory_efficiency': 0.1,
            'stability': 0.05
        }
        
        overall_score = sum(weights[k] * metrics[k] for k in weights)
        metrics['overall_score'] = overall_score
        
        return metrics
    
    def _compute_accuracy(self, predictions: jnp.ndarray, targets: jnp.ndarray) -> float:
        """Compute prediction accuracy."""
        # For regression tasks
        mse = jnp.mean((predictions - targets) ** 2)
        # Convert MSE to accuracy-like metric (higher is better)
        accuracy = 1.0 / (1.0 + mse)
        return float(accuracy)
    
    def _measure_inference_speed(self, model, data: jnp.ndarray) -> float:
        """Measure inference speed in milliseconds."""
        # Simulate timing measurement
        start_time = time.time()
        for _ in range(10):  # Multiple runs for stability
            _ = model.apply(model.params, data[:1])  # Single sample
        end_time = time.time()
        
        avg_time_ms = ((end_time - start_time) / 10) * 1000
        return avg_time_ms
    
    def _evaluate_stability(self, model, data: jnp.ndarray) -> float:
        """Evaluate numerical stability of the model."""
        try:
            # Test with slightly perturbed inputs
            noise_levels = [0.01, 0.05, 0.1]
            stability_scores = []
            
            base_output, _ = model.apply(model.params, data)
            
            for noise_level in noise_levels:
                noise = jax.random.normal(jax.random.PRNGKey(42), data.shape) * noise_level
                noisy_data = data + noise
                noisy_output, _ = model.apply(model.params, noisy_data)
                
                # Measure output sensitivity
                output_diff = jnp.mean(jnp.abs(noisy_output - base_output))
                sensitivity = output_diff / noise_level
                stability_score = 1.0 / (1.0 + sensitivity)
                stability_scores.append(stability_score)
            
            return float(jnp.mean(jnp.array(stability_scores)))
            
        except Exception:
            return 0.0  # Unstable if computation fails


class AutonomousEvolutionEngine:
    """Main autonomous evolution engine for liquid neural networks."""
    
    def __init__(self, 
                 initial_config: Dict,
                 evolution_config: EvolutionConfig):
        self.current_config = initial_config
        self.evolution_config = evolution_config
        self.memory = EvolutionaryMemory()
        self.mutator = ArchitectureMutator(evolution_config)
        self.evaluator = PerformanceEvaluator(evolution_config)
        
        self.generation = 0
        self.best_config = initial_config.copy()
        self.best_performance = 0.0
        self.stagnation_counter = 0
        
    def evolve_step(self, 
                   current_model,
                   test_data: jnp.ndarray,
                   test_targets: jnp.ndarray) -> Dict[str, Any]:
        """Single evolution step with autonomous improvement."""
        self.generation += 1
        
        # Evaluate current performance
        current_performance = self.evaluator.evaluate_comprehensive(
            current_model, test_data, test_targets, 
            self.evolution_config.energy_efficiency_target * 100
        )
        
        # Store experience in memory
        self.memory.store_experience(
            self.current_config,
            current_performance
        )
        
        # Check for improvement
        if current_performance['overall_score'] > self.best_performance:
            self.best_performance = current_performance['overall_score']
            self.best_config = self.current_config.copy()
            self.stagnation_counter = 0
        else:
            self.stagnation_counter += 1
        
        # Generate candidate mutations
        candidate_configs = []
        mutation_infos = []
        
        for _ in range(self.evolution_config.max_mutations_per_cycle):
            mutated_config, mutation_info = self.mutator.generate_mutation(
                self.current_config, self.memory
            )
            
            # Predict mutation success
            success_prob = self.memory.predict_mutation_success(mutation_info)
            
            if success_prob > 0.3:  # Only try promising mutations
                candidate_configs.append(mutated_config)
                mutation_infos.append(mutation_info)
        
        # Select best mutation (would require model creation and evaluation)
        # For now, select the first promising candidate
        if candidate_configs:
            self.current_config = candidate_configs[0]
            selected_mutation = mutation_infos[0]
        else:
            # No promising mutations, stay with current config
            selected_mutation = None
        
        # Adaptive evolution parameters
        self._adapt_evolution_parameters(current_performance)
        
        evolution_report = {
            'generation': self.generation,
            'current_performance': current_performance,
            'best_performance': self.best_performance,
            'stagnation_counter': self.stagnation_counter,
            'selected_mutation': selected_mutation,
            'memory_size': len(self.memory.experiences),
            'evolution_params': {
                'mutation_rate': self.evolution_config.mutation_rate,
                'selection_pressure': self.evolution_config.selection_pressure,
                'adaptation_speed': self.evolution_config.adaptation_speed
            }
        }
        
        return evolution_report
    
    def _adapt_evolution_parameters(self, performance: Dict[str, float]):
        """Adapt evolution parameters based on performance trends."""
        # Increase mutation rate if stagnating
        if self.stagnation_counter > self.evolution_config.convergence_patience // 2:
            self.evolution_config.mutation_rate = min(0.2, self.evolution_config.mutation_rate * 1.1)
        else:
            # Decrease mutation rate if improving
            self.evolution_config.mutation_rate = max(0.01, self.evolution_config.mutation_rate * 0.95)
        
        # Adapt selection pressure
        if performance['overall_score'] > 0.8:
            # High performance, increase pressure for fine-tuning
            self.evolution_config.selection_pressure = min(0.8, self.evolution_config.selection_pressure * 1.05)
        else:
            # Lower performance, decrease pressure for exploration
            self.evolution_config.selection_pressure = max(0.1, self.evolution_config.selection_pressure * 0.95)
    
    def get_evolution_summary(self) -> Dict[str, Any]:
        """Get comprehensive evolution summary."""
        successful_patterns = self.memory.get_successful_patterns()
        
        return {
            'total_generations': self.generation,
            'best_performance': self.best_performance,
            'best_configuration': self.best_config,
            'current_stagnation': self.stagnation_counter,
            'memory_insights': {
                'total_experiences': len(self.memory.experiences),
                'successful_patterns_count': len(successful_patterns),
                'top_mutations': [p['mutation'] for p in successful_patterns[:3] if p['mutation']]
            },
            'evolution_convergence': {
                'is_converged': self.stagnation_counter > self.evolution_config.convergence_patience,
                'convergence_confidence': min(1.0, self.generation / 100),
                'performance_trend': self.memory.performance_history[-10:] if len(self.memory.performance_history) >= 10 else self.memory.performance_history
            }
        }


def create_autonomous_evolution_demo():
    """Create demonstration of autonomous evolution system."""
    
    # Initial configuration
    initial_config = {
        'input_dim': 8,
        'hidden_dim': 16,
        'output_dim': 4,
        'tau_min': 10.0,
        'tau_max': 100.0,
        'sparsity': 0.3,
        'activation': 'tanh',
        'connectivity_type': 'dense'
    }
    
    # Evolution configuration
    evolution_config = EvolutionConfig(
        mutation_rate=0.1,
        selection_pressure=0.4,
        adaptation_speed=0.02,
        max_hidden_dim=64,
        convergence_patience=30
    )
    
    # Create evolution engine
    evolution_engine = AutonomousEvolutionEngine(
        initial_config=initial_config,
        evolution_config=evolution_config
    )
    
    return evolution_engine, initial_config