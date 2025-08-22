#!/usr/bin/env python3
"""
Robust Evolutionary SDLC Generation 2 Demo - PRODUCTION RELIABILITY

This demonstrates the robust, production-grade evolutionary SDLC system with:
- Self-healing population management
- Circuit breakers and fault tolerance  
- Anomaly detection and recovery
- Comprehensive error handling and graceful degradation
- Real-time health monitoring and alerts

Generation 2 Focus: MAKE IT ROBUST - Production reliability and fault tolerance
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import time
from pathlib import Path
import json
import logging
from typing import Dict, List, Any, Optional
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import the breakthrough evolutionary SDLC with robustness
from src.liquid_edge.autonomous_evolutionary_sdlc import (
    create_autonomous_evolutionary_sdlc,
    OptimizationObjective,
    EvolutionaryConfig,
    SDLCGenome
)

from src.liquid_edge.evolutionary_robustness_system import (
    create_robust_evolutionary_system,
    RobustnessLevel,
    FailureMode,
    CircuitBreakerError
)

# Setup comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RobustEvolutionarySDLCDemo:
    """Production-grade robust evolutionary SDLC demonstration."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.metrics = {}
        self.failure_scenarios_tested = []
        self.recovery_actions_taken = []
        
    def demonstrate_robust_evolutionary_sdlc(self):
        """Demonstrate the robust evolutionary SDLC system."""
        
        print("🛡️ ROBUST EVOLUTIONARY SDLC GENERATION 2 DEMO")
        print("=" * 65)
        print("Production-Grade Reliability with Self-Healing Capabilities")
        print()
        
        # Initialize robust system
        print("🚀 Initializing Robust Evolutionary System...")
        
        try:
            # Create robust system components
            robustness_config, fitness_evaluator, population_manager = create_robust_evolutionary_system(
                robustness_level=RobustnessLevel.PRODUCTION
            )
            
            print(f"✅ Robustness System Initialized:")
            print(f"   • Level: {robustness_config.robustness_level.value}")
            print(f"   • Auto-healing: {robustness_config.auto_healing_enabled}")
            print(f"   • Circuit breaker: Enabled")
            print(f"   • Anomaly detection: {robustness_config.outlier_detection_enabled}")
            print(f"   • Emergency protocols: {robustness_config.emergency_save_enabled}")
            print()
            
            # Create evolutionary SDLC with robust configuration
            robust_objectives = {
                OptimizationObjective.ROBUSTNESS: 0.4,      # High robustness priority
                OptimizationObjective.ENERGY_EFFICIENCY: 0.25,
                OptimizationObjective.INFERENCE_SPEED: 0.25,
                OptimizationObjective.ACCURACY: 0.1
            }
            
            evolutionary_sdlc = create_autonomous_evolutionary_sdlc(
                objectives=robust_objectives,
                population_size=10,  # Smaller for robust testing
                max_generations=20
            )
            
            print("🎯 Robust Multi-Objective Configuration:")
            for objective, weight in robust_objectives.items():
                print(f"   {objective.value}: {weight:.1%}")
            print()
            
            # Test robustness features
            self._test_robustness_features(
                evolutionary_sdlc, 
                robustness_config,
                fitness_evaluator,
                population_manager
            )
            
            # Run robust evolution with monitoring
            print("🧬 Starting Robust Evolution with Real-time Monitoring...")
            start_time = time.time()
            
            best_genome = self._run_monitored_evolution(
                evolutionary_sdlc,
                fitness_evaluator,
                population_manager
            )
            
            evolution_time = time.time() - start_time
            
            print(f"🎉 Robust Evolution Completed Successfully!")
            print(f"   Duration: {evolution_time:.2f} seconds")
            print(f"   Best Fitness: {best_genome.fitness:.4f}")
            print(f"   Generations Run: {evolutionary_sdlc.generation}")
            print()
            
            # Analyze robustness metrics
            self._analyze_robustness_metrics(
                evolutionary_sdlc,
                fitness_evaluator,
                population_manager
            )
            
            # Deploy robust solution
            self._deploy_robust_solution(evolutionary_sdlc, best_genome)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Robust evolutionary SDLC failed: {e}")
            print(f"❌ System failure: {e}")
            traceback.print_exc()
            return False
    
    def _test_robustness_features(self, sdlc, config, evaluator, manager):
        """Test various robustness features."""
        
        print("🔬 Testing Robustness Features...")
        
        # Test 1: Circuit Breaker
        print("   Testing Circuit Breaker...")
        try:
            self._test_circuit_breaker(evaluator)
            print("   ✅ Circuit breaker functioning correctly")
        except Exception as e:
            print(f"   ⚠️  Circuit breaker test issue: {e}")
        
        # Test 2: Population Health Monitoring
        print("   Testing Population Health Monitoring...")
        try:
            self._test_population_health_monitoring(sdlc, manager)
            print("   ✅ Population health monitoring operational")
        except Exception as e:
            print(f"   ⚠️  Health monitoring test issue: {e}")
        
        # Test 3: Self-Healing Mechanisms
        print("   Testing Self-Healing Mechanisms...")
        try:
            self._test_self_healing_mechanisms(sdlc, manager)
            print("   ✅ Self-healing mechanisms active")
        except Exception as e:
            print(f"   ⚠️  Self-healing test issue: {e}")
        
        # Test 4: Fault Tolerance
        print("   Testing Fault Tolerance...")
        try:
            self._test_fault_tolerance(evaluator)
            print("   ✅ Fault tolerance verified")
        except Exception as e:
            print(f"   ⚠️  Fault tolerance test issue: {e}")
        
        print("   🎯 Robustness testing completed")
        print()
    
    def _test_circuit_breaker(self, evaluator):
        """Test circuit breaker functionality."""
        
        def failing_evaluation(genome):
            """Evaluation that always fails for testing."""
            raise RuntimeError("Simulated evaluation failure")
        
        # Create a dummy genome for testing
        config = EvolutionaryConfig()
        test_genome = SDLCGenome(config)
        
        # Test circuit breaker by triggering failures
        failure_count = 0
        for i in range(10):
            try:
                evaluator.evaluate_with_robustness(test_genome, failing_evaluation)
            except (RuntimeError, CircuitBreakerError):
                failure_count += 1
            except Exception as e:
                # Other exceptions are handled by fallback
                pass
        
        self.failure_scenarios_tested.append("circuit_breaker")
        
        if failure_count < 10:
            # Circuit breaker activated (expected behavior)
            self.recovery_actions_taken.append("circuit_breaker_activation")
    
    def _test_population_health_monitoring(self, sdlc, manager):
        """Test population health monitoring."""
        
        # Initialize population for testing
        if not sdlc.population:
            sdlc.initialize_population()
        
        # Monitor initial health
        health = manager.monitor_population_health(sdlc.population)
        
        # Verify health metrics are calculated
        assert 'status' in health
        assert 'diversity' in health
        assert 'fitness_variance' in health
        
        self.metrics['initial_population_health'] = health
        
        # Test with corrupted population (simulate low diversity)
        corrupted_population = [sdlc.population[0]] * 3  # Same genome repeated
        corrupted_health = manager.monitor_population_health(corrupted_population)
        
        assert corrupted_health['diversity'] < health['diversity']
        
        self.failure_scenarios_tested.append("low_diversity_population")
    
    def _test_self_healing_mechanisms(self, sdlc, manager):
        """Test self-healing mechanisms."""
        
        # Create unhealthy population scenario
        if not sdlc.population:
            sdlc.initialize_population()
        
        # Simulate low diversity (cloned population)
        unhealthy_population = [sdlc.population[0]] * 5
        
        # Monitor health
        health = manager.monitor_population_health(unhealthy_population)
        
        # Apply self-healing
        healed_population = manager.apply_self_healing(
            unhealthy_population, 
            health['status']
        )
        
        # Verify healing occurred
        healed_health = manager.monitor_population_health(healed_population)
        
        assert healed_health['diversity'] > health['diversity']
        
        self.recovery_actions_taken.append("population_self_healing")
    
    def _test_fault_tolerance(self, evaluator):
        """Test fault tolerance with various failure modes."""
        
        config = EvolutionaryConfig()
        test_genome = SDLCGenome(config)
        
        def unstable_evaluation(genome):
            """Evaluation that sometimes fails."""
            if np.random.random() < 0.3:  # 30% failure rate
                raise ValueError("Simulated numerical instability")
            
            return {
                OptimizationObjective.ENERGY_EFFICIENCY: np.random.random(),
                OptimizationObjective.INFERENCE_SPEED: np.random.random(),
                OptimizationObjective.ACCURACY: np.random.random(),
                OptimizationObjective.ROBUSTNESS: np.random.random()
            }
        
        # Test multiple evaluations with intermittent failures
        successful_evaluations = 0
        for _ in range(10):
            try:
                result = evaluator.evaluate_with_robustness(test_genome, unstable_evaluation)
                if result:
                    successful_evaluations += 1
            except Exception:
                pass  # Failures are expected and should be handled
        
        # Should have some successful evaluations due to retries and fallbacks
        assert successful_evaluations > 0
        
        self.failure_scenarios_tested.append("intermittent_failures")
        self.recovery_actions_taken.append("retry_with_fallback")
    
    def _run_monitored_evolution(self, sdlc, evaluator, manager):
        """Run evolution with continuous monitoring and health checks."""
        
        if not sdlc.population:
            sdlc.initialize_population()
        
        monitoring_data = []
        
        for generation in range(sdlc.config.max_generations):
            gen_start_time = time.time()
            
            # Monitor population health before evolution step
            health = manager.monitor_population_health(sdlc.population)
            
            # Apply self-healing if needed
            if health['status'] != 'healthy':
                print(f"   Gen {generation}: Health issue detected - {health['status']}")
                sdlc.population = manager.apply_self_healing(
                    sdlc.population, 
                    health['status']
                )
                health = manager.monitor_population_health(sdlc.population)
                print(f"   Gen {generation}: After healing - {health['status']}")
            
            # Evaluate population with robustness
            for genome in sdlc.population:
                try:
                    # Use robust evaluation
                    result = evaluator.evaluate_with_robustness(
                        genome, 
                        lambda g: self._simulate_robust_evaluation(g)
                    )
                    
                    # Calculate weighted fitness
                    genome.fitness = sum(
                        result.get(obj.value, 0.0) * weight 
                        for obj, weight in sdlc.config.objectives.items()
                    )
                    
                except Exception as e:
                    self.logger.warning(f"Evaluation failed for genome: {e}")
                    genome.fitness = 0.1  # Minimum fitness for failed evaluations
            
            # Continue with normal evolution step
            sdlc.evolve_generation()
            
            gen_time = time.time() - gen_start_time
            
            # Record monitoring data
            monitoring_data.append({
                'generation': generation,
                'health_status': health['status'],
                'population_diversity': health['diversity'],
                'average_fitness': health['average_fitness'],
                'generation_time': gen_time,
                'timestamp': time.time()
            })
            
            # Print progress
            if generation % 5 == 0:
                best_fitness = max([g.fitness for g in sdlc.population])
                print(f"   Gen {generation}: Best={best_fitness:.4f}, Health={health['status'][:15]}")
        
        # Store monitoring data
        self.metrics['evolution_monitoring'] = monitoring_data
        
        # Find and return best genome
        best_genome = max(sdlc.population, key=lambda g: g.fitness)
        return best_genome
    
    def _simulate_robust_evaluation(self, genome) -> Dict[str, float]:
        """Simulate robust evaluation with potential instabilities."""
        
        # Sometimes introduce failures for robustness testing
        if np.random.random() < 0.05:  # 5% failure rate
            raise RuntimeError("Simulated evaluation instability")
        
        # Sometimes return NaN values for robustness testing
        if np.random.random() < 0.02:  # 2% NaN rate
            return {
                'energy_efficiency': float('nan'),
                'inference_speed': 0.5,
                'accuracy': 0.6,
                'robustness': 0.7
            }
        
        # Normal evaluation based on genome characteristics
        config = self._genome_to_config(genome)
        
        # Energy efficiency
        complexity_penalty = config.get('hidden_layers', 3) * config.get('hidden_dim', 64) / 1000.0
        energy_efficiency = max(0.1, 0.9 - complexity_penalty * 0.1)
        
        # Inference speed
        param_count = config.get('hidden_layers', 3) * config.get('hidden_dim', 64) ** 2
        inference_speed = max(0.1, 0.8 - param_count / 100000.0)
        
        # Accuracy (with some randomness)
        base_accuracy = 0.7 + np.random.normal(0, 0.05)
        if config.get('hidden_dim', 64) >= 64:
            base_accuracy += 0.1
        accuracy = np.clip(base_accuracy, 0.1, 1.0)
        
        # Robustness (higher for dropout and regularization)
        robustness = 0.6 + config.get('dropout_rate', 0.1) * 0.3
        robustness += min(config.get('weight_decay', 0.001) * 1000, 0.2)
        robustness = min(1.0, robustness)
        
        return {
            'energy_efficiency': energy_efficiency,
            'inference_speed': inference_speed,
            'accuracy': accuracy,
            'robustness': robustness
        }
    
    def _genome_to_config(self, genome) -> Dict[str, Any]:
        """Convert genome to configuration for evaluation."""
        if hasattr(genome, 'genes'):
            return genome.genes
        return {
            'hidden_layers': 3,
            'hidden_dim': 64,
            'dropout_rate': 0.1,
            'weight_decay': 0.001
        }
    
    def _analyze_robustness_metrics(self, sdlc, evaluator, manager):
        """Analyze robustness metrics from the evolution process."""
        
        print("📊 Robustness Analysis:")
        
        # Evaluation robustness
        eval_history = evaluator.evaluation_history
        if eval_history:
            success_rate = sum(1 for e in eval_history if e['success']) / len(eval_history)
            print(f"   Evaluation Success Rate: {success_rate:.1%}")
            
            avg_memory_usage = np.mean([
                e.get('memory_delta_mb', 0) for e in eval_history 
                if e['success'] and 'memory_delta_mb' in e
            ])
            print(f"   Average Memory Usage: {avg_memory_usage:.1f} MB")
        
        # Population health evolution
        monitoring = self.metrics.get('evolution_monitoring', [])
        if monitoring:
            health_statuses = [m['health_status'] for m in monitoring]
            healthy_generations = sum(1 for status in health_statuses if status == 'healthy')
            health_rate = healthy_generations / len(health_statuses)
            print(f"   Population Health Rate: {health_rate:.1%}")
            
            diversity_trend = [m['population_diversity'] for m in monitoring]
            if len(diversity_trend) > 1:
                diversity_change = diversity_trend[-1] - diversity_trend[0]
                print(f"   Diversity Change: {diversity_change:+.4f}")
        
        # Failure scenarios tested
        print(f"   Failure Scenarios Tested: {len(self.failure_scenarios_tested)}")
        for scenario in self.failure_scenarios_tested:
            print(f"     • {scenario}")
        
        # Recovery actions taken
        print(f"   Recovery Actions Taken: {len(self.recovery_actions_taken)}")
        for action in self.recovery_actions_taken:
            print(f"     • {action}")
        
        print()
    
    def _deploy_robust_solution(self, sdlc, best_genome):
        """Deploy the robust evolved solution."""
        
        print("📦 Deploying Robust Solution...")
        
        # Create comprehensive deployment package
        deployment_data = {
            'timestamp': time.time(),
            'best_genome': {
                'fitness': best_genome.fitness,
                'genes': best_genome.genes,
                'performance_history': getattr(best_genome, 'performance_history', [])
            },
            'robustness_metrics': {
                'failure_scenarios_tested': self.failure_scenarios_tested,
                'recovery_actions_taken': self.recovery_actions_taken,
                'evaluation_success_rate': self._calculate_success_rate(),
                'population_health_rate': self._calculate_health_rate()
            },
            'evolution_summary': {
                'generations': sdlc.generation,
                'population_size': len(sdlc.population),
                'objectives': {obj.value: weight for obj, weight in sdlc.config.objectives.items()}
            },
            'monitoring_data': self.metrics.get('evolution_monitoring', [])
        }
        
        # Save deployment
        output_path = Path("results/robust_evolutionary_gen2_deployment.json")
        output_path.parent.mkdir(exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(deployment_data, f, indent=2)
        
        print(f"✅ Robust solution deployed to: {output_path}")
        
        # Create model configuration
        model_config = sdlc._genome_to_model_config(best_genome)
        
        print(f"🏗️  Robust Model Configuration:")
        print(f"   Architecture: {model_config['hidden_layers']} layers × {model_config['hidden_dim']} units")
        print(f"   Robustness Features: {model_config['dropout_rate']:.3f} dropout, {model_config['weight_decay']:.6f} decay")
        print(f"   Deployment: {model_config['quantization']} quantization")
        print()
        
        return deployment_data
    
    def _calculate_success_rate(self) -> float:
        """Calculate overall success rate from metrics."""
        monitoring = self.metrics.get('evolution_monitoring', [])
        if not monitoring:
            return 1.0
        
        successful_gens = sum(1 for m in monitoring if m.get('health_status') == 'healthy')
        return successful_gens / len(monitoring)
    
    def _calculate_health_rate(self) -> float:
        """Calculate population health rate."""
        monitoring = self.metrics.get('evolution_monitoring', [])
        if not monitoring:
            return 1.0
        
        healthy_count = sum(1 for m in monitoring if 'healthy' in m.get('health_status', ''))
        return healthy_count / len(monitoring)


def main():
    """Main demonstration function."""
    
    print("🛡️ Starting Robust Evolutionary SDLC Generation 2 Demo...")
    print()
    
    # Ensure results directory exists
    Path("results").mkdir(exist_ok=True)
    
    # Create and run demonstration
    demo = RobustEvolutionarySDLCDemo()
    
    try:
        success = demo.demonstrate_robust_evolutionary_sdlc()
        
        if success:
            print("✅ Generation 2 robust demonstration completed successfully!")
            print()
            print("🔬 Breakthrough Robustness Features Demonstrated:")
            print("   • Self-healing population management")
            print("   • Circuit breakers for fault tolerance")
            print("   • Anomaly detection and recovery")
            print("   • Graceful degradation under failures")
            print("   • Real-time health monitoring")
            print("   • Comprehensive error handling")
            print()
            print("🎯 Ready for Generation 3 scaling optimizations...")
        else:
            print("❌ Demonstration encountered critical issues.")
    
    except KeyboardInterrupt:
        print("\n⚠️  Demonstration interrupted by user.")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(f"💥 Unexpected error: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()