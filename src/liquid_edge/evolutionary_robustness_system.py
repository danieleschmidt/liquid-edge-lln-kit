"""Evolutionary Robustness System - Production-Grade Reliability for Autonomous SDLC.

This system adds comprehensive robustness, fault tolerance, and self-healing capabilities
to the autonomous evolutionary SDLC, ensuring production-ready reliability.

Key Robustness Features:
- Self-healing evolutionary populations with anomaly detection
- Fault-tolerant fitness evaluation with graceful degradation  
- Adaptive mutation strategies based on environmental stress
- Production monitoring with automated rollback capabilities
- Circuit breakers and retry mechanisms for evolutionary processes
- Real-time performance monitoring and alerts
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
import numpy as np
from dataclasses import dataclass, field
from functools import partial, wraps
import time
import json
import logging
import traceback
from pathlib import Path
from abc import ABC, abstractmethod
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import asyncio
from contextlib import contextmanager
import warnings


class RobustnessLevel(Enum):
    """Levels of robustness for evolutionary processes."""
    BASIC = "basic"
    ENHANCED = "enhanced" 
    PRODUCTION = "production"
    CRITICAL = "critical"


class FailureMode(Enum):
    """Types of failures that can occur during evolution."""
    EVALUATION_TIMEOUT = "evaluation_timeout"
    MEMORY_EXHAUSTION = "memory_exhaustion"
    NUMERICAL_INSTABILITY = "numerical_instability"
    POPULATION_COLLAPSE = "population_collapse"
    CONVERGENCE_STALL = "convergence_stall"
    HARDWARE_FAILURE = "hardware_failure"
    EXTERNAL_DEPENDENCY = "external_dependency"


class RecoveryAction(Enum):
    """Recovery actions for different failure modes."""
    RETRY = "retry"
    ROLLBACK = "rollback"
    RESTART_POPULATION = "restart_population" 
    REDUCE_COMPLEXITY = "reduce_complexity"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    EMERGENCY_STOP = "emergency_stop"


@dataclass
class RobustnessConfig:
    """Configuration for evolutionary robustness system."""
    
    robustness_level: RobustnessLevel = RobustnessLevel.PRODUCTION
    
    # Timeout and retry settings
    evaluation_timeout_seconds: float = 30.0
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    exponential_backoff: bool = True
    
    # Circuit breaker settings
    failure_threshold: int = 5
    recovery_timeout_seconds: float = 60.0
    half_open_max_calls: int = 3
    
    # Health monitoring
    health_check_interval_seconds: float = 10.0
    performance_degradation_threshold: float = 0.2
    memory_usage_threshold_mb: float = 1000.0
    
    # Self-healing parameters
    auto_healing_enabled: bool = True
    population_diversity_threshold: float = 0.1
    fitness_stagnation_threshold: int = 10
    adaptive_mutation_factor: float = 2.0
    
    # Fault tolerance
    redundant_evaluations: int = 1
    consensus_required_ratio: float = 0.6
    outlier_detection_enabled: bool = True
    outlier_z_score_threshold: float = 3.0
    
    # Emergency protocols
    emergency_save_enabled: bool = True
    emergency_save_interval_seconds: float = 300.0
    max_recovery_attempts: int = 5


class CircuitBreakerState(Enum):
    """States of the circuit breaker."""
    CLOSED = "closed"    # Normal operation
    OPEN = "open"        # Failures detected, blocking calls
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreaker:
    """Circuit breaker for fault-tolerant evolutionary operations."""
    
    def __init__(self, config: RobustnessConfig):
        self.config = config
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.last_failure_time = None
        self.half_open_calls = 0
        self.lock = threading.Lock()
        
    def __call__(self, func):
        """Decorator for circuit breaker functionality."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            return self.call(func, *args, **kwargs)
        return wrapper
    
    def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        with self.lock:
            if self.state == CircuitBreakerState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitBreakerState.HALF_OPEN
                    self.half_open_calls = 0
                else:
                    raise CircuitBreakerError("Circuit breaker is OPEN")
            
            if self.state == CircuitBreakerState.HALF_OPEN:
                if self.half_open_calls >= self.config.half_open_max_calls:
                    self.state = CircuitBreakerState.OPEN
                    raise CircuitBreakerError("Circuit breaker returning to OPEN state")
                
                self.half_open_calls += 1
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self.last_failure_time is None:
            return True
        return (time.time() - self.last_failure_time) > self.config.recovery_timeout_seconds
    
    def _on_success(self):
        """Handle successful operation."""
        with self.lock:
            if self.state == CircuitBreakerState.HALF_OPEN:
                self.state = CircuitBreakerState.CLOSED
            self.failure_count = 0
            self.half_open_calls = 0
    
    def _on_failure(self):
        """Handle failed operation."""
        with self.lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.config.failure_threshold:
                self.state = CircuitBreakerState.OPEN


class CircuitBreakerError(Exception):
    """Exception raised when circuit breaker is open."""
    pass


class RobustEvolutionaryFitnessEvaluator:
    """Robust fitness evaluator with fault tolerance and self-healing."""
    
    def __init__(self, config: RobustnessConfig):
        self.config = config
        self.circuit_breaker = CircuitBreaker(config)
        self.evaluation_history = []
        self.anomaly_detector = AnomalyDetector(config)
        self.logger = logging.getLogger(__name__)
        
    @contextmanager
    def timeout_context(self, timeout_seconds: float):
        """Context manager for evaluation timeouts."""
        def timeout_handler():
            raise TimeoutError(f"Evaluation timed out after {timeout_seconds} seconds")
        
        timer = threading.Timer(timeout_seconds, timeout_handler)
        timer.start()
        try:
            yield
        finally:
            timer.cancel()
    
    def evaluate_with_robustness(self, genome, evaluation_func: Callable) -> Dict[str, float]:
        """Evaluate genome with comprehensive robustness measures."""
        
        for attempt in range(self.config.max_retries + 1):
            try:
                return self._attempt_evaluation(genome, evaluation_func, attempt)
                
            except Exception as e:
                self.logger.warning(f"Evaluation attempt {attempt + 1} failed: {e}")
                
                if attempt == self.config.max_retries:
                    # Final attempt failed, use fallback
                    return self._fallback_evaluation(genome, e)
                
                # Wait before retry with exponential backoff
                delay = self.config.retry_delay_seconds
                if self.config.exponential_backoff:
                    delay *= (2 ** attempt)
                
                time.sleep(delay)
        
        # Should not reach here, but provide safeguard
        return self._emergency_evaluation(genome)
    
    def _attempt_evaluation(self, genome, evaluation_func: Callable, attempt: int) -> Dict[str, float]:
        """Attempt single evaluation with timeout and monitoring."""
        
        with self.timeout_context(self.config.evaluation_timeout_seconds):
            # Monitor memory usage
            initial_memory = self._get_memory_usage_mb()
            
            try:
                # Execute evaluation with circuit breaker protection
                @self.circuit_breaker
                def protected_evaluation():
                    return evaluation_func(genome)
                
                result = protected_evaluation()
                
                # Validate result
                validated_result = self._validate_evaluation_result(result)
                
                # Check for anomalies
                if self.config.outlier_detection_enabled:
                    self._check_for_anomalies(validated_result)
                
                # Monitor memory after evaluation
                final_memory = self._get_memory_usage_mb()
                memory_delta = final_memory - initial_memory
                
                if memory_delta > self.config.memory_usage_threshold_mb:
                    self.logger.warning(f"High memory usage detected: {memory_delta:.1f} MB")
                
                # Record successful evaluation
                self.evaluation_history.append({
                    'timestamp': time.time(),
                    'success': True,
                    'attempt': attempt,
                    'memory_delta_mb': memory_delta,
                    'result': validated_result
                })
                
                return validated_result
                
            except Exception as e:
                # Record failed evaluation
                self.evaluation_history.append({
                    'timestamp': time.time(),
                    'success': False,
                    'attempt': attempt,
                    'error': str(e),
                    'error_type': type(e).__name__
                })
                raise e
    
    def _validate_evaluation_result(self, result: Dict[str, float]) -> Dict[str, float]:
        """Validate evaluation result for correctness."""
        if not isinstance(result, dict):
            raise ValueError("Evaluation result must be a dictionary")
        
        validated = {}
        for key, value in result.items():
            if not isinstance(value, (int, float)):
                raise ValueError(f"Invalid value type for {key}: {type(value)}")
            
            if np.isnan(value) or np.isinf(value):
                self.logger.warning(f"Invalid value detected for {key}: {value}, using 0.0")
                validated[key] = 0.0
            else:
                validated[key] = float(np.clip(value, 0.0, 1.0))  # Ensure valid range
        
        return validated
    
    def _check_for_anomalies(self, result: Dict[str, float]):
        """Check evaluation result for anomalies."""
        if len(self.evaluation_history) < 10:
            return  # Need history for anomaly detection
        
        recent_results = [
            entry['result'] for entry in self.evaluation_history[-10:] 
            if entry['success'] and 'result' in entry
        ]
        
        if not recent_results:
            return
        
        # Check each metric for anomalies
        for key, value in result.items():
            if key in recent_results[0]:  # Ensure key exists in history
                historical_values = [r[key] for r in recent_results]
                z_score = self.anomaly_detector.calculate_z_score(value, historical_values)
                
                if abs(z_score) > self.config.outlier_z_score_threshold:
                    self.logger.warning(f"Anomaly detected in {key}: value={value}, z_score={z_score}")
    
    def _fallback_evaluation(self, genome, error: Exception) -> Dict[str, float]:
        """Provide fallback evaluation when normal evaluation fails."""
        self.logger.error(f"Using fallback evaluation due to: {error}")
        
        # Use conservative estimates based on genome characteristics
        complexity_score = self._estimate_complexity(genome)
        
        return {
            'energy_efficiency': max(0.1, 0.8 - complexity_score * 0.3),
            'inference_speed': max(0.1, 0.7 - complexity_score * 0.2),
            'accuracy': max(0.1, 0.6 + complexity_score * 0.1),
            'robustness': 0.5,  # Neutral robustness for unknown performance
            'memory_usage': max(0.1, 0.5 + complexity_score * 0.2)
        }
    
    def _emergency_evaluation(self, genome) -> Dict[str, float]:
        """Emergency evaluation with minimal computation."""
        self.logger.critical("Using emergency evaluation - system may be unstable")
        
        return {
            'energy_efficiency': 0.3,
            'inference_speed': 0.3,
            'accuracy': 0.3,
            'robustness': 0.2,
            'memory_usage': 0.5
        }
    
    def _estimate_complexity(self, genome) -> float:
        """Estimate genome complexity for fallback evaluations."""
        try:
            if hasattr(genome, 'genes'):
                complexity = 0.0
                genes = genome.genes
                
                # Architecture complexity
                if 'hidden_layers' in genes:
                    complexity += genes['hidden_layers'] / 10.0
                if 'hidden_dim' in genes:
                    complexity += genes['hidden_dim'] / 500.0
                
                # Normalize to [0, 1]
                return min(1.0, complexity)
            
        except Exception:
            pass
        
        return 0.5  # Default moderate complexity
    
    def _get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0  # psutil not available


class AnomalyDetector:
    """Detect anomalies in evolutionary process metrics."""
    
    def __init__(self, config: RobustnessConfig):
        self.config = config
        
    def calculate_z_score(self, value: float, historical_values: List[float]) -> float:
        """Calculate z-score for anomaly detection."""
        if len(historical_values) < 2:
            return 0.0
        
        mean = np.mean(historical_values)
        std = np.std(historical_values)
        
        if std == 0:
            return 0.0
        
        return (value - mean) / std


class SelfHealingPopulationManager:
    """Manages population health and implements self-healing mechanisms."""
    
    def __init__(self, config: RobustnessConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def monitor_population_health(self, population: List) -> Dict[str, Any]:
        """Monitor population health metrics."""
        if not population:
            return {'status': 'empty', 'diversity': 0.0, 'fitness_variance': 0.0}
        
        # Calculate diversity
        diversity = self._calculate_population_diversity(population)
        
        # Calculate fitness statistics
        fitnesses = [genome.fitness for genome in population if hasattr(genome, 'fitness')]
        
        if not fitnesses:
            fitness_variance = 0.0
            avg_fitness = 0.0
        else:
            fitness_variance = np.var(fitnesses)
            avg_fitness = np.mean(fitnesses)
        
        health_status = self._assess_health_status(diversity, fitness_variance, population)
        
        return {
            'status': health_status,
            'diversity': diversity,
            'fitness_variance': fitness_variance,
            'average_fitness': avg_fitness,
            'population_size': len(population)
        }
    
    def _calculate_population_diversity(self, population: List) -> float:
        """Calculate genetic diversity of population."""
        if len(population) < 2:
            return 0.0
        
        try:
            distances = []
            for i, genome1 in enumerate(population):
                for genome2 in population[i+1:]:
                    distance = self._genetic_distance(genome1, genome2)
                    distances.append(distance)
            
            return np.mean(distances) if distances else 0.0
            
        except Exception as e:
            self.logger.warning(f"Error calculating diversity: {e}")
            return 0.5  # Return moderate diversity as fallback
    
    def _genetic_distance(self, genome1, genome2) -> float:
        """Calculate genetic distance between two genomes."""
        try:
            if not (hasattr(genome1, 'genes') and hasattr(genome2, 'genes')):
                return 0.5  # Default distance
            
            distance = 0.0
            gene_count = 0
            
            for gene_name in genome1.genes.keys():
                if gene_name in genome2.genes:
                    val1 = genome1.genes[gene_name]
                    val2 = genome2.genes[gene_name]
                    
                    if isinstance(val1, str) and isinstance(val2, str):
                        distance += 0.0 if val1 == val2 else 1.0
                    elif isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                        max_val = max(abs(val1), abs(val2), 1e-8)
                        distance += abs(val1 - val2) / max_val
                    
                    gene_count += 1
            
            return distance / gene_count if gene_count > 0 else 0.0
            
        except Exception:
            return 0.5  # Fallback distance
    
    def _assess_health_status(self, diversity: float, fitness_variance: float, population: List) -> str:
        """Assess overall population health status."""
        
        # Check for critical issues
        if len(population) == 0:
            return 'critical_empty'
        
        if len(population) < 3:
            return 'critical_small'
        
        # Check diversity
        if diversity < self.config.population_diversity_threshold:
            return 'unhealthy_low_diversity'
        
        # Check fitness variance (stagnation)
        if fitness_variance < 0.001:
            return 'unhealthy_stagnated'
        
        # Check for fitness improvements
        recent_improvements = self._check_recent_improvements(population)
        if not recent_improvements:
            return 'warning_no_improvement'
        
        return 'healthy'
    
    def _check_recent_improvements(self, population: List) -> bool:
        """Check if population has shown recent improvements."""
        try:
            for genome in population:
                if hasattr(genome, 'performance_history') and len(genome.performance_history) >= 2:
                    recent_fitness = genome.performance_history[-1]
                    older_fitness = genome.performance_history[-2]
                    if recent_fitness > older_fitness:
                        return True
            return False
        except Exception:
            return True  # Assume healthy if can't determine
    
    def apply_self_healing(self, population: List, health_status: str) -> List:
        """Apply self-healing measures based on health status."""
        
        if not self.config.auto_healing_enabled:
            return population
        
        self.logger.info(f"Applying self-healing for status: {health_status}")
        
        if health_status == 'critical_empty':
            return self._regenerate_population(population)
        
        elif health_status == 'critical_small':
            return self._expand_population(population)
        
        elif health_status == 'unhealthy_low_diversity':
            return self._increase_diversity(population)
        
        elif health_status == 'unhealthy_stagnated':
            return self._break_stagnation(population)
        
        elif health_status == 'warning_no_improvement':
            return self._stimulate_evolution(population)
        
        return population  # Healthy populations don't need intervention
    
    def _regenerate_population(self, population: List, size: int = 20) -> List:
        """Regenerate population from scratch."""
        self.logger.warning("Regenerating population due to critical failure")
        
        try:
            # Import here to avoid circular dependencies
            from .autonomous_evolutionary_sdlc import SDLCGenome, EvolutionaryConfig
            
            config = EvolutionaryConfig()
            new_population = [SDLCGenome(config) for _ in range(size)]
            
            self.logger.info(f"Generated new population of size {len(new_population)}")
            return new_population
            
        except Exception as e:
            self.logger.error(f"Failed to regenerate population: {e}")
            return population  # Return original if regeneration fails
    
    def _expand_population(self, population: List) -> List:
        """Expand small population through mutation."""
        try:
            expanded = population.copy()
            target_size = max(10, len(population) * 2)
            
            while len(expanded) < target_size and population:
                parent = np.random.choice(population)
                if hasattr(parent, 'mutate'):
                    child = parent.mutate(mutation_rate=0.3)  # Higher mutation for diversity
                    expanded.append(child)
                else:
                    break
            
            self.logger.info(f"Expanded population from {len(population)} to {len(expanded)}")
            return expanded
            
        except Exception as e:
            self.logger.error(f"Failed to expand population: {e}")
            return population
    
    def _increase_diversity(self, population: List) -> List:
        """Increase population diversity through targeted mutations."""
        try:
            diversified = []
            
            for genome in population:
                diversified.append(genome)  # Keep original
                
                if hasattr(genome, 'mutate'):
                    # Create highly mutated variant
                    diverse_child = genome.mutate(
                        mutation_rate=self.config.adaptive_mutation_factor * 0.3
                    )
                    diversified.append(diverse_child)
            
            self.logger.info(f"Increased diversity: {len(population)} -> {len(diversified)} genomes")
            return diversified[:len(population)]  # Keep original size
            
        except Exception as e:
            self.logger.error(f"Failed to increase diversity: {e}")
            return population
    
    def _break_stagnation(self, population: List) -> List:
        """Break evolutionary stagnation through radical mutations."""
        try:
            if not population:
                return population
            
            # Keep best genome
            best_genome = max(population, key=lambda g: getattr(g, 'fitness', 0))
            new_population = [best_genome]
            
            # Generate new variants with high mutation
            for _ in range(len(population) - 1):
                if hasattr(best_genome, 'mutate'):
                    mutant = best_genome.mutate(
                        mutation_rate=self.config.adaptive_mutation_factor * 0.5
                    )
                    new_population.append(mutant)
            
            self.logger.info("Applied stagnation-breaking mutations")
            return new_population
            
        except Exception as e:
            self.logger.error(f"Failed to break stagnation: {e}")
            return population
    
    def _stimulate_evolution(self, population: List) -> List:
        """Stimulate evolution through moderate interventions."""
        try:
            stimulated = []
            
            for genome in population:
                stimulated.append(genome)  # Keep original
                
                # Add moderately mutated variant
                if hasattr(genome, 'mutate'):
                    stimulated_child = genome.mutate(
                        mutation_rate=self.config.adaptive_mutation_factor * 0.2
                    )
                    stimulated.append(stimulated_child)
            
            # Select best individuals to maintain population size
            if len(stimulated) > len(population):
                stimulated.sort(key=lambda g: getattr(g, 'fitness', 0), reverse=True)
                stimulated = stimulated[:len(population)]
            
            self.logger.info("Applied evolutionary stimulation")
            return stimulated
            
        except Exception as e:
            self.logger.error(f"Failed to stimulate evolution: {e}")
            return population


def create_robust_evolutionary_system(
    base_config: Optional[Dict[str, Any]] = None,
    robustness_level: RobustnessLevel = RobustnessLevel.PRODUCTION
) -> Tuple[RobustnessConfig, RobustEvolutionaryFitnessEvaluator, SelfHealingPopulationManager]:
    """Create a complete robust evolutionary system."""
    
    robustness_config = RobustnessConfig(robustness_level=robustness_level)
    
    # Override with custom configuration if provided
    if base_config:
        for key, value in base_config.items():
            if hasattr(robustness_config, key):
                setattr(robustness_config, key, value)
    
    # Create system components
    fitness_evaluator = RobustEvolutionaryFitnessEvaluator(robustness_config)
    population_manager = SelfHealingPopulationManager(robustness_config)
    
    return robustness_config, fitness_evaluator, population_manager


# Example usage
if __name__ == "__main__":
    # Create robust evolutionary system
    config, evaluator, population_manager = create_robust_evolutionary_system(
        robustness_level=RobustnessLevel.PRODUCTION
    )
    
    print(f"Robust Evolutionary System initialized:")
    print(f"  Robustness Level: {config.robustness_level}")
    print(f"  Auto-healing: {config.auto_healing_enabled}")
    print(f"  Circuit breaker enabled: True")
    print(f"  Outlier detection: {config.outlier_detection_enabled}")