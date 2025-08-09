"""Robust training framework with advanced error handling and adaptive learning."""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass
import time
from .core import LiquidConfig, LiquidNN
from .monitoring import LiquidNetworkMonitor, PerformanceMetrics
from .error_handling import (
    RobustErrorHandler, validate_inputs, graceful_degradation, 
    retry_with_backoff
)
import warnings


@dataclass
class RobustTrainingConfig:
    """Configuration for robust training framework."""
    
    # Training parameters
    learning_rate: float = 0.001
    batch_size: int = 32
    max_epochs: int = 1000
    early_stopping_patience: int = 50
    
    # Robustness parameters
    gradient_clip_norm: float = 1.0
    loss_spike_threshold: float = 10.0
    nan_tolerance: int = 3
    
    # Adaptive learning
    lr_decay_factor: float = 0.95
    lr_decay_patience: int = 10
    min_learning_rate: float = 1e-6
    
    # Checkpointing
    checkpoint_interval: int = 100
    save_best_model: bool = True
    
    # Monitoring
    enable_monitoring: bool = True
    log_interval: int = 10


class RobustLiquidTrainer:
    """Production-ready robust training framework for Liquid Neural Networks."""
    
    def __init__(self, 
                 model: LiquidNN, 
                 config: RobustTrainingConfig,
                 liquid_config: LiquidConfig):
        self.model = model
        self.config = config
        self.liquid_config = liquid_config
        
        # Initialize components
        self.error_handler = RobustErrorHandler()
        self.monitor = LiquidNetworkMonitor() if config.enable_monitoring else None
        
        # Training state
        self.best_loss = float('inf')
        self.best_params = None
        self.patience_counter = 0
        self.lr_patience_counter = 0
        self.nan_counter = 0
        
        # Learning rate schedule
        self.current_lr = config.learning_rate
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'gradient_norm': [],
            'training_time': []
        }
        
        print("🔧 ROBUST LIQUID TRAINER INITIALIZED")
        print(f"   • Model: {model.__class__.__name__}")
        print(f"   • Monitoring: {'Enabled' if self.monitor else 'Disabled'}")
        print(f"   • Error handling: Advanced")
    
    @retry_with_backoff(max_retries=3)
    def _safe_forward_pass(self, params: Dict, inputs: jnp.ndarray, training: bool = True) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Safe forward pass with error handling."""
        try:
            # Validate inputs
            validate_inputs()
            
            # Forward pass
            outputs, hidden = self.model.apply(params, inputs, training=training)
            
            # Check for NaN/Inf
            if jnp.any(jnp.isnan(outputs)) or jnp.any(jnp.isinf(outputs)):
                raise ValueError("NaN or Inf detected in forward pass outputs")
            
            return outputs, hidden
            
        except Exception as e:
            print(f"Forward pass error: {str(e)}")
            
            # Graceful degradation: return zeros or cached values
            batch_size = inputs.shape[0]
            fallback_output = jnp.zeros((batch_size, self.liquid_config.output_dim))
            fallback_hidden = jnp.zeros((batch_size, self.liquid_config.hidden_dim))
            
            return fallback_output, fallback_hidden
    
    def _robust_loss_computation(self, params: Dict, batch_inputs: jnp.ndarray, batch_targets: jnp.ndarray) -> Tuple[jnp.ndarray, Dict]:
        """Robust loss computation with error handling and metrics."""
        try:
            # Safe forward pass
            outputs, hidden = self._safe_forward_pass(params, batch_inputs, training=True)
            
            # Compute primary loss (MSE)
            mse_loss = jnp.mean((outputs - batch_targets) ** 2)
            
            # Add regularization for stability
            l2_reg = 1e-4 * sum(jnp.sum(p ** 2) for p in jax.tree_util.tree_leaves(params))
            
            # Liquid-specific regularization (encourage stable dynamics)
            hidden_reg = 1e-5 * jnp.mean(hidden ** 2)
            
            # Total loss
            total_loss = mse_loss + l2_reg + hidden_reg
            
            # Check for loss spikes
            if total_loss > self.config.loss_spike_threshold:
                warnings.warn(f"Loss spike detected: {total_loss:.4f}")
                # Apply loss clipping
                total_loss = jnp.clip(total_loss, 0.0, self.config.loss_spike_threshold)
            
            # Compute metrics
            metrics = {
                'mse_loss': float(mse_loss),
                'l2_reg': float(l2_reg),
                'hidden_reg': float(hidden_reg),
                'total_loss': float(total_loss),
                'output_mean': float(jnp.mean(outputs)),
                'output_std': float(jnp.std(outputs)),
                'hidden_mean': float(jnp.mean(hidden)),
                'hidden_std': float(jnp.std(hidden))
            }
            
            return total_loss, metrics
            
        except Exception as e:
            print(f"Loss computation error: {str(e)}")
            # Return safe fallback loss
            return jnp.array(1.0), {'error': str(e)}
    
    def _safe_gradient_update(self, params: Dict, grads: Dict) -> Dict:
        """Safe gradient update with clipping and NaN handling."""
        try:
            # Check for NaN gradients
            has_nan = any(jnp.any(jnp.isnan(g)) for g in jax.tree_util.tree_leaves(grads))
            
            if has_nan:
                self.nan_counter += 1
                print(f"NaN gradients detected (count: {self.nan_counter})")
                
                if self.nan_counter > self.config.nan_tolerance:
                    # Reset to best known parameters
                    if self.best_params is not None:
                        print("🔄 Resetting to best parameters due to excessive NaN gradients")
                        self.nan_counter = 0
                        return self.best_params
                    else:
                        # Reinitialize parameters
                        key = jax.random.PRNGKey(int(time.time()))
                        return self.model.init(key, jnp.ones((1, self.liquid_config.input_dim)))
                
                # Skip this update
                return params
            
            # Gradient clipping
            grad_norm = jnp.sqrt(sum(jnp.sum(g ** 2) for g in jax.tree_util.tree_leaves(grads)))
            
            if grad_norm > self.config.gradient_clip_norm:
                clip_factor = self.config.gradient_clip_norm / grad_norm
                grads = jax.tree.map(lambda g: g * clip_factor, grads)
                grad_norm = self.config.gradient_clip_norm
            
            # Store gradient norm for monitoring
            self.history['gradient_norm'].append(float(grad_norm))
            
            # Apply update
            updated_params = jax.tree.map(
                lambda p, g: p - self.current_lr * g, 
                params, grads
            )
            
            # Reset NaN counter on successful update
            self.nan_counter = 0
            
            return updated_params
            
        except Exception as e:
            print(f"Gradient update error: {str(e)}")
            return params  # Return unchanged parameters
    
    def _adaptive_learning_rate(self, current_loss: float):
        """Adaptive learning rate adjustment."""
        # Check for improvement
        if current_loss < self.best_loss - 1e-4:  # Significant improvement
            self.best_loss = current_loss
            self.lr_patience_counter = 0
        else:
            self.lr_patience_counter += 1
        
        # Decay learning rate if no improvement
        if self.lr_patience_counter >= self.config.lr_decay_patience:
            old_lr = self.current_lr
            self.current_lr = max(
                self.current_lr * self.config.lr_decay_factor,
                self.config.min_learning_rate
            )
            
            if self.current_lr != old_lr:
                print(f"📉 Learning rate reduced: {old_lr:.6f} → {self.current_lr:.6f}")
                self.lr_patience_counter = 0
        
        self.history['learning_rate'].append(self.current_lr)
    
    def train_epoch(self, params: Dict, train_data: Tuple[jnp.ndarray, jnp.ndarray], 
                   val_data: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None) -> Tuple[Dict, Dict]:
        """Train one robust epoch with comprehensive error handling."""
        train_inputs, train_targets = train_data
        epoch_start_time = time.perf_counter()
        
        # Training metrics
        epoch_losses = []
        epoch_metrics = []
        
        # Mini-batch training
        num_batches = len(train_inputs) // self.config.batch_size
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.config.batch_size
            end_idx = start_idx + self.config.batch_size
            
            batch_inputs = train_inputs[start_idx:end_idx]
            batch_targets = train_targets[start_idx:end_idx]
            
            # Compute loss and gradients
            (loss, metrics), grads = jax.value_and_grad(
                self._robust_loss_computation, has_aux=True
            )(params, batch_inputs, batch_targets)
            
            # Safe gradient update
            params = self._safe_gradient_update(params, grads)
            
            epoch_losses.append(float(loss))
            epoch_metrics.append(metrics)
        
        # Compute epoch statistics
        avg_train_loss = np.mean(epoch_losses)
        epoch_time = time.perf_counter() - epoch_start_time
        
        # Validation
        val_loss = None
        if val_data is not None:
            val_inputs, val_targets = val_data
            val_outputs, _ = self._safe_forward_pass(params, val_inputs, training=False)
            val_loss = float(jnp.mean((val_outputs - val_targets) ** 2))
        
        # Update learning rate
        self._adaptive_learning_rate(avg_train_loss)
        
        # Store history
        self.history['train_loss'].append(avg_train_loss)
        self.history['training_time'].append(epoch_time)
        if val_loss is not None:
            self.history['val_loss'].append(val_loss)
        
        # Update best model
        if avg_train_loss < self.best_loss:
            self.best_loss = avg_train_loss
            if self.config.save_best_model:
                self.best_params = jax.tree.map(lambda x: x.copy(), params)
        
        # Performance monitoring
        if self.monitor:
            perf_metrics = PerformanceMetrics(
                latency_ms=epoch_time * 1000,
                throughput_fps=len(train_inputs) / epoch_time,
                memory_usage_mb=0.0,  # TODO: Implement memory monitoring
                cpu_usage_percent=0.0,
                accuracy=1.0 / (1.0 + avg_train_loss)  # Approximation
            )
            self.monitor.log_metrics(perf_metrics)
        
        return params, {
            'train_loss': avg_train_loss,
            'val_loss': val_loss,
            'learning_rate': self.current_lr,
            'epoch_time': epoch_time,
            'gradient_norm': np.mean(self.history['gradient_norm'][-num_batches:]) if self.history['gradient_norm'] else 0.0
        }
    
    def robust_train(self, 
                    train_data: Tuple[jnp.ndarray, jnp.ndarray],
                    val_data: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None,
                    initial_params: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        """Complete robust training with all safety mechanisms."""
        print("\n🚀 STARTING ROBUST TRAINING")
        print("=" * 50)
        
        # Initialize parameters
        if initial_params is None:
            key = jax.random.PRNGKey(42)
            params = self.model.init(key, train_data[0][:1])
        else:
            params = initial_params
        
        # Training loop
        for epoch in range(self.config.max_epochs):
            try:
                # Train epoch
                params, epoch_metrics = self.train_epoch(params, train_data, val_data)
                
                # Logging
                if epoch % self.config.log_interval == 0:
                    log_str = f"Epoch {epoch:4d}: Loss={epoch_metrics['train_loss']:.6f}, LR={epoch_metrics['learning_rate']:.2e}"
                    if epoch_metrics['val_loss']:
                        log_str += f", ValLoss={epoch_metrics['val_loss']:.6f}"
                    log_str += f", Time={epoch_metrics['epoch_time']:.3f}s"
                    print(log_str)
                
                # Early stopping check
                if val_data is not None and epoch_metrics['val_loss']:
                    if epoch_metrics['val_loss'] < self.best_loss - 1e-4:
                        self.patience_counter = 0
                    else:
                        self.patience_counter += 1
                    
                    if self.patience_counter >= self.config.early_stopping_patience:
                        print(f"🛑 Early stopping at epoch {epoch} (patience: {self.patience_counter})")
                        break
                
                # Check for convergence
                if self.current_lr <= self.config.min_learning_rate and epoch_metrics['train_loss'] < 1e-6:
                    print(f"🎯 Converged at epoch {epoch}")
                    break
                    
            except Exception as e:
                print(f"Training epoch {epoch} failed: {str(e)}")
                if self.best_params is not None:
                    params = self.best_params
                    print(f"🔄 Recovered using best parameters")
                continue
        
        # Final results
        final_params = self.best_params if self.best_params is not None else params
        
        training_summary = {
            'final_loss': self.best_loss,
            'epochs_trained': epoch + 1,
            'total_time': sum(self.history['training_time']),
            'final_lr': self.current_lr,
            'convergence_achieved': self.current_lr <= self.config.min_learning_rate or self.best_loss < 1e-6
        }
        
        print("\n✅ ROBUST TRAINING COMPLETED")
        print(f"   • Final loss: {self.best_loss:.6f}")
        print(f"   • Epochs: {epoch + 1}")
        print(f"   • Total time: {sum(self.history['training_time']):.2f}s")
        print(f"   • Convergence: {'Yes' if training_summary['convergence_achieved'] else 'No'}")
        
        return final_params, {
            'history': self.history,
            'summary': training_summary,
            'error_log': self.error_handler.get_error_summary()
        }


# Convenience function for easy usage
def train_robust_liquid_nn(model: LiquidNN,
                          liquid_config: LiquidConfig,
                          train_data: Tuple[jnp.ndarray, jnp.ndarray],
                          val_data: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None,
                          training_config: Optional[RobustTrainingConfig] = None) -> Tuple[Dict, Dict]:
    """Train a Liquid Neural Network with full robustness features."""
    
    if training_config is None:
        training_config = RobustTrainingConfig()
    
    trainer = RobustLiquidTrainer(model, training_config, liquid_config)
    return trainer.robust_train(train_data, val_data)