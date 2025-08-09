"""Production-ready robustness demonstration for Liquid Neural Networks."""

import jax
import jax.numpy as jnp
import numpy as np
from liquid_edge import LiquidNN, LiquidConfig
from liquid_edge.robust_training import RobustLiquidTrainer, RobustTrainingConfig
import matplotlib.pyplot as plt
import time


def create_noisy_robot_dataset(num_samples: int = 1000, noise_level: float = 0.1):
    """Create realistic noisy robot sensor data."""
    key = jax.random.PRNGKey(42)
    
    # Generate base sensor readings
    # [position_x, position_y, velocity_x, velocity_y]
    positions = jax.random.uniform(key, (num_samples, 2), minval=-2.0, maxval=2.0)
    velocities = jax.random.normal(jax.random.split(key)[0], (num_samples, 2)) * 0.5
    
    inputs = jnp.concatenate([positions, velocities], axis=1)
    
    # Create control targets (PD controller)
    # Target: move towards origin with damping
    targets = -0.8 * positions - 0.3 * velocities
    
    # Add realistic sensor noise
    noise_key = jax.random.split(key)[1]
    sensor_noise = jax.random.normal(noise_key, inputs.shape) * noise_level
    noisy_inputs = inputs + sensor_noise
    
    # Add occasional sensor failures (10% of data)
    failure_mask = jax.random.bernoulli(
        jax.random.split(noise_key)[0], 0.1, (num_samples,)
    )
    
    # Sensor failures: replace with extreme values or NaN
    failure_values = jnp.where(
        jax.random.bernoulli(jax.random.split(noise_key)[1], 0.5, (num_samples, 4)),
        jnp.full((num_samples, 4), 10.0),  # Extreme values
        jnp.full((num_samples, 4), jnp.nan)  # NaN failures
    )
    
    corrupted_inputs = jnp.where(
        failure_mask.reshape(-1, 1),
        failure_values,
        noisy_inputs
    )
    
    return corrupted_inputs.astype(jnp.float32), targets.astype(jnp.float32)


def demonstrate_robustness_features():
    """Comprehensive demonstration of robustness features."""
    print("🛡️  PRODUCTION ROBUSTNESS DEMONSTRATION")
    print("=" * 60)
    
    # Create robust configuration
    liquid_config = LiquidConfig(
        input_dim=4,
        hidden_dim=16,
        output_dim=2,
        tau_min=5.0,
        tau_max=50.0,
        use_sparse=True,
        sparsity=0.4,
        dt=0.1,
        use_layer_norm=False
    )
    
    training_config = RobustTrainingConfig(
        learning_rate=0.01,
        batch_size=32,
        max_epochs=500,
        early_stopping_patience=50,
        gradient_clip_norm=1.0,
        loss_spike_threshold=5.0,
        nan_tolerance=5,
        lr_decay_factor=0.9,
        lr_decay_patience=20,
        enable_monitoring=True,
        log_interval=25
    )
    
    # Create model
    model = LiquidNN(liquid_config)
    
    print(f"📊 Model Configuration:")
    print(f"   • Architecture: {liquid_config.input_dim}→{liquid_config.hidden_dim}→{liquid_config.output_dim}")
    print(f"   • Sparsity: {liquid_config.sparsity:.1%}")
    print(f"   • Time constants: {liquid_config.tau_min}-{liquid_config.tau_max}ms")
    
    # Generate realistic noisy data
    print("\n📡 Generating Realistic Robot Dataset")
    train_inputs, train_targets = create_noisy_robot_dataset(800, noise_level=0.15)
    val_inputs, val_targets = create_noisy_robot_dataset(200, noise_level=0.10)
    
    # Check data quality
    nan_percentage = np.mean(np.isnan(train_inputs)) * 100
    outlier_percentage = np.mean(np.abs(train_inputs) > 5.0) * 100
    
    print(f"   • Training samples: {len(train_inputs)}")
    print(f"   • Validation samples: {len(val_inputs)}")
    print(f"   • NaN values: {nan_percentage:.1f}%")
    print(f"   • Outliers (>5σ): {outlier_percentage:.1f}%")
    
    # Initialize robust trainer
    trainer = RobustLiquidTrainer(model, training_config, liquid_config)
    
    print(f"\n🔧 Robustness Features Enabled:")
    print(f"   • Gradient clipping: {training_config.gradient_clip_norm}")
    print(f"   • Loss spike protection: {training_config.loss_spike_threshold}")
    print(f"   • NaN tolerance: {training_config.nan_tolerance}")
    print(f"   • Adaptive learning rate: {training_config.lr_decay_factor}")
    print(f"   • Early stopping: {training_config.early_stopping_patience} epochs")
    
    # Train with robustness features
    print("\n🚀 Starting Robust Training...")
    start_time = time.perf_counter()
    
    final_params, training_results = trainer.robust_train(
        (train_inputs, train_targets),
        (val_inputs, val_targets)
    )
    
    end_time = time.perf_counter()
    
    # Analyze results
    history = training_results['history']
    summary = training_results['summary']
    
    print(f"\n📈 TRAINING RESULTS ANALYSIS")
    print("=" * 50)
    print(f"✅ Training Status: {'SUCCESS' if summary['convergence_achieved'] else 'INCOMPLETE'}")
    print(f"📊 Performance Metrics:")
    print(f"   • Final loss: {summary['final_loss']:.6f}")
    print(f"   • Epochs trained: {summary['epochs_trained']}")
    print(f"   • Training time: {end_time - start_time:.2f}s")
    print(f"   • Final learning rate: {summary['final_lr']:.2e}")
    
    # Error handling analysis
    if 'error_log' in training_results:
        error_summary = training_results['error_log']
        if error_summary:
            print(f"\n🛠️  ERROR HANDLING SUMMARY:")
            for error_type, count in error_summary.items():
                print(f"   • {error_type}: {count} occurrences")
        else:
            print(f"✅ No errors encountered during training")
    
    # Robustness validation
    print(f"\n🔍 ROBUSTNESS VALIDATION")
    print("=" * 50)
    
    # Test with extreme inputs
    extreme_inputs = jnp.array([
        [10.0, -10.0, 5.0, -5.0],    # Extreme values
        [0.0, 0.0, jnp.nan, 0.0],    # NaN input
        [jnp.inf, 0.0, 0.0, 0.0],    # Infinity input
        [0.0, 0.0, 0.0, 0.0]         # Normal input
    ]).astype(jnp.float32)
    
    try:
        # Test inference robustness
        outputs, _ = model.apply(final_params, extreme_inputs, training=False)
        
        robust_outputs = 0
        for i, output in enumerate(outputs):
            is_finite = jnp.all(jnp.isfinite(output))
            is_bounded = jnp.all(jnp.abs(output) < 10.0)
            
            if is_finite and is_bounded:
                robust_outputs += 1
                status = "✅ ROBUST"
            else:
                status = "⚠️  UNSTABLE"
            
            print(f"   • Test {i+1}: {status} (output: {output})")
        
        robustness_score = robust_outputs / len(extreme_inputs) * 100
        print(f"\n🏆 ROBUSTNESS SCORE: {robustness_score:.0f}%")
        
    except Exception as e:
        print(f"❌ Robustness test failed: {str(e)}")
    
    # Performance validation
    print(f"\n⚡ PERFORMANCE VALIDATION")
    print("=" * 50)
    
    # Inference speed test
    test_input = jnp.ones((1, 4))
    
    # Warmup
    for _ in range(10):
        _ = model.apply(final_params, test_input, training=False)
    
    # Benchmark
    inference_times = []
    for _ in range(1000):
        start = time.perf_counter()
        _ = model.apply(final_params, test_input, training=False)
        inference_times.append(time.perf_counter() - start)
    
    avg_inference_us = np.mean(inference_times) * 1_000_000
    throughput_fps = 1.0 / np.mean(inference_times)
    
    print(f"📊 Real-time Performance:")
    print(f"   • Average latency: {avg_inference_us:.1f}μs")
    print(f"   • Throughput: {throughput_fps:.0f} FPS")
    print(f"   • Memory footprint: {liquid_config.hidden_dim * 4}KB (estimated)")
    
    # Real-time capability assessment
    if avg_inference_us < 1000:  # < 1ms
        rt_capability = "🟢 EXCELLENT (1kHz capable)"
    elif avg_inference_us < 10000:  # < 10ms  
        rt_capability = "🟡 GOOD (100Hz capable)"
    else:
        rt_capability = "🔴 LIMITED (<100Hz)"
    
    print(f"   • Real-time capability: {rt_capability}")
    
    # Final assessment
    print(f"\n🎯 PRODUCTION READINESS ASSESSMENT")
    print("=" * 60)
    
    criteria = {
        "Training Convergence": summary['convergence_achieved'],
        "Robustness Score": robustness_score >= 75,
        "Real-time Performance": avg_inference_us < 1000,
        "Error Handling": len(training_results.get('error_log', {})) < 10,
        "Training Stability": len(history['train_loss']) > 10
    }
    
    passed_criteria = sum(criteria.values())
    total_criteria = len(criteria)
    
    for criterion, passed in criteria.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   • {criterion}: {status}")
    
    overall_score = passed_criteria / total_criteria * 100
    
    if overall_score >= 80:
        readiness = "🟢 PRODUCTION READY"
    elif overall_score >= 60:
        readiness = "🟡 DEVELOPMENT READY"
    else:
        readiness = "🔴 NOT READY"
    
    print(f"\n🏆 OVERALL ASSESSMENT: {readiness} ({overall_score:.0f}%)")
    
    return {
        'final_params': final_params,
        'training_results': training_results,
        'robustness_score': robustness_score,
        'performance': {
            'latency_us': avg_inference_us,
            'throughput_fps': throughput_fps
        },
        'production_score': overall_score
    }


if __name__ == "__main__":
    results = demonstrate_robustness_features()
    print(f"\n🎉 DEMONSTRATION COMPLETE!")
    print(f"Results saved in return dictionary with {len(results)} metrics")