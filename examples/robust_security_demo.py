#!/usr/bin/env python3
"""Demonstration of advanced security and fault tolerance features."""

import time
import numpy as np
import jax
import jax.numpy as jnp
from liquid_edge import LiquidNN, LiquidConfig
from liquid_edge.advanced_security import SecurityConfig, SecureLiquidInference, SecurityError
from liquid_edge.fault_tolerance import FaultToleranceConfig, FaultTolerantSystem


def create_test_data(num_samples: int = 100) -> tuple:
    """Create test data including normal and adversarial examples."""
    np.random.seed(42)
    
    # Normal sensor data
    normal_data = np.random.normal(0.0, 0.5, (num_samples, 4))
    normal_targets = np.sin(normal_data.sum(axis=1, keepdims=True)) * np.array([[1.0, -0.5]])
    
    # Adversarial data (extreme values, NaN, etc.)
    adversarial_data = []
    
    # Extreme values
    extreme_data = np.random.normal(0.0, 10.0, (10, 4))
    adversarial_data.append(extreme_data)
    
    # NaN injection
    nan_data = np.random.normal(0.0, 0.5, (5, 4))
    nan_data[0, 0] = np.nan
    adversarial_data.append(nan_data)
    
    # Infinity injection
    inf_data = np.random.normal(0.0, 0.5, (5, 4))
    inf_data[0, 1] = np.inf
    adversarial_data.append(inf_data)
    
    return normal_data, normal_targets, adversarial_data


def test_security_features():
    """Test advanced security monitoring."""
    print("🔒 Testing Advanced Security Features")
    print("=" * 50)
    
    # Create liquid neural network
    config = LiquidConfig(
        input_dim=4,
        hidden_dim=12,
        output_dim=2,
        energy_budget_mw=80.0
    )
    
    model = LiquidNN(config)
    key = jax.random.PRNGKey(42)
    params = model.init(key, jnp.ones((1, 4)))
    
    # Configure security
    security_config = SecurityConfig(
        enable_input_validation=True,
        enable_adversarial_detection=True,
        enable_model_integrity=True,
        enable_timing_protection=True,
        adversarial_threshold=0.1,
        max_inference_time_ms=100.0
    )
    
    # Create secure inference wrapper
    secure_model = SecureLiquidInference(model, security_config)
    secure_model.set_model_params(params)
    
    # Generate test data
    normal_data, _, adversarial_data = create_test_data()
    
    print("📊 Testing normal inputs...")
    successful_inferences = 0
    total_attempts = 0
    
    for i in range(10):
        try:
            inputs = jnp.array(normal_data[i:i+1])
            output, _ = secure_model(params, inputs)
            successful_inferences += 1
            total_attempts += 1
            print(f"  ✅ Normal inference {i+1}: Success")
        except SecurityError as e:
            total_attempts += 1
            print(f"  ❌ Normal inference {i+1}: {e}")
        except Exception as e:
            total_attempts += 1
            print(f"  ⚠️  Normal inference {i+1}: Unexpected error: {e}")
    
    print(f"\nNormal inference success rate: {successful_inferences}/{total_attempts} ({successful_inferences/total_attempts*100:.1f}%)")
    
    print("\n🚨 Testing adversarial inputs...")
    security_blocks = 0
    
    for i, adv_batch in enumerate(adversarial_data):
        for j, adv_input in enumerate(adv_batch):
            try:
                inputs = jnp.array(adv_input.reshape(1, -1))
                output, _ = secure_model(params, inputs)
                print(f"  ⚠️  Adversarial input {i}-{j}: Passed security (unexpected)")
            except SecurityError as e:
                security_blocks += 1
                print(f"  🛡️  Adversarial input {i}-{j}: Blocked - {e}")
            except Exception as e:
                print(f"  💥 Adversarial input {i}-{j}: System error - {e}")
    
    print(f"\nSecurity blocked {security_blocks} adversarial attempts")
    
    # Test model integrity
    print("\n🔍 Testing model integrity...")
    original_integrity = secure_model.security_monitor.check_model_integrity(params)
    print(f"Original model integrity: {'✅ Valid' if original_integrity else '❌ Invalid'}")
    
    # Simulate model tampering
    tampered_params = params.copy()
    if 'params' in tampered_params and 'liquid_cell' in tampered_params['params']:
        # Modify a parameter slightly
        cell_params = tampered_params['params']['liquid_cell']
        if 'kernel' in cell_params:
            cell_params['kernel'] = cell_params['kernel'] + 0.001
    
    tampered_integrity = secure_model.security_monitor.check_model_integrity(tampered_params)
    print(f"Tampered model integrity: {'✅ Valid' if tampered_integrity else '❌ Invalid (Expected)'}")
    
    # Generate security report
    print("\n📋 Security Report:")
    report = secure_model.get_security_report()
    print(f"  Security Score: {report['security_status']['security_score']:.1f}/100")
    print(f"  Total Security Events: {report['security_status']['total_events']}")
    print(f"  Threat Counts: {report['security_status']['threat_counts']}")
    print(f"  Model Integrity: {'✅ Enabled' if report['config']['model_integrity'] else '❌ Disabled'}")
    
    if report['recommendations']:
        print("  Recommendations:")
        for rec in report['recommendations']:
            print(f"    • {rec}")


def test_fault_tolerance():
    """Test fault tolerance and recovery mechanisms."""
    print("\n\n🛠️  Testing Fault Tolerance Features")
    print("=" * 50)
    
    # Create liquid neural network
    config = LiquidConfig(
        input_dim=4,
        hidden_dim=8,
        output_dim=2,
        energy_budget_mw=60.0
    )
    
    model = LiquidNN(config)
    key = jax.random.PRNGKey(42)
    params = model.init(key, jnp.ones((1, 4)))
    
    # Configure fault tolerance
    ft_config = FaultToleranceConfig(
        max_retries=3,
        retry_delay_ms=50.0,
        timeout_ms=200.0,
        enable_redundancy=True,
        enable_graceful_degradation=True,
        enable_checkpointing=True,
        checkpoint_interval=5
    )
    
    # Create fault-tolerant system
    ft_system = FaultTolerantSystem(ft_config)
    
    # Register primary model and backup
    ft_system.register_model(model, is_primary=True)
    
    # Create a backup model (simplified)
    backup_model = LiquidNN(config)
    backup_params = backup_model.init(jax.random.PRNGKey(123), jnp.ones((1, 4)))
    ft_system.register_model(backup_model, is_primary=False)
    
    def model_inference(params, inputs, **kwargs):
        """Wrapper for model inference."""
        return model.apply(params, inputs, **kwargs)
    
    def faulty_inference(params, inputs, fault_type="timeout", **kwargs):
        """Simulate different types of faults."""
        if fault_type == "timeout":
            time.sleep(0.3)  # Longer than timeout
            return model.apply(params, inputs, **kwargs)
        elif fault_type == "nan":
            # Return NaN to simulate computation error
            result = model.apply(params, inputs, **kwargs)
            return result[0] * np.nan, result[1]
        elif fault_type == "memory":
            raise MemoryError("Simulated memory allocation failure")
        elif fault_type == "communication":
            raise ConnectionError("Simulated sensor communication failure")
        else:
            return model.apply(params, inputs, **kwargs)
    
    # Generate test data
    normal_data, _, _ = create_test_data()
    test_inputs = jnp.array(normal_data[:10])
    
    print("📊 Testing normal fault-tolerant inference...")
    for i in range(5):
        try:
            inputs = test_inputs[i:i+1]
            result = ft_system.fault_tolerant_inference(
                model_inference, params, inputs
            )
            print(f"  ✅ Normal inference {i+1}: Success, output shape: {result.shape}")
        except Exception as e:
            print(f"  ❌ Normal inference {i+1}: Failed - {e}")
    
    print("\n🚨 Testing fault scenarios...")
    
    # Test timeout handling
    print("\n  Testing timeout recovery:")
    try:
        inputs = test_inputs[0:1]
        result = ft_system.fault_tolerant_inference(
            lambda p, x, **kw: faulty_inference(p, x, "timeout", **kw),
            params, inputs
        )
        print(f"    ✅ Timeout handled, output shape: {result.shape}")
    except Exception as e:
        print(f"    ⚠️  Timeout recovery failed: {e}")
    
    # Test computation error recovery
    print("\n  Testing computation error recovery:")
    try:
        inputs = test_inputs[1:2]
        result = ft_system.fault_tolerant_inference(
            lambda p, x, **kw: faulty_inference(p, x, "nan", **kw),
            params, inputs
        )
        print(f"    ✅ Computation error handled, output shape: {result.shape}")
    except Exception as e:
        print(f"    ⚠️  Computation error recovery failed: {e}")
    
    # Test memory error recovery
    print("\n  Testing memory error recovery:")
    try:
        inputs = test_inputs[2:3]
        result = ft_system.fault_tolerant_inference(
            lambda p, x, **kw: faulty_inference(p, x, "memory", **kw),
            params, inputs
        )
        print(f"    ✅ Memory error handled, output shape: {result.shape}")
    except Exception as e:
        print(f"    ⚠️  Memory error recovery failed: {e}")
    
    # Test checkpointing
    print("\n  Testing checkpointing:")
    ft_system.create_checkpoint("test_checkpoint", {"params": params, "metadata": "test"})
    checkpoint_data = ft_system.restore_checkpoint("test_checkpoint")
    if checkpoint_data:
        print("    ✅ Checkpoint created and restored successfully")
    else:
        print("    ❌ Checkpoint restoration failed")
    
    # Generate fault tolerance report
    print("\n📋 Fault Tolerance Report:")
    report = ft_system.get_fault_report()
    print(f"  System State: {report['system_state']}")
    print(f"  Total Inferences: {report['total_inferences']}")
    print(f"  Total Faults: {report['total_faults']}")
    print(f"  Recovery Success Rate: {report['recovery_success_rate']:.1f}%")
    print(f"  Energy Level: {report['energy_level']:.1f}%")
    print(f"  Active Model: {report['active_model_index']}/{report['available_models']-1}")
    print(f"  Available Checkpoints: {len(report['checkpoints'])}")
    
    if report['recent_faults']:
        print("  Recent Faults:")
        for fault in report['recent_faults'][-3:]:  # Show last 3 faults
            print(f"    • {fault['type']}: {fault['message']}")
            print(f"      Recovery: {fault['recovery']} ({'✅' if fault['successful'] else '❌'})")


def test_integrated_robustness():
    """Test integrated security and fault tolerance."""
    print("\n\n🛡️  Testing Integrated Robustness")
    print("=" * 50)
    
    # Create system with both security and fault tolerance
    config = LiquidConfig(
        input_dim=4,
        hidden_dim=10,
        output_dim=2,
        energy_budget_mw=70.0
    )
    
    model = LiquidNN(config)
    key = jax.random.PRNGKey(42)
    params = model.init(key, jnp.ones((1, 4)))
    
    # Configure security
    security_config = SecurityConfig(
        enable_input_validation=True,
        enable_adversarial_detection=True,
        enable_model_integrity=True,
        adversarial_threshold=0.15
    )
    
    # Configure fault tolerance
    ft_config = FaultToleranceConfig(
        max_retries=2,
        enable_graceful_degradation=True,
        enable_checkpointing=True
    )
    
    # Create integrated system
    secure_model = SecureLiquidInference(model, security_config)
    secure_model.set_model_params(params)
    
    ft_system = FaultTolerantSystem(ft_config)
    ft_system.register_model(model, is_primary=True)
    
    def secure_fault_tolerant_inference(params, inputs):
        """Combined secure and fault-tolerant inference."""
        # First apply security checks
        valid, error_msg = secure_model.security_monitor.validate_input(inputs)
        if not valid:
            raise SecurityError(f"Security validation failed: {error_msg}")
        
        # Then apply fault tolerance
        return ft_system.fault_tolerant_inference(
            lambda p, x, **kw: secure_model(p, x, **kw),
            params, inputs
        )
    
    # Test with various challenging inputs
    test_cases = [
        ("Normal input", jnp.array([[0.1, 0.2, -0.1, 0.3]])),
        ("Large but valid input", jnp.array([[2.0, -1.5, 1.8, -2.2]])),
        ("Edge case input", jnp.array([[0.0, 0.0, 0.0, 0.0]])),
    ]
    
    print("🧪 Testing integrated robustness...")
    for test_name, test_input in test_cases:
        try:
            result = secure_fault_tolerant_inference(params, test_input)
            print(f"  ✅ {test_name}: Success, output: {result[0][:2]}")
        except SecurityError as e:
            print(f"  🛡️  {test_name}: Security block - {e}")
        except Exception as e:
            print(f"  ⚠️  {test_name}: System error - {e}")
    
    # Generate combined report
    print("\n📊 Integrated Robustness Summary:")
    security_report = secure_model.get_security_report()
    fault_report = ft_system.get_fault_report()
    
    print(f"  Security Score: {security_report['security_status']['security_score']:.1f}/100")
    print(f"  Fault Recovery Rate: {fault_report['recovery_success_rate']:.1f}%")
    print(f"  Total Security Events: {security_report['security_status']['total_events']}")
    print(f"  Total System Faults: {fault_report['total_faults']}")
    print(f"  System State: {fault_report['system_state']}")
    
    # Calculate overall robustness score
    robustness_score = (
        security_report['security_status']['security_score'] * 0.6 +
        fault_report['recovery_success_rate'] * 0.4
    )
    
    print(f"\n🏆 Overall Robustness Score: {robustness_score:.1f}/100")
    
    if robustness_score >= 90:
        print("  🌟 Excellent - Ready for critical production deployment")
    elif robustness_score >= 80:
        print("  ✅ Good - Suitable for most production environments")
    elif robustness_score >= 70:
        print("  ⚠️  Fair - Consider additional hardening before deployment")
    else:
        print("  ❌ Poor - Significant improvements needed before production use")


if __name__ == "__main__":
    print("🚀 Liquid Edge LLN - Advanced Robustness Demonstration")
    print("=" * 60)
    
    try:
        test_security_features()
        test_fault_tolerance()
        test_integrated_robustness()
        
        print("\n\n✅ All robustness tests completed successfully!")
        print("\n🔗 Next Steps:")
        print("  1. Deploy to edge device: liquid-lln deploy --target esp32s3")
        print("  2. Monitor in production: liquid-lln monitor --security --fault-tolerance")
        print("  3. Configure alerts: liquid-lln alerts setup --critical")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()