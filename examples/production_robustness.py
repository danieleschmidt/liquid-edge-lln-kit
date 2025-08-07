#!/usr/bin/env python3
"""Production-grade robustness example with monitoring and error handling."""

import jax
import jax.numpy as jnp
import numpy as np
import time
import threading
from typing import Dict, Any, Optional
import signal
import sys
from contextlib import contextmanager

from liquid_edge import LiquidNN, LiquidConfig
from liquid_edge.monitoring import (
    LiquidNetworkMonitor, PerformanceMetrics, AlertLevel, 
    CircuitBreaker, create_monitor
)
from liquid_edge.error_handling import (
    RobustErrorHandler, ModelInferenceError, SensorTimeoutError,
    EnergyBudgetExceededError, ErrorSeverity, retry_with_backoff,
    graceful_degradation, validate_inputs, create_error_handler,
    attach_error_handler, error_boundary
)


class ProductionLiquidController:
    """Production-grade liquid neural network controller with full robustness."""
    
    def __init__(self, config: LiquidConfig, monitor_name: str = "production_controller"):
        self.config = config
        
        # Initialize monitoring system
        self.monitor = create_monitor(
            name=monitor_name,
            enable_prometheus=True,
            enable_opentelemetry=True,
            prometheus_port=8000
        )
        
        # Initialize error handling
        self.error_handler = create_error_handler(monitor_name)
        
        # Create model
        self.model = LiquidNN(config)
        
        # Initialize model parameters
        key = jax.random.PRNGKey(42)
        dummy_input = jnp.ones((1, config.input_dim))
        self.params = self.model.init(key, dummy_input, training=False)
        
        # Controller state
        self.hidden_state = jnp.zeros((1, config.hidden_dim))
        self.last_sensor_data = None
        self.last_sensor_timestamp = 0.0
        self.is_running = False
        
        # Performance tracking
        self.inference_count = 0
        self.error_count = 0
        
        # Circuit breaker for inference
        self.inference_breaker = CircuitBreaker(
            failure_threshold=5,
            timeout_seconds=30.0,
            expected_exception=ModelInferenceError
        )
        
        # Setup custom recovery strategies
        self._setup_recovery_strategies()
        
        # Setup alert callbacks
        self._setup_alert_callbacks()
        
        # Setup graceful shutdown
        self._setup_graceful_shutdown()
        
        self.monitor.alert(
            AlertLevel.INFO,
            f"Production controller initialized with {config.hidden_dim} neurons",
            "controller",
            {"config": str(config)}
        )
    
    def _setup_recovery_strategies(self):
        """Setup custom error recovery strategies."""
        
        def sensor_timeout_recovery(error_context):
            """Recovery for sensor timeouts - use last known data."""
            if self.last_sensor_data is not None:
                self.monitor.alert(
                    AlertLevel.WARNING,
                    "Using last known sensor data due to timeout",
                    "sensor_recovery"
                )
                return self.last_sensor_data
            else:
                # No previous data, use safe defaults
                return jnp.zeros(self.config.input_dim)
        
        def inference_error_recovery(error_context):
            """Recovery for inference errors - return safe control commands."""
            self.monitor.alert(
                AlertLevel.ERROR,
                "Inference failed, using safe stop commands",
                "inference_recovery"
            )
            # Return safe stop commands
            return jnp.zeros(self.config.output_dim)
        
        def energy_budget_recovery(error_context):
            """Recovery for energy budget exceeded - reduce performance."""
            self.monitor.alert(
                AlertLevel.CRITICAL,
                "Energy budget exceeded, entering low-power mode",
                "energy_recovery"
            )
            # Trigger performance reduction (in real system)
            return None
        
        # Register strategies
        self.error_handler.register_recovery_strategy(SensorTimeoutError, sensor_timeout_recovery)
        self.error_handler.register_recovery_strategy(ModelInferenceError, inference_error_recovery)
        self.error_handler.register_recovery_strategy(EnergyBudgetExceededError, energy_budget_recovery)
    
    def _setup_alert_callbacks(self):
        """Setup alert callback handlers."""
        
        def critical_alert_handler(alert):
            """Handle critical alerts."""
            if alert.level == AlertLevel.CRITICAL:
                print(f"\n🚨 CRITICAL ALERT: {alert.message}")
                print(f"   Component: {alert.component}")
                print(f"   Metadata: {alert.metadata}")
                
                # In production, could trigger pager/SMS/Slack notification
        
        def performance_alert_handler(alert):
            """Handle performance-related alerts."""
            if "energy" in alert.message.lower() or "performance" in alert.message.lower():
                print(f"⚡ Performance Alert: {alert.message}")
        
        self.monitor.add_alert_callback(critical_alert_handler)
        self.monitor.add_alert_callback(performance_alert_handler)
    
    def _setup_graceful_shutdown(self):
        """Setup graceful shutdown handlers."""
        
        def signal_handler(signum, frame):
            print(f"\n🛑 Received signal {signum}, shutting down gracefully...")
            self.shutdown()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    @retry_with_backoff(max_retries=3, backoff_factor=1.5, severity=ErrorSeverity.MEDIUM)
    @validate_inputs(sensor_data=lambda x: len(x) == 8, timeout_s=lambda x: x > 0)
    def process_sensors(self, sensor_data: jnp.ndarray, timeout_s: float = 0.1) -> jnp.ndarray:
        """Process sensor data with validation and error handling."""
        current_time = time.time()
        
        # Check for stale data
        if self.last_sensor_timestamp > 0:
            data_age = current_time - self.last_sensor_timestamp
            if data_age > timeout_s:
                raise SensorTimeoutError(
                    f"Sensor data is {data_age:.3f}s old (timeout: {timeout_s}s)",
                    severity=ErrorSeverity.HIGH,
                    context={"data_age": data_age, "timeout": timeout_s}
                )
        
        # Validate sensor ranges (0-5m for distance sensors)
        if jnp.any(sensor_data < 0) or jnp.any(sensor_data > 5.0):
            raise ValueError(f"Sensor values out of range [0, 5.0]: {sensor_data}")
        
        # Store last known good data
        self.last_sensor_data = sensor_data
        self.last_sensor_timestamp = current_time
        
        return sensor_data
    
    @graceful_degradation(fallback_value=jnp.zeros(2), severity=ErrorSeverity.HIGH)  
    def run_inference(self, processed_sensors: jnp.ndarray) -> jnp.ndarray:
        """Run liquid neural network inference with circuit breaker protection."""
        
        def _inference():
            # Check energy budget before inference
            estimated_energy = self.model.energy_estimate()
            if estimated_energy > self.config.energy_budget_mw * 1.2:  # 20% tolerance
                raise EnergyBudgetExceededError(
                    f"Estimated energy {estimated_energy:.1f}mW exceeds budget {self.config.energy_budget_mw}mW",
                    severity=ErrorSeverity.CRITICAL,
                    context={"estimated_energy": estimated_energy, "budget": self.config.energy_budget_mw}
                )
            
            # Run inference
            try:
                output, new_hidden = self.model.apply(
                    self.params, 
                    processed_sensors.reshape(1, -1),
                    self.hidden_state,
                    training=False
                )
                
                # Update hidden state
                self.hidden_state = new_hidden
                
                return output.flatten()
                
            except Exception as e:
                raise ModelInferenceError(
                    f"Inference failed: {str(e)}",
                    severity=ErrorSeverity.HIGH,
                    context={"exception_type": type(e).__name__}
                )
        
        # Apply circuit breaker protection
        return self.inference_breaker(_inference)()
    
    def control_step(self, sensor_data: jnp.ndarray) -> Dict[str, Any]:
        """Execute one control step with full monitoring and error handling."""
        
        with self.monitor.track_inference("production_controller"):
            start_time = time.time()
            
            try:
                # Process sensors with error handling
                processed_sensors = self.process_sensors(sensor_data, timeout_s=0.1)
                
                # Run inference with protection
                motor_commands = self.run_inference(processed_sensors)
                
                # Validate outputs
                motor_commands = self._validate_motor_commands(motor_commands)
                
                # Update metrics
                inference_time_us = (time.time() - start_time) * 1e6
                energy_estimate = self.model.energy_estimate()
                
                metrics = PerformanceMetrics(
                    inference_time_us=inference_time_us,
                    energy_consumption_mw=energy_estimate,
                    throughput_fps=1.0 / max(time.time() - start_time, 0.001)
                )
                
                self.monitor.record_metrics(metrics)
                self.inference_count += 1
                
                return {
                    "success": True,
                    "motor_commands": motor_commands,
                    "inference_time_us": inference_time_us,
                    "energy_mw": energy_estimate,
                    "circuit_breaker_state": self.inference_breaker.state,
                    "hidden_state_norm": float(jnp.linalg.norm(self.hidden_state))
                }
                
            except Exception as e:
                self.error_count += 1
                
                # Use error boundary for recovery
                with error_boundary(self.error_handler, severity=ErrorSeverity.HIGH, reraise=False):
                    raise e
                
                # Return safe fallback
                return {
                    "success": False,
                    "motor_commands": jnp.zeros(self.config.output_dim),
                    "error": str(e),
                    "circuit_breaker_state": self.inference_breaker.state,
                    "recovery_used": True
                }
    
    def _validate_motor_commands(self, commands: jnp.ndarray) -> jnp.ndarray:
        """Validate and clamp motor commands to safe ranges."""
        # Clamp linear velocity to [-1, 1] m/s
        # Clamp angular velocity to [-2, 2] rad/s
        safe_commands = jnp.array([
            jnp.clip(commands[0], -1.0, 1.0),  # Linear velocity
            jnp.clip(commands[1], -2.0, 2.0)   # Angular velocity
        ])
        
        # Check for NaN/inf values
        if jnp.any(~jnp.isfinite(safe_commands)):
            self.monitor.alert(
                AlertLevel.ERROR,
                f"Invalid motor commands detected: {commands}",
                "command_validation"
            )
            return jnp.zeros(self.config.output_dim)
        
        return safe_commands
    
    def run_continuous(self, duration_seconds: float = 60.0, control_frequency_hz: float = 50.0):
        """Run controller continuously with monitoring."""
        
        self.is_running = True
        control_period = 1.0 / control_frequency_hz
        end_time = time.time() + duration_seconds
        
        print(f"\n🚀 Starting production controller for {duration_seconds}s at {control_frequency_hz}Hz")
        print(f"   Monitoring: Prometheus on port {self.monitor.prometheus_port}")
        print(f"   Circuit Breaker: {self.inference_breaker.state}")
        print(f"   Health Status: {self.monitor.get_health_status().value}")
        
        cycle_count = 0
        
        try:
            while self.is_running and time.time() < end_time:
                cycle_start = time.time()
                
                # Simulate sensor data (in production, read from actual sensors)
                sensor_data = self._generate_mock_sensor_data(cycle_count)
                
                # Execute control step
                result = self.control_step(sensor_data)
                
                # Print status every 50 cycles (1 second at 50Hz)
                if cycle_count % 50 == 0:
                    status = "✅ OK" if result["success"] else "❌ ERROR"
                    print(f"   Cycle {cycle_count:4d}: {status} | "
                          f"Energy: {result.get('energy_mw', 0):.1f}mW | "
                          f"Latency: {result.get('inference_time_us', 0):.0f}µs | "
                          f"CB: {result.get('circuit_breaker_state', 'UNKNOWN')}")
                
                cycle_count += 1
                
                # Maintain control frequency
                elapsed = time.time() - cycle_start
                if elapsed < control_period:
                    time.sleep(control_period - elapsed)
                else:
                    self.monitor.alert(
                        AlertLevel.WARNING,
                        f"Control loop overrun: {elapsed:.3f}s > {control_period:.3f}s",
                        "timing"
                    )
        
        except KeyboardInterrupt:
            print("\n⏹️  Interrupted by user")
        
        finally:
            self.is_running = False
            self._print_final_statistics()
    
    def _generate_mock_sensor_data(self, cycle: int) -> jnp.ndarray:
        """Generate mock sensor data for demonstration."""
        # Simulate 8-direction distance sensors with some patterns
        t = cycle * 0.02  # 50Hz -> 0.02s per cycle
        
        # Base distances with some dynamic obstacles
        distances = jnp.array([
            2.0 + 0.5 * jnp.sin(t * 0.1),      # Front
            3.0 + 0.3 * jnp.cos(t * 0.15),     # Front-right
            2.5 + 0.4 * jnp.sin(t * 0.12),     # Right
            2.8 + 0.2 * jnp.cos(t * 0.18),     # Back-right
            3.2 + 0.1 * jnp.sin(t * 0.08),     # Back
            2.9 + 0.3 * jnp.cos(t * 0.14),     # Back-left
            2.6 + 0.2 * jnp.sin(t * 0.16),     # Left
            2.7 + 0.4 * jnp.cos(t * 0.11),     # Front-left
        ])
        
        # Add some noise
        noise = 0.05 * jax.random.normal(jax.random.PRNGKey(cycle), (8,))
        
        # Occasional sensor timeout simulation (5% chance)
        if cycle % 200 == 199:  # Every 4 seconds
            # Simulate sensor timeout by making data very old
            time.sleep(0.15)  # Cause timeout
        
        # Occasional bad sensor data (1% chance)
        if cycle % 1000 == 999:
            distances = distances.at[0].set(-1.0)  # Invalid negative distance
        
        return distances + noise
    
    def _print_final_statistics(self):
        """Print final statistics and monitoring data."""
        print("\n📊 Final Statistics:")
        
        # Controller statistics
        error_rate = self.error_count / max(self.inference_count, 1) * 100
        print(f"   Total inferences: {self.inference_count}")
        print(f"   Total errors: {self.error_count}")
        print(f"   Error rate: {error_rate:.2f}%")
        print(f"   Circuit breaker state: {self.inference_breaker.state}")
        print(f"   Final health status: {self.monitor.get_health_status().value}")
        
        # Monitoring statistics
        monitor_stats = self.monitor.get_statistics()
        print(f"   Uptime: {monitor_stats['uptime_seconds']:.1f}s")
        print(f"   Avg inference time: {monitor_stats.get('avg_inference_time_us', 0):.0f}µs")
        print(f"   Avg energy consumption: {monitor_stats.get('avg_energy_consumption_mw', 0):.1f}mW")
        
        # Error handler statistics
        error_stats = self.error_handler.get_error_statistics()
        print(f"   Error types encountered: {error_stats['error_types']}")
        print(f"   Recent errors (5min): {error_stats['recent_errors_5min']}")
        
        if error_stats['most_common_errors']:
            print(f"   Most common error: {error_stats['most_common_errors'][0][0]}")
    
    def shutdown(self):
        """Graceful shutdown of the controller."""
        print("\n🔄 Shutting down production controller...")
        
        self.is_running = False
        
        # Export monitoring data
        try:
            self.monitor.export_metrics("results/production_monitoring_export.json")
            print("   📊 Monitoring data exported")
        except Exception as e:
            print(f"   ❌ Failed to export monitoring data: {e}")
        
        # Stop monitoring
        self.monitor.stop_monitoring()
        
        print("   ✅ Controller shutdown complete")


def main():
    """Demonstrate production-grade robustness."""
    print("🏭 Liquid Edge LLN - Production Robustness Demo")
    print("=" * 55)
    print("Demonstrating: Monitoring | Error Handling | Circuit Breakers | Graceful Degradation")
    
    # Configuration for a small but robust controller
    config = LiquidConfig(
        input_dim=8,           # 8-direction distance sensors
        hidden_dim=10,         # Compact liquid network
        output_dim=2,          # Linear and angular velocity
        tau_min=20.0,
        tau_max=80.0,
        use_sparse=True,
        sparsity=0.3,
        energy_budget_mw=50.0,  # Conservative budget
        target_fps=50
    )
    
    # Create production controller
    controller = ProductionLiquidController(config, "production_demo")
    
    # Attach error handler to decorated functions
    attach_error_handler(controller.error_handler)(controller.process_sensors)
    attach_error_handler(controller.error_handler)(controller.run_inference)
    
    print("\n🔧 Production Features Enabled:")
    print("   ✅ Comprehensive monitoring (Prometheus + OpenTelemetry)")
    print("   ✅ Robust error handling with recovery strategies")
    print("   ✅ Circuit breaker pattern for inference protection")
    print("   ✅ Input validation and graceful degradation")
    print("   ✅ Performance tracking and alerting")
    print("   ✅ Graceful shutdown handlers")
    
    print("\n🌐 Monitoring Endpoints:")
    print(f"   Prometheus metrics: http://localhost:8000/metrics")
    print(f"   Health check: Monitor health_status metric")
    
    print("\n🚀 Running production demo...")
    print("   Press Ctrl+C for graceful shutdown")
    
    # Run for 30 seconds
    try:
        controller.run_continuous(duration_seconds=30.0, control_frequency_hz=50.0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
    finally:
        controller.shutdown()
    
    print("\n🎯 Production Demo Complete!")
    print("\n📈 Key Robustness Features Demonstrated:")
    print("   🔄 Automatic error recovery")
    print("   ⚡ Circuit breaker protection")
    print("   📊 Real-time monitoring")
    print("   🛡️ Graceful degradation")
    print("   📋 Comprehensive logging")
    print("   🚨 Alert system")
    
    print("\n🏭 Ready for production deployment!")

if __name__ == "__main__":
    import os
    os.makedirs("results", exist_ok=True)
    main()
