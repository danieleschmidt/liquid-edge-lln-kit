# Performance Testing Guide

This guide covers performance testing strategies for the Liquid Edge LLN Kit, focusing on energy efficiency, inference speed, and memory usage.

## Overview

Performance testing is critical for edge AI applications where resources are severely constrained. Our performance testing strategy covers:

- **Energy Consumption**: Primary optimization target
- **Inference Latency**: Real-time requirements
- **Memory Usage**: Embedded constraints
- **Accuracy Preservation**: Quality vs. efficiency trade-offs

## Energy Performance Testing

### Measurement Strategies

#### 1. Hardware-Based Measurement
```python
from liquid_edge.profiling import EnergyProfiler

# Real hardware measurement with INA219
profiler = EnergyProfiler(
    device="esp32s3",
    interface="ina219",
    sampling_rate=1000  # Hz
)

with profiler.measure("inference"):
    output = model.apply(params, input_data)
    
energy_mj = profiler.get_energy_mj()
power_mw = profiler.get_average_power_mw()
```

#### 2. Simulation-Based Estimation
```python
from liquid_edge.simulation import EnergySimulator

# Platform-specific energy models
simulator = EnergySimulator(platform="stm32h743")

# Estimate energy from operation counts
ops_profile = model.profile_operations(input_data)
estimated_energy = simulator.estimate_energy(ops_profile)
```

### Benchmark Suite

#### Standard Benchmarks
```python
@pytest.mark.benchmark
def test_inference_energy_benchmark(benchmark, liquid_model):
    """Benchmark inference energy consumption."""
    model, params = liquid_model
    input_data = jnp.ones((1, 4))
    
    def inference():
        return model.apply(params, input_data)
    
    result = benchmark(inference)
    
    # Energy performance assertions
    assert result.stats.mean < 0.1  # <100mW average
    assert result.stats.max < 0.2   # <200mW peak
```

#### Comparative Benchmarks
```python
def test_energy_comparison_vs_dense_network():
    """Compare liquid network vs dense network energy."""
    # Train equivalent models
    liquid_model = train_liquid_network(dataset)
    dense_model = train_dense_network(dataset)
    
    # Measure energy
    liquid_energy = measure_energy(liquid_model, test_data)
    dense_energy = measure_energy(dense_model, test_data)
    
    # Assert 5x improvement
    assert liquid_energy < dense_energy / 5.0
```

## Latency Performance Testing

### Real-Time Requirements
```python
@pytest.mark.benchmark
def test_inference_latency(benchmark, liquid_model):
    """Test inference meets real-time requirements."""
    model, params = liquid_model
    
    def inference():
        return model.apply(params, jnp.ones((1, 4)))
    
    result = benchmark.pedantic(inference, rounds=1000)
    
    # Real-time assertions (100Hz control loop)
    assert result.stats.mean < 0.010  # <10ms mean
    assert result.stats.max < 0.015   # <15ms worst case
    assert result.stats.stddev < 0.002  # <2ms jitter
```

### Hardware-Specific Latency
```python
@pytest.mark.hardware
def test_mcu_inference_latency():
    """Test inference latency on actual MCU."""
    with MCUConnection(port="/dev/ttyUSB0") as mcu:
        # Flash model to MCU
        mcu.flash_model("test_model.bin")
        
        # Measure inference timing
        latencies = []
        for _ in range(1000):
            start_time = mcu.get_timestamp()
            mcu.run_inference()
            end_time = mcu.get_timestamp()
            latencies.append(end_time - start_time)
        
        mean_latency = np.mean(latencies)
        max_latency = np.max(latencies)
        
        assert mean_latency < 10.0  # <10ms on 168MHz MCU
        assert max_latency < 15.0   # <15ms worst case
```

## Memory Performance Testing

### Static Memory Analysis
```python
def test_model_memory_footprint():
    """Test model fits in target memory constraints."""
    model = create_liquid_model(config)
    params = model.init(PRNGKey(42), jnp.ones((1, 4)))
    
    # Calculate memory requirements
    param_memory = calculate_parameter_memory(params)
    activation_memory = calculate_activation_memory(model, input_shape=(1, 4))
    total_memory = param_memory + activation_memory
    
    # Assert fits in 256KB constraint
    assert total_memory < 256 * 1024  # 256KB limit
```

### Dynamic Memory Profiling
```python
@pytest.mark.benchmark
def test_memory_usage_during_inference(memory_profiler):
    """Profile memory usage during inference."""
    model, params = create_test_model()
    
    with memory_profiler.profile("inference"):
        for _ in range(100):
            output = model.apply(params, random_input())
    
    peak_memory = memory_profiler.get_peak_memory_mb()
    assert peak_memory < 32  # <32MB peak during inference
```

## Accuracy Performance Testing

### Accuracy vs Efficiency Trade-offs
```python
def test_quantization_accuracy_degradation():
    """Test accuracy degradation from quantization."""
    # Full precision baseline
    fp32_model, fp32_params = train_liquid_model(dataset, precision="fp32")
    fp32_accuracy = evaluate_model(fp32_model, fp32_params, test_dataset)
    
    # Quantized model
    int8_model, int8_params = quantize_model(fp32_model, fp32_params, "int8")
    int8_accuracy = evaluate_model(int8_model, int8_params, test_dataset)
    
    # Assert <3% accuracy degradation
    accuracy_loss = fp32_accuracy - int8_accuracy
    assert accuracy_loss < 0.03
```

### Energy-Accuracy Pareto Frontier
```python
def test_energy_accuracy_pareto_optimal():
    """Test model is Pareto optimal in energy-accuracy space."""
    configurations = [
        {"sparsity": 0.1, "quantization": "fp16"},
        {"sparsity": 0.3, "quantization": "int8"},
        {"sparsity": 0.5, "quantization": "int8"},
        {"sparsity": 0.7, "quantization": "int4"},
    ]
    
    results = []
    for config in configurations:
        model = train_liquid_model(dataset, **config)
        accuracy = evaluate_accuracy(model)
        energy = measure_energy(model)
        results.append((accuracy, energy, config))
    
    # Verify Pareto optimality
    for i, (acc_i, eng_i, cfg_i) in enumerate(results):
        for j, (acc_j, eng_j, cfg_j) in enumerate(results):
            if i != j:
                # No configuration should dominate another
                dominates = (acc_i >= acc_j and eng_i <= eng_j and 
                           (acc_i > acc_j or eng_i < eng_j))
                if dominates:
                    pytest.fail(f"Config {cfg_i} dominates {cfg_j}")
```

## Regression Testing

### Performance Regression Detection
```python
@pytest.mark.benchmark
def test_energy_regression():
    """Detect energy consumption regressions."""
    current_energy = measure_model_energy()
    
    # Load baseline from previous version
    baseline_energy = load_baseline_energy("v0.1.0")
    
    # Allow 5% regression tolerance
    regression_threshold = baseline_energy * 1.05
    assert current_energy <= regression_threshold, \
        f"Energy regression: {current_energy} > {regression_threshold}"
```

### Continuous Performance Monitoring
```python
def test_performance_trend_analysis():
    """Analyze performance trends over time."""
    # Load historical performance data
    history = load_performance_history()
    
    # Fit trend line
    from scipy import stats
    times = [p.timestamp for p in history]
    energies = [p.energy for p in history]
    slope, intercept, r_value, p_value, std_err = stats.linregress(times, energies)
    
    # Ensure performance is improving (negative slope)
    assert slope <= 0, f"Performance degrading over time: slope={slope}"
```

## Benchmark Data Management

### Baseline Storage
```python
# Save benchmark results as baselines
def save_benchmark_baseline(results, version):
    """Save benchmark results for future comparison."""
    baseline_path = f"benchmarks/baselines/{version}.json"
    with open(baseline_path, 'w') as f:
        json.dump({
            'version': version,
            'timestamp': datetime.now().isoformat(),
            'energy_mw': results.energy_mw,
            'latency_ms': results.latency_ms,
            'memory_kb': results.memory_kb,
            'accuracy': results.accuracy,
            'platform': results.platform
        }, f)
```

### Performance Reports
```python
def generate_performance_report():
    """Generate comprehensive performance report."""
    report = PerformanceReport()
    
    # Current results
    current = run_all_benchmarks()
    report.add_current_results(current)
    
    # Historical comparison
    history = load_benchmark_history()
    report.add_trend_analysis(history)
    
    # Hardware comparison
    platforms = ["stm32h743", "esp32s3", "nrf52840"]
    for platform in platforms:
        results = run_platform_benchmarks(platform)
        report.add_platform_results(platform, results)
    
    # Generate report
    report.save_html("performance_report.html")
    report.save_pdf("performance_report.pdf")
```

## CI/CD Integration

### Performance Gates
```yaml
# .github/workflows/performance.yml
name: Performance Tests
on: [push, pull_request]

jobs:
  performance:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run Performance Tests
        run: |
          pytest tests/benchmarks/ --benchmark-only
          pytest tests/performance/ --performance-gate
      
      - name: Check Performance Regression
        run: |
          python scripts/check_performance_regression.py \
            --baseline-version ${{ github.base_ref }} \
            --threshold 0.05
```

### Hardware Testing in CI
```yaml
# Hardware test job (requires self-hosted runner with hardware)
hardware-performance:
  runs-on: [self-hosted, hardware]
  steps:
    - name: Setup Hardware
      run: |
        scripts/setup_hardware.sh
        scripts/verify_connections.sh
    
    - name: Run Hardware Benchmarks
      run: |
        pytest tests/hardware/ --hardware-config ci-hardware.yaml
        
    - name: Upload Results
      uses: actions/upload-artifact@v3
      with:
        name: hardware-benchmark-results
        path: benchmark-results/
```

This performance testing framework ensures that the Liquid Edge LLN Kit maintains its energy efficiency promise while delivering the required accuracy and real-time performance for edge robotics applications.