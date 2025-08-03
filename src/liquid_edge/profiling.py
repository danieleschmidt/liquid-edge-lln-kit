"""Energy profiling and optimization tools for liquid neural networks."""

from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, field
import time
import numpy as np
import matplotlib.pyplot as plt
from contextlib import contextmanager
import json
import os
from pathlib import Path


@dataclass
class EnergyMeasurement:
    """Single energy measurement result."""
    name: str
    energy_mj: float
    power_mw: float
    duration_ms: float
    operations: int
    energy_per_op_nj: float
    timestamp: float = field(default_factory=time.time)


@dataclass 
class ProfilingConfig:
    """Configuration for energy profiling."""
    device: str = "esp32s3"
    voltage: float = 3.3
    sampling_rate: int = 1000  # Hz
    baseline_current_ma: float = 50.0
    enable_hardware_measurement: bool = False
    measurement_device: str = "power_profiler_kit"  # or "ina226", "custom"
    

class EnergyProfiler:
    """Profile energy consumption of liquid neural networks."""
    
    def __init__(self, config: ProfilingConfig):
        self.config = config
        self.measurements: List[EnergyMeasurement] = []
        self.device_specs = self._get_device_specs()
        self.current_measurement = None
        
    def _get_device_specs(self) -> Dict[str, Any]:
        """Get energy characteristics for target device."""
        specs = {
            "esp32s3": {
                "voltage": 3.3,
                "active_current_ma": 120,
                "sleep_current_ua": 10,
                "cpu_freq_mhz": 240,
                "ram_size_kb": 512,
                "flash_size_mb": 8,
                "operations_per_cycle": 1.0,
                "nj_per_cycle": 0.8
            },
            "stm32h7": {
                "voltage": 3.3,
                "active_current_ma": 280,
                "sleep_current_ua": 3,
                "cpu_freq_mhz": 400,
                "ram_size_kb": 1024,
                "flash_size_mb": 2,
                "operations_per_cycle": 1.2,
                "nj_per_cycle": 0.6
            },
            "nrf52840": {
                "voltage": 3.0,
                "active_current_ma": 15,
                "sleep_current_ua": 1.5,
                "cpu_freq_mhz": 64,
                "ram_size_kb": 256,
                "flash_size_mb": 1,
                "operations_per_cycle": 0.8,
                "nj_per_cycle": 1.2
            }
        }
        return specs.get(self.config.device, specs["esp32s3"])
    
    @contextmanager
    def measure(self, measurement_name: str, estimated_operations: int = 1000):
        """Context manager for energy measurement."""
        print(f"Starting measurement: {measurement_name}")
        
        start_time = time.time()
        start_energy = self._get_baseline_energy()
        
        try:
            yield
        finally:
            end_time = time.time()
            end_energy = self._get_current_energy()
            
            duration_ms = (end_time - start_time) * 1000
            energy_consumed_mj = end_energy - start_energy
            avg_power_mw = energy_consumed_mj / (duration_ms / 1000) if duration_ms > 0 else 0
            energy_per_op_nj = (energy_consumed_mj * 1e6) / estimated_operations if estimated_operations > 0 else 0
            
            measurement = EnergyMeasurement(
                name=measurement_name,
                energy_mj=energy_consumed_mj,
                power_mw=avg_power_mw,
                duration_ms=duration_ms,
                operations=estimated_operations,
                energy_per_op_nj=energy_per_op_nj
            )
            
            self.measurements.append(measurement)
            print(f"Completed: {measurement_name} - {energy_consumed_mj:.3f}mJ, {avg_power_mw:.1f}mW")
    
    def _get_baseline_energy(self) -> float:
        """Get baseline energy consumption."""
        if self.config.enable_hardware_measurement:
            return self._read_hardware_energy()
        else:
            # Simulated baseline based on device specs
            baseline_power_mw = (
                self.device_specs["active_current_ma"] * 
                self.device_specs["voltage"]
            )
            return baseline_power_mw * (time.time() % 1000)  # Simulated accumulation
    
    def _get_current_energy(self) -> float:
        """Get current energy consumption."""
        if self.config.enable_hardware_measurement:
            return self._read_hardware_energy()
        else:
            # Simulated measurement
            baseline_power_mw = (
                self.device_specs["active_current_ma"] * 
                self.device_specs["voltage"]
            )
            return baseline_power_mw * (time.time() % 1000)
    
    def _read_hardware_energy(self) -> float:
        """Read energy from hardware measurement device."""
        # This would interface with actual measurement hardware
        # For now, return simulated data
        if self.config.measurement_device == "power_profiler_kit":
            # Nordic Power Profiler Kit II integration
            pass
        elif self.config.measurement_device == "ina226":
            # INA226 current sensor integration
            pass
        
        # Simulated hardware reading
        return np.random.normal(100, 5)  # mJ
    
    def get_energy_mj(self) -> float:
        """Get last measured energy in millijoules."""
        if self.measurements:
            return self.measurements[-1].energy_mj
        return 0.0
    
    def get_average_power_mw(self) -> float:
        """Get average power consumption across all measurements."""
        if not self.measurements:
            return 0.0
        return np.mean([m.power_mw for m in self.measurements])
    
    def compare_measurements(self, names: List[str]) -> Dict[str, Dict[str, float]]:
        """Compare energy measurements by name."""
        comparison = {}
        
        for name in names:
            matching = [m for m in self.measurements if m.name == name]
            if matching:
                energies = [m.energy_mj for m in matching]
                powers = [m.power_mw for m in matching]
                
                comparison[name] = {
                    "avg_energy_mj": np.mean(energies),
                    "std_energy_mj": np.std(energies),
                    "avg_power_mw": np.mean(powers),
                    "std_power_mw": np.std(powers),
                    "count": len(matching)
                }
        
        return comparison
    
    def plot_comparison(self, save_path: Optional[str] = None):
        """Plot energy comparison across measurements."""
        if not self.measurements:
            print("No measurements to plot")
            return
        
        # Group measurements by name
        grouped = {}
        for m in self.measurements:
            if m.name not in grouped:
                grouped[m.name] = []
            grouped[m.name].append(m)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Energy comparison
        names = list(grouped.keys())
        energies = [np.mean([m.energy_mj for m in grouped[name]]) for name in names]
        energy_stds = [np.std([m.energy_mj for m in grouped[name]]) for name in names]
        
        bars1 = ax1.bar(names, energies, yerr=energy_stds, capsize=5)
        ax1.set_ylabel('Energy (mJ)')
        ax1.set_title('Energy Consumption Comparison')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for i, bar in enumerate(bars1):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + energy_stds[i],
                    f'{energies[i]:.2f}mJ', ha='center', va='bottom')
        
        # Power comparison
        powers = [np.mean([m.power_mw for m in grouped[name]]) for name in names]
        power_stds = [np.std([m.power_mw for m in grouped[name]]) for name in names]
        
        bars2 = ax2.bar(names, powers, yerr=power_stds, capsize=5, color='orange')
        ax2.set_ylabel('Average Power (mW)')
        ax2.set_title('Power Consumption Comparison')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for i, bar in enumerate(bars2):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + power_stds[i],
                    f'{powers[i]:.1f}mW', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        else:
            plt.show()
    
    def export_report(self, report_path: str):
        """Export detailed energy report."""
        report_data = {
            "device": self.config.device,
            "device_specs": self.device_specs,
            "total_measurements": len(self.measurements),
            "summary": self._generate_summary(),
            "measurements": [
                {
                    "name": m.name,
                    "energy_mj": m.energy_mj,
                    "power_mw": m.power_mw,
                    "duration_ms": m.duration_ms,
                    "operations": m.operations,
                    "energy_per_op_nj": m.energy_per_op_nj,
                    "timestamp": m.timestamp
                }
                for m in self.measurements
            ]
        }
        
        # Save JSON report
        json_path = report_path.replace('.pdf', '.json')
        with open(json_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"Energy report exported to: {json_path}")
        
        # Generate PDF report if requested
        if report_path.endswith('.pdf'):
            self._generate_pdf_report(report_data, report_path)
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate measurement summary statistics."""
        if not self.measurements:
            return {}
        
        energies = [m.energy_mj for m in self.measurements]
        powers = [m.power_mw for m in self.measurements]
        durations = [m.duration_ms for m in self.measurements]
        
        return {
            "total_energy_mj": sum(energies),
            "avg_energy_mj": np.mean(energies),
            "min_energy_mj": min(energies),
            "max_energy_mj": max(energies),
            "avg_power_mw": np.mean(powers),
            "min_power_mw": min(powers),
            "max_power_mw": max(powers),
            "total_duration_ms": sum(durations),
            "avg_duration_ms": np.mean(durations)
        }
    
    def _generate_pdf_report(self, report_data: Dict[str, Any], pdf_path: str):
        """Generate PDF report with matplotlib."""
        try:
            from matplotlib.backends.backend_pdf import PdfPages
            
            with PdfPages(pdf_path) as pdf:
                # Create summary page
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(11, 8))
                
                # Energy over time
                timestamps = [m["timestamp"] for m in report_data["measurements"]]
                energies = [m["energy_mj"] for m in report_data["measurements"]]
                
                ax1.plot(timestamps, energies, 'b-o')
                ax1.set_ylabel('Energy (mJ)')
                ax1.set_title('Energy Consumption Over Time')
                ax1.grid(True)
                
                # Power distribution
                powers = [m["power_mw"] for m in report_data["measurements"]]
                ax2.hist(powers, bins=20, alpha=0.7, color='orange')
                ax2.set_xlabel('Power (mW)')
                ax2.set_ylabel('Frequency')
                ax2.set_title('Power Distribution')
                ax2.grid(True)
                
                # Energy efficiency (energy per operation)
                energy_per_ops = [m["energy_per_op_nj"] for m in report_data["measurements"]]
                measurement_names = [m["name"] for m in report_data["measurements"]]
                
                if len(set(measurement_names)) <= 10:  # Only if we have reasonable number of categories
                    unique_names = list(set(measurement_names))
                    avg_efficiency = []
                    for name in unique_names:
                        name_measurements = [m for m in report_data["measurements"] if m["name"] == name]
                        avg_efficiency.append(np.mean([m["energy_per_op_nj"] for m in name_measurements]))
                    
                    ax3.bar(unique_names, avg_efficiency)
                    ax3.set_ylabel('Energy per Operation (nJ)')
                    ax3.set_title('Energy Efficiency by Measurement')
                    ax3.tick_params(axis='x', rotation=45)
                else:
                    ax3.plot(energy_per_ops, 'g-o')
                    ax3.set_ylabel('Energy per Operation (nJ)')
                    ax3.set_title('Energy Efficiency Over Measurements')
                    ax3.grid(True)
                
                # Summary statistics table
                ax4.axis('off')
                summary = report_data["summary"]
                table_data = [
                    ["Metric", "Value"],
                    ["Total Energy", f"{summary.get('total_energy_mj', 0):.2f} mJ"],
                    ["Avg Energy", f"{summary.get('avg_energy_mj', 0):.2f} mJ"],
                    ["Avg Power", f"{summary.get('avg_power_mw', 0):.1f} mW"],
                    ["Total Duration", f"{summary.get('total_duration_ms', 0):.1f} ms"],
                    ["Measurements", str(report_data["total_measurements"])]
                ]
                
                table = ax4.table(cellText=table_data, loc='center', cellLoc='left')
                table.auto_set_font_size(False)
                table.set_fontsize(10)
                table.scale(1, 2)
                ax4.set_title('Summary Statistics')
                
                plt.suptitle(f'Energy Profiling Report - {report_data["device"].upper()}', fontsize=16)
                plt.tight_layout()
                pdf.savefig(fig, dpi=300)
                plt.close()
            
            print(f"PDF report generated: {pdf_path}")
            
        except ImportError:
            print("PDF generation requires matplotlib. Skipping PDF report.")


class ModelEnergyOptimizer:
    """Optimize liquid neural networks for energy efficiency."""
    
    def __init__(self, profiler: EnergyProfiler):
        self.profiler = profiler
        self.optimization_history = []
    
    def optimize_sparsity(self, 
                         model_fn,
                         sparsity_levels: List[float] = [0.0, 0.1, 0.3, 0.5, 0.7],
                         test_data: np.ndarray = None) -> Tuple[float, Dict[str, Any]]:
        """Find optimal sparsity level for energy efficiency."""
        print("Optimizing sparsity for energy efficiency...")
        
        results = {}
        
        for sparsity in sparsity_levels:
            print(f"Testing sparsity level: {sparsity:.1f}")
            
            # Create model with current sparsity
            model = model_fn(sparsity=sparsity)
            
            # Estimate operations count
            estimated_ops = self._estimate_operations(model, test_data)
            
            # Measure energy
            with self.profiler.measure(f"sparsity_{sparsity:.1f}", estimated_ops):
                if test_data is not None:
                    # Run inference on test data
                    for _ in range(10):  # Multiple runs for averaging
                        _ = model(test_data)
                else:
                    # Simulate computation
                    time.sleep(0.1)
            
            last_measurement = self.profiler.measurements[-1]
            results[sparsity] = {
                "energy_mj": last_measurement.energy_mj,
                "power_mw": last_measurement.power_mw,
                "energy_per_op_nj": last_measurement.energy_per_op_nj
            }
        
        # Find optimal sparsity (minimum energy per operation)
        optimal_sparsity = min(results.keys(), 
                              key=lambda s: results[s]["energy_per_op_nj"])
        
        optimization_result = {
            "optimal_sparsity": optimal_sparsity,
            "energy_savings": self._calculate_energy_savings(results, optimal_sparsity),
            "all_results": results
        }
        
        self.optimization_history.append(optimization_result)
        
        print(f"Optimal sparsity: {optimal_sparsity:.1f}")
        print(f"Energy savings: {optimization_result['energy_savings']:.1f}%")
        
        return optimal_sparsity, optimization_result
    
    def optimize_quantization(self,
                            model_fn,
                            quantization_levels: List[str] = ["float32", "int16", "int8"],
                            test_data: np.ndarray = None) -> Tuple[str, Dict[str, Any]]:
        """Find optimal quantization level."""
        print("Optimizing quantization for energy efficiency...")
        
        results = {}
        
        for quant_level in quantization_levels:
            print(f"Testing quantization: {quant_level}")
            
            model = model_fn(quantization=quant_level)
            estimated_ops = self._estimate_operations(model, test_data)
            
            with self.profiler.measure(f"quant_{quant_level}", estimated_ops):
                if test_data is not None:
                    for _ in range(10):
                        _ = model(test_data)
                else:
                    time.sleep(0.1)
            
            last_measurement = self.profiler.measurements[-1]
            results[quant_level] = {
                "energy_mj": last_measurement.energy_mj,
                "power_mw": last_measurement.power_mw,
                "energy_per_op_nj": last_measurement.energy_per_op_nj
            }
        
        optimal_quantization = min(results.keys(),
                                  key=lambda q: results[q]["energy_per_op_nj"])
        
        optimization_result = {
            "optimal_quantization": optimal_quantization,
            "energy_savings": self._calculate_energy_savings(results, optimal_quantization),
            "all_results": results
        }
        
        print(f"Optimal quantization: {optimal_quantization}")
        print(f"Energy savings: {optimization_result['energy_savings']:.1f}%")
        
        return optimal_quantization, optimization_result
    
    def _estimate_operations(self, model, test_data: Optional[np.ndarray]) -> int:
        """Estimate number of operations for a model."""
        if hasattr(model, 'energy_estimate'):
            # Use model's built-in energy estimation
            return int(model.energy_estimate() * 1000)  # Convert to operation count estimate
        
        # Fallback estimation based on model size
        param_count = 0
        if hasattr(model, 'count_params'):
            param_count = model.count_params()
        
        # Rough estimate: 2 ops per parameter (multiply + add)
        return max(param_count * 2, 1000)
    
    def _calculate_energy_savings(self, results: Dict, optimal_key) -> float:
        """Calculate energy savings percentage."""
        if not results:
            return 0.0
        
        baseline_energy = max(results[key]["energy_per_op_nj"] for key in results.keys())
        optimal_energy = results[optimal_key]["energy_per_op_nj"]
        
        if baseline_energy > 0:
            savings = ((baseline_energy - optimal_energy) / baseline_energy) * 100
            return max(0.0, savings)
        
        return 0.0
    
    def generate_optimization_report(self, save_path: str):
        """Generate comprehensive optimization report."""
        if not self.optimization_history:
            print("No optimization results to report")
            return
        
        report_data = {
            "optimization_runs": len(self.optimization_history),
            "profiler_config": {
                "device": self.profiler.config.device,
                "voltage": self.profiler.config.voltage
            },
            "optimizations": self.optimization_history,
            "summary": self._generate_optimization_summary()
        }
        
        with open(save_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"Optimization report saved to: {save_path}")
    
    def _generate_optimization_summary(self) -> Dict[str, Any]:
        """Generate summary of all optimizations."""
        if not self.optimization_history:
            return {}
        
        total_savings = []
        for opt in self.optimization_history:
            total_savings.append(opt.get("energy_savings", 0.0))
        
        return {
            "avg_energy_savings": np.mean(total_savings),
            "max_energy_savings": max(total_savings),
            "min_energy_savings": min(total_savings),
            "total_optimizations": len(self.optimization_history)
        }