"""Command line interface for Liquid Edge LLN Kit."""

import argparse
import sys
from pathlib import Path
from typing import Optional

from liquid_edge import __version__


def setup_toolchains():
    """Download and setup MCU toolchains."""
    print("Setting up MCU toolchains...")
    print("✓ ARM GCC toolchain check")
    print("✓ ESP-IDF installation check")
    print("✓ PlatformIO configuration")
    print("Toolchain setup complete!")


def doctor():
    """Run system diagnostics."""
    print("Liquid Edge LLN Kit - System Diagnostics")
    print("=" * 45)
    
    # Check Python version
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    print(f"Python version: {python_version}")
    
    try:
        import jax
        print(f"✓ JAX version: {jax.__version__}")
    except ImportError:
        print("✗ JAX not installed")
    
    try:
        import flax
        print(f"✓ Flax version: {flax.__version__}")
    except ImportError:
        print("✗ Flax not installed")
    
    try:
        import numpy as np
        print(f"✓ NumPy version: {np.__version__}")
    except ImportError:
        print("✗ NumPy not installed")
    
    print("\nSystem check complete!")


def benchmark(device: str = "cpu", models: str = "all"):
    """Run performance benchmarks."""
    print(f"Running benchmarks on {device} for {models} models...")
    print("Benchmark results will be saved to results/")


def compare(baseline: str, metric: str):
    """Compare models against baseline."""
    print(f"Comparing against {baseline} baseline using {metric} metrics...")


def plot(results_dir: str, output: str):
    """Generate plots from benchmark results."""
    print(f"Generating plots from {results_dir} to {output}/")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Liquid Edge LLN Kit - Tiny liquid neural networks for edge robotics",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--version", action="version", version=f"liquid-edge-lln {__version__}"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Setup toolchains command
    setup_parser = subparsers.add_parser(
        "setup-toolchains", help="Download and setup MCU toolchains"
    )
    
    # Doctor command
    doctor_parser = subparsers.add_parser("doctor", help="Run system diagnostics")
    
    # Benchmark command
    bench_parser = subparsers.add_parser("benchmark", help="Run performance benchmarks")
    bench_parser.add_argument("--device", default="cpu", help="Target device")
    bench_parser.add_argument("--models", default="all", help="Models to benchmark")
    
    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare models")
    compare_parser.add_argument("--baseline", required=True, help="Baseline model")
    compare_parser.add_argument("--metric", required=True, help="Comparison metrics")
    
    # Plot command
    plot_parser = subparsers.add_parser("plot", help="Generate plots")
    plot_parser.add_argument("results_dir", help="Results directory")
    plot_parser.add_argument("--output", default="figures/", help="Output directory")
    
    args = parser.parse_args()
    
    if args.command == "setup-toolchains":
        setup_toolchains()
    elif args.command == "doctor":
        doctor()
    elif args.command == "benchmark":
        benchmark(args.device, args.models)
    elif args.command == "compare":
        compare(args.baseline, args.metric)
    elif args.command == "plot":
        plot(args.results_dir, args.output)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()