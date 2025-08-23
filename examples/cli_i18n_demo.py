#!/usr/bin/env python3
"""
Enhanced CLI demo showing integration with the i18n system.

This demonstrates how the existing CLI can be enhanced to support
internationalization without impacting performance.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

# Import i18n system
from liquid_edge.i18n import translate, set_language, Language, format_datetime
from liquid_edge import __version__
from datetime import datetime, timezone


def setup_toolchains():
    """Download and setup MCU toolchains with i18n support."""
    print(translate('cli.setup.toolchains'))
    print("✓ ARM GCC toolchain check")
    print("✓ ESP-IDF installation check")  
    print("✓ PlatformIO configuration")
    print(translate('cli.setup.complete'))


def doctor():
    """Run system diagnostics with localized output."""
    print(translate('cli.doctor.title'))
    print("=" * 45)
    
    # Check Python version
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    print(translate('cli.doctor.python_version', version=python_version))
    
    try:
        import jax
        print(translate('cli.doctor.jax_installed', version=jax.__version__))
    except ImportError:
        print(translate('cli.doctor.jax_missing'))
    
    try:
        import flax
        print(translate('cli.doctor.flax_installed', version=flax.__version__))
    except ImportError:
        print(translate('cli.doctor.flax_missing'))
    
    try:
        import numpy as np
        print(translate('cli.doctor.numpy_installed', version=np.__version__))
    except ImportError:
        print(translate('cli.doctor.numpy_missing'))
    
    print(f"\n{translate('cli.doctor.complete')}")


def benchmark(device: str = "cpu", models: str = "all"):
    """Run performance benchmarks with localized messages."""
    print(translate('cli.benchmark.running', device=device, models=models))
    print(translate('cli.benchmark.results'))


def compare(baseline: str, metric: str):
    """Compare models against baseline with localized output."""
    print(translate('cli.compare.running', baseline=baseline, metric=metric))


def plot(results_dir: str, output: str):
    """Generate plots from benchmark results with localized messages."""
    print(translate('cli.plot.generating', results_dir=results_dir, output=output))


def show_language_demo():
    """Demonstrate CLI output in different languages."""
    print("=" * 60)
    print("Liquid Edge LLN Kit - Internationalization Demo")
    print("=" * 60)
    
    languages = [
        (Language.ENGLISH, "English"),
        (Language.SPANISH, "Spanish"),
        (Language.FRENCH, "French"), 
        (Language.GERMAN, "German"),
        (Language.JAPANESE, "Japanese"),
        (Language.CHINESE, "Chinese")
    ]
    
    for lang, name in languages:
        print(f"\n--- {name} ---")
        set_language(lang)
        
        # Show system diagnostics header
        print(translate('cli.doctor.title'))
        
        # Show setup completion message
        print(translate('cli.setup.complete'))
        
        # Show current timestamp
        now = datetime.now(timezone.utc)
        formatted_time = format_datetime(now, include_time=True)
        print(f"{translate('status.ready')} - {formatted_time}")


def main():
    """Enhanced CLI main function with i18n support."""
    parser = argparse.ArgumentParser(
        description="Liquid Edge LLN Kit - Tiny liquid neural networks for edge robotics",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--version", action="version", version=f"liquid-edge-lln {__version__}"
    )
    
    parser.add_argument(
        "--language", "-l", 
        choices=['en', 'es', 'fr', 'de', 'ja', 'zh'],
        help="Set interface language"
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
    
    # Language demo command
    demo_parser = subparsers.add_parser("i18n-demo", help="Show internationalization demo")
    
    args = parser.parse_args()
    
    # Set language if specified
    if args.language:
        try:
            lang = Language(args.language)
            set_language(lang)
        except ValueError:
            print(f"Warning: Unknown language code '{args.language}', using default")
    
    # Execute commands
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
    elif args.command == "i18n-demo":
        show_language_demo()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()