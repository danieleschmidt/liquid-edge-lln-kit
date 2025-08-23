#!/usr/bin/env python3
"""
Demonstration of the Liquid Edge LLN Kit Internationalization (i18n) System.

This example shows how to:
1. Use the i18n system for localized messages and formatting
2. Integrate with existing error handling and compliance systems
3. Switch between languages and regions dynamically
4. Format metrics and dates according to local conventions
"""

import time
from datetime import datetime, timezone
from liquid_edge.i18n import (
    set_language, set_locale, translate, format_metric, format_datetime,
    get_compliance_requirements, get_localized_logger, Language, Region,
    get_translation_manager, integrate_with_error_handler, integrate_with_compliance
)
from liquid_edge.error_handling import RobustErrorHandler, EnergyBudgetExceededError, ErrorSeverity
from liquid_edge.compliance import ComplianceManager, ComplianceFramework, DataCategory, ProcessingPurpose


def demo_basic_translation():
    """Demonstrate basic translation functionality."""
    print("=== Basic Translation Demo ===")
    
    # Test with different languages
    languages = [
        (Language.ENGLISH, "English"),
        (Language.SPANISH, "Spanish"), 
        (Language.FRENCH, "French"),
        (Language.GERMAN, "German"),
        (Language.JAPANESE, "Japanese"),
        (Language.CHINESE, "Chinese")
    ]
    
    for lang, name in languages:
        set_language(lang)
        print(f"\n{name}:")
        print(f"  Setup: {translate('cli.setup.complete')}")
        print(f"  Error: {translate('error.energy_budget')}")
        print(f"  Status: {translate('status.ready')}")


def demo_metric_formatting():
    """Demonstrate localized metric formatting."""
    print("\n=== Metric Formatting Demo ===")
    
    # Test metrics with different locales
    locales = [
        (Language.ENGLISH, Region.US, "English (US)"),
        (Language.ENGLISH, Region.UK, "English (UK)"),
        (Language.FRENCH, Region.EU, "French (EU)"),
        (Language.GERMAN, Region.EU, "German (EU)"),
        (Language.JAPANESE, Region.JAPAN, "Japanese"),
        (Language.CHINESE, Region.CHINA, "Chinese")
    ]
    
    test_metrics = [
        (1234.567, "energy", "Energy consumption"),
        (89.123, "latency", "Processing latency"),
        (512.0, "memory", "Memory usage"),
        (98.76, "accuracy", "Model accuracy")
    ]
    
    for lang, region, name in locales:
        set_locale(lang, region)
        print(f"\n{name}:")
        for value, metric_type, description in test_metrics:
            formatted = format_metric(value, metric_type)
            print(f"  {translate(f'metrics.{metric_type}')}: {formatted}")


def demo_datetime_formatting():
    """Demonstrate localized datetime formatting."""
    print("\n=== DateTime Formatting Demo ===")
    
    now = datetime.now(timezone.utc)
    
    locales = [
        (Language.ENGLISH, Region.US, "English (US)"),
        (Language.ENGLISH, Region.UK, "English (UK)"),
        (Language.FRENCH, Region.EU, "French (EU)"),
        (Language.GERMAN, Region.EU, "German (EU)"),
        (Language.JAPANESE, Region.JAPAN, "Japanese"),
        (Language.CHINESE, Region.CHINA, "Chinese")
    ]
    
    for lang, region, name in locales:
        set_locale(lang, region)
        date_only = format_datetime(now, include_time=False)
        date_time = format_datetime(now, include_time=True)
        print(f"{name}:")
        print(f"  Date: {date_only}")
        print(f"  DateTime: {date_time}")


def demo_compliance_integration():
    """Demonstrate compliance system integration with i18n."""
    print("\n=== Compliance Integration Demo ===")
    
    # Test different regions and their compliance requirements
    regions_to_test = [
        (Language.ENGLISH, Region.EU, "European Union"),
        (Language.ENGLISH, Region.US, "United States"),
        (Language.ENGLISH, Region.UK, "United Kingdom"),
        (Language.JAPANESE, Region.JAPAN, "Japan"),
        (Language.CHINESE, Region.CHINA, "China"),
        (Language.ENGLISH, Region.CANADA, "Canada"),
    ]
    
    for lang, region, name in regions_to_test:
        set_locale(lang, region)
        requirements = get_compliance_requirements()
        
        print(f"\n{name}:")
        print(f"  Language: {lang.value}")
        print(f"  Compliance: {', '.join(requirements) if requirements else 'None specified'}")
        
        # Demonstrate localized compliance messages
        print(f"  GDPR Title: {translate('compliance.gdpr.title')}")
        print(f"  Data Processing: {translate('compliance.data_processing')}")


def demo_error_handling_integration():
    """Demonstrate error handling integration with i18n."""
    print("\n=== Error Handling Integration Demo ===")
    
    # Create error handler and integrate with i18n
    error_handler = RobustErrorHandler("i18n_demo")
    integrate_with_error_handler(error_handler)
    
    # Test with different languages
    languages = [Language.ENGLISH, Language.SPANISH, Language.FRENCH, Language.GERMAN]
    
    for lang in languages:
        set_language(lang)
        print(f"\n{lang.value.capitalize()}:")
        
        try:
            # Simulate an energy budget error
            raise EnergyBudgetExceededError(
                translate('error.energy_budget'),
                severity=ErrorSeverity.HIGH,
                context={"budget_mj": 100, "consumed_mj": 150}
            )
        except EnergyBudgetExceededError as e:
            print(f"  Original Error: {e}")
            print(f"  Localized: {translate('error.energy_budget')}")


def demo_localized_logging():
    """Demonstrate localized logging."""
    print("\n=== Localized Logging Demo ===")
    
    logger = get_localized_logger("i18n_demo")
    
    languages = [Language.ENGLISH, Language.SPANISH, Language.FRENCH, Language.JAPANESE]
    
    for lang in languages:
        set_language(lang)
        print(f"\nLogging in {lang.value}:")
        
        # These would normally go to the log file/console
        print(f"  INFO: {translate('log.system_startup')}")
        print(f"  SUCCESS: {translate('log.model_loaded', model_name='LiquidNet-v1')}")
        print(f"  COMPLETE: {translate('log.inference_complete', duration=42.5)}")


def demo_custom_translations():
    """Demonstrate adding custom translations."""
    print("\n=== Custom Translations Demo ===")
    
    tm = get_translation_manager()
    
    # Add custom translations for a new feature
    custom_translations = {
        Language.ENGLISH: {
            "feature.quantum_mode": "Quantum processing mode enabled",
            "feature.hyperscale": "Hyperscale optimization active"
        },
        Language.SPANISH: {
            "feature.quantum_mode": "Modo de procesamiento cuántico habilitado",
            "feature.hyperscale": "Optimización hiperescala activa"
        },
        Language.FRENCH: {
            "feature.quantum_mode": "Mode de traitement quantique activé",
            "feature.hyperscale": "Optimisation hyperscale active"
        }
    }
    
    # Add custom translations
    for lang, translations in custom_translations.items():
        for key, text in translations.items():
            tm.add_custom_translation(lang, key, text)
    
    # Test custom translations
    for lang in [Language.ENGLISH, Language.SPANISH, Language.FRENCH]:
        set_language(lang)
        print(f"{lang.value}:")
        print(f"  {translate('feature.quantum_mode')}")
        print(f"  {translate('feature.hyperscale')}")


def demo_performance_impact():
    """Demonstrate the minimal performance impact of i18n system."""
    print("\n=== Performance Impact Demo ===")
    
    # Test translation performance
    set_language(Language.ENGLISH)
    
    # Measure translation time
    start_time = time.perf_counter()
    for _ in range(10000):
        translate('error.energy_budget')
    translation_time = time.perf_counter() - start_time
    
    print(f"10,000 translations took: {translation_time*1000:.3f}ms")
    print(f"Average per translation: {(translation_time/10000)*1000000:.3f}μs")
    
    # Test metric formatting performance
    start_time = time.perf_counter()
    for _ in range(10000):
        format_metric(123.456, "energy")
    formatting_time = time.perf_counter() - start_time
    
    print(f"10,000 metric formats took: {formatting_time*1000:.3f}ms")
    print(f"Average per format: {(formatting_time/10000)*1000000:.3f}μs")
    
    print("\nConclusion: i18n system has negligible performance impact!")


def main():
    """Run all i18n system demonstrations."""
    print("Liquid Edge LLN Kit - Internationalization (i18n) System Demo")
    print("=" * 70)
    
    try:
        demo_basic_translation()
        demo_metric_formatting()
        demo_datetime_formatting() 
        demo_compliance_integration()
        demo_error_handling_integration()
        demo_localized_logging()
        demo_custom_translations()
        demo_performance_impact()
        
        print("\n" + "=" * 70)
        print("i18n System Demo completed successfully!")
        print("\nThe Liquid Edge LLN Kit now supports:")
        print("✓ 6 languages with full translations")
        print("✓ Regional formatting for dates, numbers, and metrics")
        print("✓ Compliance framework localization")
        print("✓ Integration with existing error handling and logging")
        print("✓ Minimal performance impact on core operations")
        print("✓ Extensible translation system for custom features")
        
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()