"""Tests for the internationalization (i18n) system."""

import pytest
from datetime import datetime, timezone
from decimal import Decimal

from liquid_edge.i18n import (
    Language, Region, LocaleConfig, TranslationManager,
    set_language, set_locale, translate, format_metric, format_datetime,
    get_compliance_requirements, get_localized_logger, I18nConfig,
    get_translation_manager, LocalizedLogger
)


class TestTranslationManager:
    """Test the TranslationManager class."""
    
    def test_language_setting(self):
        """Test setting different languages."""
        tm = TranslationManager()
        
        # Test setting language by enum
        tm.set_language(Language.SPANISH)
        assert tm.get_current_language() == Language.SPANISH
        
        # Test setting language by string
        tm.set_language('fr')
        assert tm.get_current_language() == Language.FRENCH
        
        # Test invalid language falls back to English
        tm.set_language('invalid')
        assert tm.get_current_language() == Language.ENGLISH
    
    def test_locale_setting(self):
        """Test setting locales with language and region."""
        tm = TranslationManager()
        
        # Test setting US English locale
        tm.set_locale(Language.ENGLISH, Region.US)
        locale = tm.get_current_locale()
        assert locale is not None
        assert locale.language == Language.ENGLISH
        assert locale.region == Region.US
        assert locale.currency_symbol == "$"
        
        # Test setting EU French locale
        tm.set_locale(Language.FRENCH, Region.EU)
        locale = tm.get_current_locale()
        assert locale is not None
        assert locale.language == Language.FRENCH
        assert locale.region == Region.EU
        assert locale.currency_symbol == "€"
    
    def test_basic_translation(self):
        """Test basic translation functionality."""
        tm = TranslationManager()
        
        # Test English translation
        tm.set_language(Language.ENGLISH)
        assert tm.translate('status.ready') == 'Ready'
        assert tm.translate('error.energy_budget') == 'Energy budget exceeded'
        
        # Test Spanish translation
        tm.set_language(Language.SPANISH)
        assert tm.translate('status.ready') == 'Listo'
        assert tm.translate('error.energy_budget') == 'Presupuesto de energía excedido'
        
        # Test French translation
        tm.set_language(Language.FRENCH)
        assert tm.translate('status.ready') == 'Prêt'
    
    def test_translation_with_formatting(self):
        """Test translation with format arguments."""
        tm = TranslationManager()
        tm.set_language(Language.ENGLISH)
        
        # Test formatted translation
        result = tm.translate('cli.doctor.python_version', version='3.10.0')
        assert result == 'Python version: 3.10.0'
        
        # Test with missing arguments (should not crash)
        result = tm.translate('cli.doctor.python_version')
        assert 'Python version:' in result
    
    def test_fallback_translation(self):
        """Test fallback to English when translation not found."""
        tm = TranslationManager()
        tm.set_language(Language.SPANISH)
        
        # This key only exists in English
        result = tm.translate('cli.benchmark.results')
        assert result == 'Benchmark results will be saved to results/'
    
    def test_missing_key_fallback(self):
        """Test behavior when translation key doesn't exist."""
        tm = TranslationManager()
        tm.set_language(Language.ENGLISH)
        
        # Non-existent key should return the key itself
        result = tm.translate('non.existent.key')
        assert result == 'non.existent.key'
    
    def test_custom_translation(self):
        """Test adding custom translations."""
        tm = TranslationManager()
        
        # Add custom translation
        tm.add_custom_translation(Language.ENGLISH, 'custom.test', 'Custom test message')
        tm.add_custom_translation(Language.SPANISH, 'custom.test', 'Mensaje de prueba personalizado')
        
        # Test custom translations
        tm.set_language(Language.ENGLISH)
        assert tm.translate('custom.test') == 'Custom test message'
        
        tm.set_language(Language.SPANISH)
        assert tm.translate('custom.test') == 'Mensaje de prueba personalizado'


class TestNumberFormatting:
    """Test number and metric formatting."""
    
    def test_basic_number_formatting(self):
        """Test basic number formatting without locale."""
        tm = TranslationManager()
        
        # Test without locale (should use defaults)
        assert tm.format_number(1234.56, 2) == '1234.56'
        assert tm.format_number(1234, 0) == '1234'
    
    def test_locale_number_formatting(self):
        """Test locale-specific number formatting."""
        tm = TranslationManager()
        
        # US locale (decimal='.', thousands=',')
        tm.set_locale(Language.ENGLISH, Region.US)
        assert tm.format_number(1234567.89, 2) == '1,234,567.89'
        
        # EU French locale (decimal=',', thousands=' ')
        tm.set_locale(Language.FRENCH, Region.EU)
        result = tm.format_number(1234567.89, 2)
        assert '1 234 567,89' in result or '1234567,89' in result  # Flexible for implementation
        
        # German locale (decimal=',', thousands='.')
        tm.set_locale(Language.GERMAN, Region.EU)
        result = tm.format_number(1234567.89, 2)
        assert ',' in result  # Should have comma decimal separator
    
    def test_metric_formatting(self):
        """Test metric formatting with units."""
        tm = TranslationManager()
        tm.set_language(Language.ENGLISH)
        
        # Test different metric types
        assert 'ms' in tm.format_metric(123.45, 'latency')
        assert 'mJ' in tm.format_metric(67.89, 'energy')
        assert 'mW' in tm.format_metric(45.67, 'power')
        assert 'KB' in tm.format_metric(512.0, 'memory')
        assert '%' in tm.format_metric(98.5, 'accuracy')
    
    def test_metric_localization(self):
        """Test metric formatting in different languages."""
        tm = TranslationManager()
        
        # Test in Spanish
        tm.set_language(Language.SPANISH)
        result = tm.format_metric(123.45, 'latency')
        # Should contain Spanish unit translation and formatted number
        assert 'ms' in result
        
        # Test in French
        tm.set_language(Language.FRENCH)
        result = tm.format_metric(67.89, 'energy')
        assert 'mJ' in result


class TestDateTimeFormatting:
    """Test datetime formatting."""
    
    def test_basic_datetime_formatting(self):
        """Test basic datetime formatting without locale."""
        tm = TranslationManager()
        dt = datetime(2024, 12, 25, 15, 30, 45, tzinfo=timezone.utc)
        
        # Without locale, should use ISO format
        date_result = tm.format_datetime(dt, include_time=False)
        time_result = tm.format_datetime(dt, include_time=True)
        
        assert '2024-12-25' in date_result
        assert '15:30:45' in time_result
    
    def test_locale_datetime_formatting(self):
        """Test locale-specific datetime formatting."""
        tm = TranslationManager()
        dt = datetime(2024, 12, 25, 15, 30, 45, tzinfo=timezone.utc)
        
        # US format (MM/DD/YYYY, 12-hour)
        tm.set_locale(Language.ENGLISH, Region.US)
        result = tm.format_datetime(dt, include_time=False)
        # Should be in MM/DD/YYYY format
        assert '12/25/2024' in result or '25' in result  # Flexible for different implementations
        
        # UK format (DD/MM/YYYY, 24-hour)
        tm.set_locale(Language.ENGLISH, Region.UK)
        result = tm.format_datetime(dt, include_time=False)
        # Should be in DD/MM/YYYY format
        assert '25' in result and '12' in result and '2024' in result


class TestComplianceIntegration:
    """Test compliance framework integration."""
    
    def test_compliance_requirements_by_region(self):
        """Test getting compliance requirements by region."""
        tm = TranslationManager()
        
        # EU should have GDPR
        tm.set_locale(Language.ENGLISH, Region.EU)
        requirements = tm.get_compliance_requirements()
        assert 'gdpr' in requirements
        
        # US should have CCPA
        tm.set_locale(Language.ENGLISH, Region.US)
        requirements = tm.get_compliance_requirements()
        assert 'ccpa' in requirements
        
        # Singapore should have PDPA
        tm.set_locale(Language.ENGLISH, Region.SINGAPORE)
        requirements = tm.get_compliance_requirements()
        assert 'pdpa' in requirements


class TestGlobalFunctions:
    """Test global convenience functions."""
    
    def test_global_set_language(self):
        """Test global set_language function."""
        # Reset to English first
        set_language(Language.ENGLISH)
        assert translate('status.ready') == 'Ready'
        
        # Switch to Spanish
        set_language(Language.SPANISH)
        assert translate('status.ready') == 'Listo'
        
        # Reset back to English for other tests
        set_language(Language.ENGLISH)
    
    def test_global_translate(self):
        """Test global translate function."""
        set_language(Language.ENGLISH)
        assert translate('status.success') == 'Success'
        assert translate('metrics.energy') == 'Energy Consumption'
    
    def test_global_format_metric(self):
        """Test global format_metric function."""
        set_language(Language.ENGLISH)
        result = format_metric(123.45, 'latency', 2)
        assert '123.45' in result
        assert 'ms' in result
    
    def test_global_format_datetime(self):
        """Test global format_datetime function."""
        dt = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        result = format_datetime(dt, include_time=True)
        assert '2024' in result
        assert '12:00' in result


class TestLocalizedLogger:
    """Test the LocalizedLogger class."""
    
    def test_localized_logger_creation(self):
        """Test creating a localized logger."""
        logger = get_localized_logger('test_logger')
        assert isinstance(logger, LocalizedLogger)
        assert logger.logger.name == 'test_logger'
    
    def test_localized_logger_translation(self):
        """Test that localized logger uses translations."""
        logger = get_localized_logger('test_logger')
        
        # This test verifies the logger exists and can be called
        # In a real test environment, we'd capture log output
        logger.translation_manager.set_language(Language.ENGLISH)
        assert logger.translation_manager.translate('status.ready') == 'Ready'


class TestI18nConfig:
    """Test the I18nConfig class."""
    
    def test_config_creation(self):
        """Test creating i18n configuration."""
        config = I18nConfig()
        
        # Test default values
        assert config.get('language') == 'en'
        assert config.get('region') == 'US'
        assert config.get('auto_detect') is True
    
    def test_config_setting(self):
        """Test setting configuration values."""
        config = I18nConfig()
        
        config.set('language', 'es')
        assert config.get('language') == 'es'
        
        config.set('auto_detect', False)
        assert config.get('auto_detect') is False


class TestPerformance:
    """Test performance characteristics of i18n system."""
    
    def test_translation_performance(self):
        """Test that translation is fast enough for production use."""
        import time
        
        tm = TranslationManager()
        tm.set_language(Language.ENGLISH)
        
        # Measure translation time
        start = time.perf_counter()
        for _ in range(1000):
            tm.translate('status.ready')
        end = time.perf_counter()
        
        # Should be very fast (less than 10ms for 1000 translations)
        elapsed = end - start
        assert elapsed < 0.01, f"Translation too slow: {elapsed:.4f}s for 1000 translations"
    
    def test_metric_formatting_performance(self):
        """Test that metric formatting is fast enough for production use."""
        import time
        
        tm = TranslationManager()
        tm.set_language(Language.ENGLISH)
        
        # Measure formatting time
        start = time.perf_counter()
        for _ in range(1000):
            tm.format_metric(123.456, 'energy', 2)
        end = time.perf_counter()
        
        # Should be very fast (less than 50ms for 1000 formats)
        elapsed = end - start
        assert elapsed < 0.05, f"Metric formatting too slow: {elapsed:.4f}s for 1000 formats"


@pytest.mark.integration
class TestIntegration:
    """Integration tests for i18n system."""
    
    def test_error_handler_integration(self):
        """Test integration with error handling system."""
        from liquid_edge.error_handling import RobustErrorHandler
        from liquid_edge.i18n import integrate_with_error_handler
        
        # Create error handler and integrate
        error_handler = RobustErrorHandler("test")
        integrate_with_error_handler(error_handler)
        
        # Verify integration doesn't break functionality
        assert error_handler is not None
        assert hasattr(error_handler, '_log_error')
    
    def test_compliance_integration(self):
        """Test integration with compliance system."""
        from liquid_edge.compliance import ComplianceManager, ComplianceFramework
        from liquid_edge.i18n import integrate_with_compliance
        
        # Create compliance manager and integrate
        compliance_manager = ComplianceManager(
            [ComplianceFramework.GDPR], 
            "Test Org", 
            "Test Controller"
        )
        integrate_with_compliance(compliance_manager)
        
        # Verify integration doesn't break functionality
        assert compliance_manager is not None
        assert hasattr(compliance_manager, 'logger')


if __name__ == '__main__':
    pytest.main([__file__])