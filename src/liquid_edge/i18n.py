"""
Comprehensive Internationalization (i18n) System for Liquid Edge LLN Kit.

This module provides:
- Translation support for English, Spanish, French, German, Japanese, and Chinese
- Automatic language detection from system locale
- Localized number formatting for metrics
- Date/time formatting for different regions
- Cultural adaptations for compliance requirements (GDPR, CCPA, PDPA)
- Integration with existing logging and CLI systems
"""

import locale
import os
import sys
import json
import logging
from typing import Dict, Any, Optional, List, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timezone
from decimal import Decimal
import threading
from pathlib import Path

# Thread-safe logger instance
_logger = logging.getLogger(__name__)


class Language(Enum):
    """Supported languages with their ISO codes."""
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    JAPANESE = "ja"
    CHINESE = "zh"


class Region(Enum):
    """Supported regions for compliance and formatting."""
    US = "US"          # United States
    EU = "EU"          # European Union
    UK = "GB"          # United Kingdom
    CANADA = "CA"      # Canada
    JAPAN = "JP"       # Japan
    CHINA = "CN"       # China
    SINGAPORE = "SG"   # Singapore
    AUSTRALIA = "AU"   # Australia
    BRAZIL = "BR"      # Brazil


@dataclass
class LocaleConfig:
    """Configuration for locale-specific formatting and compliance."""
    language: Language
    region: Region
    decimal_separator: str = "."
    thousands_separator: str = ","
    currency_symbol: str = "$"
    date_format: str = "%Y-%m-%d"
    time_format: str = "%H:%M:%S"
    energy_unit: str = "mJ"
    power_unit: str = "mW"
    temperature_unit: str = "°C"
    compliance_frameworks: List[str] = field(default_factory=list)
    rtl_support: bool = False  # Right-to-left text support
    
    def __post_init__(self):
        """Set region-specific defaults after initialization."""
        if not self.compliance_frameworks:
            self.compliance_frameworks = self._get_default_compliance()
    
    def _get_default_compliance(self) -> List[str]:
        """Get default compliance frameworks for region."""
        region_compliance = {
            Region.EU: ["gdpr"],
            Region.UK: ["gdpr", "uk_dpa"],
            Region.US: ["ccpa"],
            Region.CANADA: ["pipeda"],
            Region.SINGAPORE: ["pdpa"],
            Region.BRAZIL: ["lgpd"],
            Region.CHINA: ["pipl"],
            Region.JAPAN: ["appi"],
            Region.AUSTRALIA: ["privacy_act"]
        }
        return region_compliance.get(self.region, [])


# Default locale configurations for each supported region
_DEFAULT_LOCALES = {
    (Language.ENGLISH, Region.US): LocaleConfig(
        Language.ENGLISH, Region.US,
        decimal_separator=".", thousands_separator=",",
        currency_symbol="$", date_format="%m/%d/%Y",
        time_format="%I:%M:%S %p"
    ),
    (Language.ENGLISH, Region.UK): LocaleConfig(
        Language.ENGLISH, Region.UK,
        decimal_separator=".", thousands_separator=",",
        currency_symbol="£", date_format="%d/%m/%Y",
        time_format="%H:%M:%S"
    ),
    (Language.SPANISH, Region.US): LocaleConfig(
        Language.SPANISH, Region.US,
        decimal_separator=".", thousands_separator=",",
        currency_symbol="$", date_format="%d/%m/%Y",
        time_format="%H:%M:%S"
    ),
    (Language.FRENCH, Region.EU): LocaleConfig(
        Language.FRENCH, Region.EU,
        decimal_separator=",", thousands_separator=" ",
        currency_symbol="€", date_format="%d/%m/%Y",
        time_format="%H:%M:%S"
    ),
    (Language.GERMAN, Region.EU): LocaleConfig(
        Language.GERMAN, Region.EU,
        decimal_separator=",", thousands_separator=".",
        currency_symbol="€", date_format="%d.%m.%Y",
        time_format="%H:%M:%S"
    ),
    (Language.JAPANESE, Region.JAPAN): LocaleConfig(
        Language.JAPANESE, Region.JAPAN,
        decimal_separator=".", thousands_separator=",",
        currency_symbol="¥", date_format="%Y/%m/%d",
        time_format="%H:%M:%S"
    ),
    (Language.CHINESE, Region.CHINA): LocaleConfig(
        Language.CHINESE, Region.CHINA,
        decimal_separator=".", thousands_separator=",",
        currency_symbol="¥", date_format="%Y-%m-%d",
        time_format="%H:%M:%S"
    ),
}


class TranslationManager:
    """Manages translations and localization for the Liquid Edge LLN Kit."""
    
    def __init__(self):
        self._translations = {}
        self._current_language = Language.ENGLISH
        self._current_locale = None
        self._fallback_language = Language.ENGLISH
        self._lock = threading.RLock()
        
        # Load built-in translations
        self._load_builtin_translations()
        
        # Detect system language
        self._auto_detect_language()
    
    def _load_builtin_translations(self):
        """Load built-in translations for all supported languages."""
        
        # English (base language)
        self._translations[Language.ENGLISH] = {
            # CLI Messages
            "cli.setup.toolchains": "Setting up MCU toolchains...",
            "cli.setup.complete": "Toolchain setup complete!",
            "cli.doctor.title": "Liquid Edge LLN Kit - System Diagnostics",
            "cli.doctor.python_version": "Python version: {version}",
            "cli.doctor.jax_installed": "✓ JAX version: {version}",
            "cli.doctor.jax_missing": "✗ JAX not installed",
            "cli.doctor.flax_installed": "✓ Flax version: {version}",
            "cli.doctor.flax_missing": "✗ Flax not installed",
            "cli.doctor.numpy_installed": "✓ NumPy version: {version}",
            "cli.doctor.numpy_missing": "✗ NumPy not installed",
            "cli.doctor.complete": "System check complete!",
            "cli.benchmark.running": "Running benchmarks on {device} for {models} models...",
            "cli.benchmark.results": "Benchmark results will be saved to results/",
            "cli.compare.running": "Comparing against {baseline} baseline using {metric} metrics...",
            "cli.plot.generating": "Generating plots from {results_dir} to {output}/",
            
            # Error Messages
            "error.model_inference": "Model inference failed",
            "error.energy_budget": "Energy budget exceeded",
            "error.sensor_timeout": "Sensor data timeout",
            "error.deployment": "Model deployment failed",
            "error.configuration": "Configuration error",
            "error.resource_exhaustion": "System resources exhausted",
            "error.recovery_attempted": "Attempting recovery for {error_type}",
            "error.recovery_success": "Recovery successful for {error_type}",
            "error.recovery_failed": "Recovery failed for {error_type}: {details}",
            "error.no_recovery": "No recovery strategy found for {error_type}",
            
            # Metrics and Performance
            "metrics.latency": "Latency",
            "metrics.energy": "Energy Consumption",
            "metrics.power": "Power Consumption",
            "metrics.memory": "Memory Usage",
            "metrics.accuracy": "Accuracy",
            "metrics.temperature": "Temperature",
            "metrics.voltage": "Voltage",
            "metrics.current": "Current",
            
            # Units
            "unit.milliseconds": "ms",
            "unit.millijoules": "mJ",
            "unit.milliwatts": "mW",
            "unit.kilobytes": "KB",
            "unit.percentage": "%",
            "unit.celsius": "°C",
            "unit.volts": "V",
            "unit.milliamps": "mA",
            
            # Compliance Messages
            "compliance.gdpr.title": "GDPR Compliance",
            "compliance.ccpa.title": "CCPA Compliance",
            "compliance.pdpa.title": "PDPA Compliance",
            "compliance.data_processing": "Data processing activity recorded",
            "compliance.consent_obtained": "User consent obtained",
            "compliance.consent_withdrawn": "User consent withdrawn",
            "compliance.access_request": "Data access request processed",
            "compliance.deletion_request": "Data deletion request processed",
            "compliance.portability_request": "Data portability request processed",
            
            # Logging Messages
            "log.system_startup": "Liquid Edge LLN Kit starting up",
            "log.system_shutdown": "Liquid Edge LLN Kit shutting down",
            "log.model_loaded": "Model loaded successfully: {model_name}",
            "log.inference_start": "Starting inference",
            "log.inference_complete": "Inference completed in {duration}ms",
            "log.training_start": "Starting training",
            "log.training_epoch": "Epoch {epoch}: loss={loss:.4f}",
            "log.training_complete": "Training completed",
            
            # Status Messages
            "status.ready": "Ready",
            "status.running": "Running",
            "status.stopped": "Stopped",
            "status.error": "Error",
            "status.warning": "Warning",
            "status.success": "Success",
        }
        
        # Spanish translations
        self._translations[Language.SPANISH] = {
            "cli.setup.toolchains": "Configurando cadenas de herramientas MCU...",
            "cli.setup.complete": "¡Configuración de cadena de herramientas completa!",
            "cli.doctor.title": "Kit Liquid Edge LLN - Diagnósticos del Sistema",
            "cli.doctor.python_version": "Versión de Python: {version}",
            "cli.doctor.jax_installed": "✓ Versión JAX: {version}",
            "cli.doctor.jax_missing": "✗ JAX no instalado",
            "cli.doctor.flax_installed": "✓ Versión Flax: {version}",
            "cli.doctor.flax_missing": "✗ Flax no instalado",
            "cli.doctor.numpy_installed": "✓ Versión NumPy: {version}",
            "cli.doctor.numpy_missing": "✗ NumPy no instalado",
            "cli.doctor.complete": "¡Verificación del sistema completa!",
            
            "error.model_inference": "Falló la inferencia del modelo",
            "error.energy_budget": "Presupuesto de energía excedido",
            "error.sensor_timeout": "Tiempo de espera de datos del sensor agotado",
            "error.deployment": "Falló el despliegue del modelo",
            "error.configuration": "Error de configuración",
            "error.resource_exhaustion": "Recursos del sistema agotados",
            
            "metrics.latency": "Latencia",
            "metrics.energy": "Consumo de Energía",
            "metrics.power": "Consumo de Energía",
            "metrics.memory": "Uso de Memoria",
            "metrics.accuracy": "Precisión",
            "metrics.temperature": "Temperatura",
            
            "compliance.gdpr.title": "Cumplimiento GDPR",
            "compliance.ccpa.title": "Cumplimiento CCPA",
            "compliance.pdpa.title": "Cumplimiento PDPA",
            "compliance.data_processing": "Actividad de procesamiento de datos registrada",
            
            "status.ready": "Listo",
            "status.running": "Ejecutándose",
            "status.stopped": "Detenido",
            "status.error": "Error",
            "status.warning": "Advertencia",
            "status.success": "Éxito",
        }
        
        # French translations
        self._translations[Language.FRENCH] = {
            "cli.setup.toolchains": "Configuration des chaînes d'outils MCU...",
            "cli.setup.complete": "Configuration de la chaîne d'outils terminée !",
            "cli.doctor.title": "Kit Liquid Edge LLN - Diagnostics du Système",
            "cli.doctor.python_version": "Version Python : {version}",
            "cli.doctor.jax_installed": "✓ Version JAX : {version}",
            "cli.doctor.jax_missing": "✗ JAX non installé",
            "cli.doctor.flax_installed": "✓ Version Flax : {version}",
            "cli.doctor.flax_missing": "✗ Flax non installé",
            "cli.doctor.numpy_installed": "✓ Version NumPy : {version}",
            "cli.doctor.numpy_missing": "✗ NumPy non installé",
            "cli.doctor.complete": "Vérification du système terminée !",
            
            "error.model_inference": "Échec de l'inférence du modèle",
            "error.energy_budget": "Budget énergétique dépassé",
            "error.sensor_timeout": "Délai d'attente des données du capteur",
            "error.deployment": "Échec du déploiement du modèle",
            "error.configuration": "Erreur de configuration",
            "error.resource_exhaustion": "Ressources système épuisées",
            
            "metrics.latency": "Latence",
            "metrics.energy": "Consommation d'Énergie",
            "metrics.power": "Consommation d'Énergie",
            "metrics.memory": "Utilisation de la Mémoire",
            "metrics.accuracy": "Précision",
            "metrics.temperature": "Température",
            
            "compliance.gdpr.title": "Conformité RGPD",
            "compliance.ccpa.title": "Conformité CCPA",
            "compliance.pdpa.title": "Conformité PDPA",
            "compliance.data_processing": "Activité de traitement des données enregistrée",
            
            "status.ready": "Prêt",
            "status.running": "En cours",
            "status.stopped": "Arrêté",
            "status.error": "Erreur",
            "status.warning": "Avertissement",
            "status.success": "Succès",
        }
        
        # German translations
        self._translations[Language.GERMAN] = {
            "cli.setup.toolchains": "MCU-Toolchains werden eingerichtet...",
            "cli.setup.complete": "Toolchain-Setup abgeschlossen!",
            "cli.doctor.title": "Liquid Edge LLN Kit - Systemdiagnose",
            "cli.doctor.python_version": "Python-Version: {version}",
            "cli.doctor.jax_installed": "✓ JAX-Version: {version}",
            "cli.doctor.jax_missing": "✗ JAX nicht installiert",
            "cli.doctor.flax_installed": "✓ Flax-Version: {version}",
            "cli.doctor.flax_missing": "✗ Flax nicht installiert",
            "cli.doctor.numpy_installed": "✓ NumPy-Version: {version}",
            "cli.doctor.numpy_missing": "✗ NumPy nicht installiert",
            "cli.doctor.complete": "Systemprüfung abgeschlossen!",
            
            "error.model_inference": "Modellinferenz fehlgeschlagen",
            "error.energy_budget": "Energiebudget überschritten",
            "error.sensor_timeout": "Sensordaten-Timeout",
            "error.deployment": "Modellbereitstellung fehlgeschlagen",
            "error.configuration": "Konfigurationsfehler",
            "error.resource_exhaustion": "Systemressourcen erschöpft",
            
            "metrics.latency": "Latenz",
            "metrics.energy": "Energieverbrauch",
            "metrics.power": "Leistungsaufnahme",
            "metrics.memory": "Speichernutzung",
            "metrics.accuracy": "Genauigkeit",
            "metrics.temperature": "Temperatur",
            
            "compliance.gdpr.title": "DSGVO-Konformität",
            "compliance.ccpa.title": "CCPA-Konformität",
            "compliance.pdpa.title": "PDPA-Konformität",
            "compliance.data_processing": "Datenverarbeitungsaktivität erfasst",
            
            "status.ready": "Bereit",
            "status.running": "Läuft",
            "status.stopped": "Gestoppt",
            "status.error": "Fehler",
            "status.warning": "Warnung",
            "status.success": "Erfolg",
        }
        
        # Japanese translations
        self._translations[Language.JAPANESE] = {
            "cli.setup.toolchains": "MCUツールチェーンを設定中...",
            "cli.setup.complete": "ツールチェーンの設定が完了しました！",
            "cli.doctor.title": "Liquid Edge LLNキット - システム診断",
            "cli.doctor.python_version": "Pythonバージョン: {version}",
            "cli.doctor.jax_installed": "✓ JAXバージョン: {version}",
            "cli.doctor.jax_missing": "✗ JAXがインストールされていません",
            "cli.doctor.flax_installed": "✓ Flaxバージョン: {version}",
            "cli.doctor.flax_missing": "✗ Flaxがインストールされていません",
            "cli.doctor.numpy_installed": "✓ NumPyバージョン: {version}",
            "cli.doctor.numpy_missing": "✗ NumPyがインストールされていません",
            "cli.doctor.complete": "システムチェック完了！",
            
            "error.model_inference": "モデル推論に失敗しました",
            "error.energy_budget": "エネルギー予算を超過しました",
            "error.sensor_timeout": "センサーデータのタイムアウト",
            "error.deployment": "モデルのデプロイに失敗しました",
            "error.configuration": "設定エラー",
            "error.resource_exhaustion": "システムリソースが不足しています",
            
            "metrics.latency": "レイテンシ",
            "metrics.energy": "エネルギー消費",
            "metrics.power": "電力消費",
            "metrics.memory": "メモリ使用量",
            "metrics.accuracy": "精度",
            "metrics.temperature": "温度",
            
            "compliance.gdpr.title": "GDPR準拠",
            "compliance.ccpa.title": "CCPA準拠",
            "compliance.pdpa.title": "PDPA準拠",
            "compliance.data_processing": "データ処理活動が記録されました",
            
            "status.ready": "準備完了",
            "status.running": "実行中",
            "status.stopped": "停止",
            "status.error": "エラー",
            "status.warning": "警告",
            "status.success": "成功",
        }
        
        # Chinese (Simplified) translations
        self._translations[Language.CHINESE] = {
            "cli.setup.toolchains": "正在设置MCU工具链...",
            "cli.setup.complete": "工具链设置完成！",
            "cli.doctor.title": "Liquid Edge LLN套件 - 系统诊断",
            "cli.doctor.python_version": "Python版本: {version}",
            "cli.doctor.jax_installed": "✓ JAX版本: {version}",
            "cli.doctor.jax_missing": "✗ JAX未安装",
            "cli.doctor.flax_installed": "✓ Flax版本: {version}",
            "cli.doctor.flax_missing": "✗ Flax未安装",
            "cli.doctor.numpy_installed": "✓ NumPy版本: {version}",
            "cli.doctor.numpy_missing": "✗ NumPy未安装",
            "cli.doctor.complete": "系统检查完成！",
            
            "error.model_inference": "模型推理失败",
            "error.energy_budget": "能源预算超出",
            "error.sensor_timeout": "传感器数据超时",
            "error.deployment": "模型部署失败",
            "error.configuration": "配置错误",
            "error.resource_exhaustion": "系统资源耗尽",
            
            "metrics.latency": "延迟",
            "metrics.energy": "能耗",
            "metrics.power": "功耗",
            "metrics.memory": "内存使用",
            "metrics.accuracy": "准确性",
            "metrics.temperature": "温度",
            
            "compliance.gdpr.title": "GDPR合规性",
            "compliance.ccpa.title": "CCPA合规性",
            "compliance.pdpa.title": "PDPA合规性",
            "compliance.data_processing": "数据处理活动已记录",
            
            "status.ready": "就绪",
            "status.running": "运行中",
            "status.stopped": "已停止",
            "status.error": "错误",
            "status.warning": "警告",
            "status.success": "成功",
        }
    
    def _auto_detect_language(self):
        """Automatically detect system language and set as current language."""
        try:
            # Try to get language from environment variables
            lang_env = os.environ.get('LANG') or os.environ.get('LC_ALL') or os.environ.get('LANGUAGE')
            
            if lang_env:
                # Extract language code (first 2 characters)
                lang_code = lang_env[:2].lower()
                
                # Map to supported languages
                lang_mapping = {
                    'en': Language.ENGLISH,
                    'es': Language.SPANISH,
                    'fr': Language.FRENCH,
                    'de': Language.GERMAN,
                    'ja': Language.JAPANESE,
                    'zh': Language.CHINESE,
                }
                
                if lang_code in lang_mapping:
                    with self._lock:
                        self._current_language = lang_mapping[lang_code]
                        _logger.info(f"Auto-detected language: {self._current_language.value}")
                        return
            
            # Fallback to system locale
            try:
                system_locale = locale.getdefaultlocale()[0]
                if system_locale:
                    lang_code = system_locale[:2].lower()
                    lang_mapping = {
                        'en': Language.ENGLISH,
                        'es': Language.SPANISH,
                        'fr': Language.FRENCH,
                        'de': Language.GERMAN,
                        'ja': Language.JAPANESE,
                        'zh': Language.CHINESE,
                    }
                    
                    if lang_code in lang_mapping:
                        with self._lock:
                            self._current_language = lang_mapping[lang_code]
                            _logger.info(f"Language detected from system locale: {self._current_language.value}")
                            return
            except Exception as e:
                _logger.debug(f"Failed to detect system locale: {e}")
        
        except Exception as e:
            _logger.debug(f"Language auto-detection failed: {e}")
        
        # Default to English if detection fails
        _logger.info("Using default language: English")
    
    def set_language(self, language: Union[Language, str]):
        """Set the current language."""
        if isinstance(language, str):
            try:
                language = Language(language.lower())
            except ValueError:
                _logger.warning(f"Unknown language code: {language}, falling back to English")
                language = Language.ENGLISH
        
        with self._lock:
            self._current_language = language
            self._current_locale = None  # Reset locale to force recomputation
            _logger.info(f"Language set to: {language.value}")
    
    def set_locale(self, language: Union[Language, str], region: Union[Region, str]):
        """Set current language and region for locale-specific formatting."""
        if isinstance(language, str):
            language = Language(language.lower())
        if isinstance(region, str):
            region = Region(region.upper())
        
        with self._lock:
            self._current_language = language
            
            # Get or create locale configuration
            locale_key = (language, region)
            if locale_key in _DEFAULT_LOCALES:
                self._current_locale = _DEFAULT_LOCALES[locale_key]
            else:
                # Create custom locale config
                self._current_locale = LocaleConfig(language, region)
            
            _logger.info(f"Locale set to: {language.value}_{region.value}")
    
    def get_current_language(self) -> Language:
        """Get the current language."""
        return self._current_language
    
    def get_current_locale(self) -> Optional[LocaleConfig]:
        """Get the current locale configuration."""
        return self._current_locale
    
    def translate(self, key: str, **kwargs) -> str:
        """
        Translate a message key to the current language.
        
        Args:
            key: Translation key (e.g., "cli.setup.complete")
            **kwargs: Format arguments for the translation
            
        Returns:
            Translated and formatted message
        """
        with self._lock:
            # Get translations for current language
            translations = self._translations.get(self._current_language, {})
            
            # Try to find the translation
            message = translations.get(key)
            
            # Fallback to English if translation not found
            if message is None and self._current_language != self._fallback_language:
                fallback_translations = self._translations.get(self._fallback_language, {})
                message = fallback_translations.get(key)
            
            # Final fallback to the key itself
            if message is None:
                message = key
                _logger.debug(f"Translation not found for key: {key}")
            
            # Format the message with provided arguments
            try:
                return message.format(**kwargs)
            except (KeyError, ValueError) as e:
                _logger.debug(f"Error formatting translation '{key}': {e}")
                return message
    
    def format_number(self, 
                     value: Union[int, float, Decimal], 
                     decimal_places: Optional[int] = None) -> str:
        """
        Format a number according to current locale.
        
        Args:
            value: Number to format
            decimal_places: Number of decimal places (auto-detect if None)
            
        Returns:
            Formatted number string
        """
        if self._current_locale is None:
            # Use default formatting
            if decimal_places is not None:
                return f"{value:.{decimal_places}f}"
            return str(value)
        
        # Convert to string with specified decimal places
        if decimal_places is not None:
            formatted = f"{float(value):.{decimal_places}f}"
        else:
            formatted = str(float(value))
        
        # Split into integer and decimal parts
        if '.' in formatted:
            int_part, dec_part = formatted.split('.')
        else:
            int_part, dec_part = formatted, ""
        
        # Format integer part with thousands separator
        if len(int_part) > 3:
            # Add thousands separators
            int_part = self._add_thousands_separators(int_part)
        
        # Combine with locale-specific decimal separator
        if dec_part and dec_part != "0" * len(dec_part):
            return f"{int_part}{self._current_locale.decimal_separator}{dec_part}"
        else:
            return int_part
    
    def _add_thousands_separators(self, int_str: str) -> str:
        """Add thousands separators to integer string."""
        if not self._current_locale:
            return int_str
        
        # Add separators from right to left
        result = ""
        for i, digit in enumerate(reversed(int_str)):
            if i > 0 and i % 3 == 0:
                result = self._current_locale.thousands_separator + result
            result = digit + result
        
        return result
    
    def format_metric(self, 
                     value: Union[int, float], 
                     metric_type: str,
                     decimal_places: int = 2) -> str:
        """
        Format a metric value with appropriate unit and localization.
        
        Args:
            value: Metric value
            metric_type: Type of metric (latency, energy, power, memory, etc.)
            decimal_places: Number of decimal places
            
        Returns:
            Formatted metric string with unit
        """
        # Format the number
        formatted_value = self.format_number(value, decimal_places)
        
        # Get the appropriate unit
        unit_key = f"unit.{self._get_unit_key(metric_type)}"
        unit = self.translate(unit_key)
        
        return f"{formatted_value} {unit}"
    
    def _get_unit_key(self, metric_type: str) -> str:
        """Get unit key for metric type."""
        unit_mapping = {
            'latency': 'milliseconds',
            'energy': 'millijoules',
            'power': 'milliwatts',
            'memory': 'kilobytes',
            'accuracy': 'percentage',
            'temperature': 'celsius',
            'voltage': 'volts',
            'current': 'milliamps',
        }
        return unit_mapping.get(metric_type.lower(), metric_type.lower())
    
    def format_datetime(self, 
                       dt: datetime,
                       include_time: bool = True) -> str:
        """
        Format datetime according to current locale.
        
        Args:
            dt: Datetime object to format
            include_time: Whether to include time portion
            
        Returns:
            Formatted datetime string
        """
        if self._current_locale is None:
            # Use ISO format as default
            if include_time:
                return dt.strftime("%Y-%m-%d %H:%M:%S")
            else:
                return dt.strftime("%Y-%m-%d")
        
        # Use locale-specific format
        if include_time:
            format_str = f"{self._current_locale.date_format} {self._current_locale.time_format}"
        else:
            format_str = self._current_locale.date_format
        
        return dt.strftime(format_str)
    
    def get_compliance_requirements(self) -> List[str]:
        """Get compliance requirements for current locale."""
        if self._current_locale is None:
            return []
        return self._current_locale.compliance_frameworks.copy()
    
    def add_custom_translation(self, 
                              language: Union[Language, str],
                              key: str, 
                              translation: str):
        """
        Add or update a custom translation.
        
        Args:
            language: Target language
            key: Translation key
            translation: Translation text
        """
        if isinstance(language, str):
            language = Language(language.lower())
        
        with self._lock:
            if language not in self._translations:
                self._translations[language] = {}
            
            self._translations[language][key] = translation
            _logger.debug(f"Added custom translation for {language.value}: {key}")
    
    def load_translations_from_file(self, file_path: Union[str, Path]):
        """
        Load translations from a JSON file.
        
        Expected format:
        {
            "en": {"key1": "English translation", ...},
            "es": {"key1": "Spanish translation", ...},
            ...
        }
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                translations_data = json.load(f)
            
            with self._lock:
                for lang_code, translations in translations_data.items():
                    try:
                        language = Language(lang_code.lower())
                        if language not in self._translations:
                            self._translations[language] = {}
                        
                        self._translations[language].update(translations)
                        
                    except ValueError:
                        _logger.warning(f"Unknown language code in file: {lang_code}")
                        continue
            
            _logger.info(f"Translations loaded from: {file_path}")
            
        except Exception as e:
            _logger.error(f"Failed to load translations from {file_path}: {e}")
    
    def export_translations(self, file_path: Union[str, Path]):
        """Export current translations to a JSON file."""
        try:
            export_data = {}
            with self._lock:
                for language, translations in self._translations.items():
                    export_data[language.value] = translations
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False, sort_keys=True)
            
            _logger.info(f"Translations exported to: {file_path}")
            
        except Exception as e:
            _logger.error(f"Failed to export translations to {file_path}: {e}")


# Global translation manager instance
_translation_manager = TranslationManager()


def get_translation_manager() -> TranslationManager:
    """Get the global translation manager instance."""
    return _translation_manager


def set_language(language: Union[Language, str]):
    """Set the current language globally."""
    _translation_manager.set_language(language)


def set_locale(language: Union[Language, str], region: Union[Region, str]):
    """Set the current locale globally."""
    _translation_manager.set_locale(language, region)


def translate(key: str, **kwargs) -> str:
    """Translate a message key using the global translation manager."""
    return _translation_manager.translate(key, **kwargs)


def format_metric(value: Union[int, float], 
                 metric_type: str,
                 decimal_places: int = 2) -> str:
    """Format a metric value using the global translation manager."""
    return _translation_manager.format_metric(value, metric_type, decimal_places)


def format_datetime(dt: datetime, include_time: bool = True) -> str:
    """Format datetime using the global translation manager."""
    return _translation_manager.format_datetime(dt, include_time)


def get_compliance_requirements() -> List[str]:
    """Get compliance requirements for current locale."""
    return _translation_manager.get_compliance_requirements()


class LocalizedLogger:
    """Logger wrapper that provides localized log messages."""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.translation_manager = get_translation_manager()
    
    def info(self, key: str, **kwargs):
        """Log info message with translation."""
        message = self.translation_manager.translate(key, **kwargs)
        self.logger.info(message)
    
    def warning(self, key: str, **kwargs):
        """Log warning message with translation."""
        message = self.translation_manager.translate(key, **kwargs)
        self.logger.warning(message)
    
    def error(self, key: str, **kwargs):
        """Log error message with translation."""
        message = self.translation_manager.translate(key, **kwargs)
        self.logger.error(message)
    
    def debug(self, key: str, **kwargs):
        """Log debug message with translation."""
        message = self.translation_manager.translate(key, **kwargs)
        self.logger.debug(message)
    
    def critical(self, key: str, **kwargs):
        """Log critical message with translation."""
        message = self.translation_manager.translate(key, **kwargs)
        self.logger.critical(message)


def get_localized_logger(name: str) -> LocalizedLogger:
    """Get a localized logger instance."""
    return LocalizedLogger(name)


# Configuration management
class I18nConfig:
    """Configuration class for internationalization settings."""
    
    def __init__(self):
        self._config = {
            'language': 'en',
            'region': 'US',
            'auto_detect': True,
            'fallback_language': 'en',
            'custom_translations_path': None,
            'enable_rtl': False,
            'date_format_override': None,
            'time_format_override': None,
            'number_format_override': None,
        }
    
    def load_from_env(self):
        """Load configuration from environment variables."""
        env_mapping = {
            'LIQUID_EDGE_LANGUAGE': 'language',
            'LIQUID_EDGE_REGION': 'region',
            'LIQUID_EDGE_I18N_AUTO_DETECT': 'auto_detect',
            'LIQUID_EDGE_I18N_FALLBACK_LANG': 'fallback_language',
            'LIQUID_EDGE_I18N_TRANSLATIONS_PATH': 'custom_translations_path',
            'LIQUID_EDGE_I18N_RTL': 'enable_rtl',
            'LIQUID_EDGE_DATE_FORMAT': 'date_format_override',
            'LIQUID_EDGE_TIME_FORMAT': 'time_format_override',
            'LIQUID_EDGE_NUMBER_FORMAT': 'number_format_override',
        }
        
        for env_var, config_key in env_mapping.items():
            value = os.environ.get(env_var)
            if value is not None:
                # Convert string values to appropriate types
                if config_key in ['auto_detect', 'enable_rtl']:
                    value = value.lower() in ['true', '1', 'yes', 'on']
                self._config[config_key] = value
    
    def get(self, key: str, default=None):
        """Get configuration value."""
        return self._config.get(key, default)
    
    def set(self, key: str, value):
        """Set configuration value."""
        self._config[key] = value
    
    def apply_to_translation_manager(self, tm: TranslationManager):
        """Apply configuration to translation manager."""
        if not self.get('auto_detect', True):
            # Manual language setting
            language = self.get('language', 'en')
            region = self.get('region', 'US')
            try:
                tm.set_locale(language, region)
            except ValueError as e:
                _logger.warning(f"Invalid locale configuration: {e}")
        
        # Load custom translations if specified
        custom_path = self.get('custom_translations_path')
        if custom_path and Path(custom_path).exists():
            tm.load_translations_from_file(custom_path)


# Initialize default configuration
_i18n_config = I18nConfig()
_i18n_config.load_from_env()
_i18n_config.apply_to_translation_manager(_translation_manager)


def get_i18n_config() -> I18nConfig:
    """Get the global i18n configuration instance."""
    return _i18n_config


# Integration helpers for existing systems
def integrate_with_error_handler(error_handler):
    """Integrate i18n with the existing error handling system."""
    from liquid_edge.error_handling import RobustErrorHandler
    
    if not isinstance(error_handler, RobustErrorHandler):
        return
    
    # Add localized error messages
    original_log_error = error_handler._log_error
    
    def localized_log_error(exception, context):
        """Enhanced error logging with localization."""
        exception_type = type(exception).__name__.lower()
        
        # Try to find localized error message
        error_key = f"error.{exception_type}"
        localized_message = translate(error_key)
        
        if localized_message != error_key:  # Translation was found
            context.metadata['localized_message'] = localized_message
        
        # Call original method
        original_log_error(exception, context)
    
    error_handler._log_error = localized_log_error


def integrate_with_cli():
    """Integrate i18n with the existing CLI system."""
    # This would be called during CLI initialization to set up localized messages
    # The CLI module would import this and use the translate() function
    pass


def integrate_with_compliance(compliance_manager):
    """Integrate i18n with the existing compliance system."""
    from liquid_edge.compliance import ComplianceManager
    
    if not isinstance(compliance_manager, ComplianceManager):
        return
    
    # Add localized compliance messages
    original_log_info = compliance_manager.logger.info
    
    def localized_compliance_log(message, *args, **kwargs):
        """Enhanced compliance logging with localization."""
        # Try to find localized compliance message
        if 'consent' in message.lower():
            localized_message = translate('compliance.consent_obtained')
        elif 'processing' in message.lower():
            localized_message = translate('compliance.data_processing')
        else:
            localized_message = message
        
        original_log_info(localized_message, *args, **kwargs)
    
    compliance_manager.logger.info = localized_compliance_log


# Performance consideration: The i18n system is designed to be lightweight
# - Translation lookups are O(1) dictionary operations
# - Thread-safe operations use RLock for minimal performance impact
# - Lazy loading and caching reduce memory footprint
# - No impact on core neural network operations as translations are separate