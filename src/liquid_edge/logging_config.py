# Copyright (c) 2025 Liquid Edge LLN Kit Contributors
# SPDX-License-Identifier: MIT

"""Centralized logging configuration for Liquid Edge LLN Kit."""

import logging
import logging.config
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import structlog

# Default log levels
DEFAULT_LOG_LEVEL = "INFO"
DEVELOPMENT_LOG_LEVEL = "DEBUG"

# Log format configurations
SIMPLE_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DETAILED_FORMAT = (
    "%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(funcName)s:%(lineno)d - %(message)s"
)
JSON_FORMAT = "json"


def get_log_level() -> str:
    """Get the log level from environment or defaults."""
    env_level = os.getenv("LOG_LEVEL", "").upper()
    if env_level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
        return env_level
    
    # Use debug level in development
    if os.getenv("LIQUID_ENV", "production").lower() in ["development", "dev"]:
        return DEVELOPMENT_LOG_LEVEL
    
    return DEFAULT_LOG_LEVEL


def get_log_format() -> str:
    """Get the log format from environment or defaults."""
    return os.getenv("LOG_FORMAT", "simple").lower()


def setup_logging(
    level: Optional[str] = None,
    format_type: Optional[str] = None,
    log_file: Optional[Path] = None,
    enable_structured: bool = True,
) -> None:
    """
    Setup centralized logging configuration.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_type: Format type (simple, detailed, json)
        log_file: Optional log file path
        enable_structured: Whether to use structured logging
    """
    if level is None:
        level = get_log_level()
    
    if format_type is None:
        format_type = get_log_format()
    
    # Configure basic logging
    if format_type == "json":
        _setup_json_logging(level, log_file)
    else:
        _setup_standard_logging(level, format_type, log_file)
    
    # Configure structured logging if enabled
    if enable_structured:
        _setup_structured_logging()
    
    # Configure specific loggers
    _configure_library_loggers()


def _setup_standard_logging(level: str, format_type: str, log_file: Optional[Path]) -> None:
    """Setup standard Python logging."""
    if format_type == "detailed":
        formatter = logging.Formatter(DETAILED_FORMAT)
    else:
        formatter = logging.Formatter(SIMPLE_FORMAT)
    
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, level),
        handlers=handlers,
        format=formatter._fmt,
    )


def _setup_json_logging(level: str, log_file: Optional[Path]) -> None:
    """Setup JSON logging configuration."""
    import json_logging
    
    json_logging.init_non_web(enable_json=True)
    
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, level))
    
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(log_file)
        logger.addHandler(handler)


def _setup_structured_logging() -> None:
    """Setup structured logging with structlog."""
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer() if _is_development() else structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, get_log_level())
        ),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def _configure_library_loggers() -> None:
    """Configure logging for third-party libraries."""
    # Reduce verbosity of common libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("docker").setLevel(logging.WARNING)
    
    # JAX can be very verbose
    logging.getLogger("jax").setLevel(logging.WARNING)
    logging.getLogger("jaxlib").setLevel(logging.WARNING)
    
    # TensorFlow/PyTorch if used
    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    logging.getLogger("torch").setLevel(logging.WARNING)


def _is_development() -> bool:
    """Check if running in development environment."""
    return os.getenv("LIQUID_ENV", "production").lower() in ["development", "dev"]


def get_logger(name: str) -> logging.Logger:
    """
    Get a configured logger instance.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


def get_structured_logger(name: str) -> structlog.BoundLogger:
    """
    Get a structured logger instance.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Configured structured logger instance
    """
    return structlog.get_logger(name)


# Performance monitoring context
class PerformanceLogger:
    """Logger for performance metrics."""
    
    def __init__(self, name: str):
        self.logger = get_structured_logger(f"performance.{name}")
    
    def log_inference_metrics(
        self,
        latency_ms: float,
        energy_mj: float,
        memory_kb: float,
        accuracy: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        """Log inference performance metrics."""
        self.logger.info(
            "inference_metrics",
            latency_ms=latency_ms,
            energy_mj=energy_mj,
            memory_kb=memory_kb,
            accuracy=accuracy,
            **kwargs,
        )
    
    def log_training_metrics(
        self,
        epoch: int,
        loss: float,
        learning_rate: float,
        training_time_s: float,
        **kwargs: Any,
    ) -> None:
        """Log training metrics."""
        self.logger.info(
            "training_metrics",
            epoch=epoch,
            loss=loss,
            learning_rate=learning_rate,
            training_time_s=training_time_s,
            **kwargs,
        )
    
    def log_deployment_metrics(
        self,
        target_device: str,
        deployment_time_s: float,
        model_size_kb: float,
        success: bool,
        **kwargs: Any,
    ) -> None:
        """Log deployment metrics."""
        self.logger.info(
            "deployment_metrics",
            target_device=target_device,
            deployment_time_s=deployment_time_s,
            model_size_kb=model_size_kb,
            success=success,
            **kwargs,
        )


# Energy monitoring context
class EnergyLogger:
    """Logger specifically for energy consumption metrics."""
    
    def __init__(self):
        self.logger = get_structured_logger("energy")
    
    def log_power_consumption(
        self,
        device: str,
        power_mw: float,
        voltage_v: float,
        current_ma: float,
        temperature_c: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        """Log power consumption metrics."""
        self.logger.info(
            "power_consumption",
            device=device,
            power_mw=power_mw,
            voltage_v=voltage_v,
            current_ma=current_ma,
            temperature_c=temperature_c,
            **kwargs,
        )
    
    def log_energy_budget(
        self,
        operation: str,
        energy_consumed_mj: float,
        energy_budget_mj: float,
        budget_remaining_pct: float,
        **kwargs: Any,
    ) -> None:
        """Log energy budget utilization."""
        self.logger.info(
            "energy_budget",
            operation=operation,
            energy_consumed_mj=energy_consumed_mj,
            energy_budget_mj=energy_budget_mj,
            budget_remaining_pct=budget_remaining_pct,
            **kwargs,
        )


# Error tracking
class ErrorLogger:
    """Logger for error tracking and debugging."""
    
    def __init__(self, name: str):
        self.logger = get_structured_logger(f"error.{name}")
    
    def log_model_error(
        self,
        error_type: str,
        error_message: str,
        model_info: Dict[str, Any],
        input_shape: Optional[tuple] = None,
        **kwargs: Any,
    ) -> None:
        """Log model-related errors."""
        self.logger.error(
            "model_error",
            error_type=error_type,
            error_message=error_message,
            model_info=model_info,
            input_shape=input_shape,
            **kwargs,
        )
    
    def log_hardware_error(
        self,
        device: str,
        error_type: str,
        error_message: str,
        device_status: Dict[str, Any],
        **kwargs: Any,
    ) -> None:
        """Log hardware-related errors."""
        self.logger.error(
            "hardware_error",
            device=device,
            error_type=error_type,
            error_message=error_message,
            device_status=device_status,
            **kwargs,
        )


# Initialize logging on module import if not already configured
if not logging.getLogger().handlers:
    setup_logging()