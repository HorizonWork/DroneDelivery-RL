import logging
import logging.handlers
import sys
import os
from typing import Dict, Optional, Any
from pathlib import Path
from datetime import datetime
import json
import traceback

class SystemLogger:

    def __init__(self, config: Dict[str, Any]):
        self.config = config

        self.log_level = getattr(logging, config.get("level", "INFO").upper())
        self.log_dir = Path(config.get("log_dir", "logs"))
        self.max_file_size = config.get("max_file_size_mb", 10)  1024  1024
        self.backup_count = config.get("backup_count", 5)

        self.console_format = config.get(
            "console_format", "(asctime)s - (name)s - (levelname)s - (message)s"
        )
        self.file_format = config.get(
            "file_format",
            "(asctime)s - (name)s - (levelname)s - (funcName)s:(lineno)d - (message)s",
        )

        self.component_loggers = {}

        self.log_dir.mkdir(parents=True, exist_ok=True)

        self._setup_root_logger()

        self._setup_component_loggers()

        self.logger = logging.getLogger(__name__)
        self.logger.info("System Logger initialized")

    def _setup_root_logger(self):

        root_logger = logging.getLogger()
        root_logger.setLevel(self.log_level)

        root_logger.handlers.clear()

        if self.config.get("console_logging", True):
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(self.log_level)
            console_formatter = logging.Formatter(self.console_format)
            console_handler.setFormatter(console_formatter)
            root_logger.addHandler(console_handler)

        if self.config.get("file_logging", True):
            log_file = self.log_dir / "system.log"
            file_handler = logging.handlers.RotatingFileHandler(
                log_file, maxBytes=self.max_file_size, backupCount=self.backup_count
            )
            file_handler.setLevel(self.log_level)
            file_formatter = logging.Formatter(self.file_format)
            file_handler.setFormatter(file_formatter)
            root_logger.addHandler(file_handler)

        if self.config.get("error_file_logging", True):
            error_file = self.log_dir / "errors.log"
            error_handler = logging.handlers.RotatingFileHandler(
                error_file, maxBytes=self.max_file_size, backupCount=self.backup_count
            )
            error_handler.setLevel(logging.ERROR)
            error_formatter = logging.Formatter(self.file_format)
            error_handler.setFormatter(error_formatter)
            root_logger.addHandler(error_handler)

    def _setup_component_loggers(self):

        components = ["localization", "planning", "rl", "environment", "utils"]

        for component in components:
            component_config = self.config.get("components", {}).get(component, {})

            if component_config.get("enabled", True):
                logger = self._create_component_logger(component, component_config)
                self.component_loggers[component] = logger

    def _create_component_logger(
        self, component_name: str, component_config: Dict[str, Any]
    ):

        logger = logging.getLogger(component_name)

        component_level = getattr(
            logging, component_config.get("level", "INFO").upper()
        )
        logger.setLevel(component_level)

        if component_config.get("separate_file", False):
            log_file = self.log_dir / f"{component_name}.log"
            file_handler = logging.handlers.RotatingFileHandler(
                log_file, maxBytes=self.max_file_size, backupCount=self.backup_count
            )
            file_handler.setLevel(component_level)
            file_formatter = logging.Formatter(self.file_format)
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)

        return logger

    def get_logger(self, name: str) - logging.Logger:

        return logging.getLogger(name)

    def log_system_event(
        self, event_type: str, component: str, data: Dict[str, Any], level: str = "INFO"
    ):

        logger = self.component_loggers.get(component, logging.getLogger(component))
        log_level = getattr(logging, level.upper())

        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "component": component,
            "data": data,
        }

        logger.log(log_level, f"SYSTEM_EVENT: {json.dumps(log_entry)}")

    def log_performance_metrics(self, component: str, metrics: Dict[str, float]):

        self.log_system_event("performance_metrics", component, metrics, "INFO")

    def log_error_with_context(
        self, component: str, error: Exception, context: Dict[str, Any] = None
    ):

        logger = self.component_loggers.get(component, logging.getLogger(component))

        error_data = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc(),
            "context": context or {},
        }

        self.log_system_event("error", component, error_data, "ERROR")

class StructuredLogger:

    def __init__(self, component_name: str, log_file: Path):
        self.component_name = component_name
        self.log_file = log_file

        log_file.parent.mkdir(parents=True, exist_ok=True)

    def log_structured(
        self, event_type: str, data: Dict[str, Any], level: str = "INFO"
    ):

        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "component": self.component_name,
            "event_type": event_type,
            "level": level,
            "data": data,
        }

        with open(self.log_file, "a") as f:
            f.write(json.dumps(log_entry, default=str) + "\n")

def setup_logging(config: Dict[str, Any]) - SystemLogger:

    default_config = {
        "level": "INFO",
        "log_dir": "logs",
        "console_logging": True,
        "file_logging": True,
        "error_file_logging": True,
        "max_file_size_mb": 10,
        "backup_count": 5,
        "components": {
            "localization": {"enabled": True, "level": "INFO"},
            "planning": {"enabled": True, "level": "INFO"},
            "rl": {"enabled": True, "level": "INFO"},
            "environment": {"enabled": True, "level": "INFO"},
            "utils": {"enabled": True, "level": "WARNING"},
        },
    }

    merged_config = {default_config, config}

    return SystemLogger(merged_config)

def get_logger(name: str) - logging.Logger:

    return logging.getLogger(name)

def log_exceptions(component: str):

    def decorator(func):
        def wrapper(args, kwargs):
            try:
                return func(args, kwargs)
            except Exception as e:
                logger = logging.getLogger(component)
                logger.error(f"Exception in {func.__name__}: {e}", exc_info=True)
                raise

        return wrapper

    return decorator
