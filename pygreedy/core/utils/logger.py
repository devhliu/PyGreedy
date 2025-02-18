"""
PyGreedy Logger Module
====================

This module provides logging functionality for the PyGreedy package,
including customized formatters and handlers for registration tracking.

Created by: devhliu
Created at: 2025-02-18 04:50:12 UTC
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any, Union
from datetime import datetime
import json
import threading
from functools import wraps

class RegistrationLogFormatter(logging.Formatter):
    """Custom formatter for registration logging."""
    
    def __init__(self):
        """Initialize formatter with custom format."""
        super().__init__(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
    def format(self, record):
        """Format log record with additional registration-specific information."""
        if hasattr(record, 'registration_params'):
            record.msg = f"Registration Parameters: {record.registration_params}"
        if hasattr(record, 'registration_result'):
            record.msg = f"Registration Result: {record.registration_result}"
        return super().format(record)

class RegistrationLogger:
    """
    Logger class for registration tracking with support for metrics
    and parameter logging.
    """
    
    def __init__(
        self,
        name: str,
        log_dir: Optional[Union[str, Path]] = None,
        level: int = logging.INFO
    ):
        """
        Initialize registration logger.

        Parameters
        ----------
        name : str
            Logger name
        log_dir : Optional[Union[str, Path]], optional
            Directory for log files, by default None
        level : int, optional
            Logging level, by default logging.INFO
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # Create formatter
        formatter = RegistrationLogFormatter()
        
        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # Create file handler if log_dir is provided
        if log_dir is not None:
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = log_dir / f"registration_{timestamp}.log"
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            
        # Initialize metrics storage
        self.metrics: Dict[str, list] = {}
        self._metrics_lock = threading.Lock()
        
    def log_parameters(
        self,
        params: Dict[str, Any]
    ) -> None:
        """
        Log registration parameters.

        Parameters
        ----------
        params : Dict[str, Any]
            Registration parameters
        """
        self.logger.info(
            "Registration parameters",
            extra={'registration_params': json.dumps(params, indent=2)}
        )
        
    def log_result(
        self,
        result: Dict[str, Any]
    ) -> None:
        """
        Log registration result.

        Parameters
        ----------
        result : Dict[str, Any]
            Registration result
        """
        self.logger.info(
            "Registration result",
            extra={'registration_result': json.dumps(result, indent=2)}
        )
        
    def log_metric(
        self,
        name: str,
        value: float,
        step: Optional[int] = None
    ) -> None:
        """
        Log metric value.

        Parameters
        ----------
        name : str
            Metric name
        value : float
            Metric value
        step : Optional[int], optional
            Step number, by default None
        """
        with self._metrics_lock:
            if name not in self.metrics:
                self.metrics[name] = []
            self.metrics[name].append((step, value))
        
        self.logger.debug(f"Metric {name}: {value} (step: {step})")
        
    def get_metrics(
        self,
        name: Optional[str] = None
    ) -> Dict[str, list]:
        """
        Get logged metrics.

        Parameters
        ----------
        name : Optional[str], optional
            Metric name, by default None (returns all metrics)

        Returns
        -------
        Dict[str, list]
            Dictionary of metric values
        """
        with self._metrics_lock:
            if name is not None:
                return {name: self.metrics.get(name, [])}
            return self.metrics.copy()
        
    def clear_metrics(self) -> None:
        """Clear all logged metrics."""
        with self._metrics_lock:
            self.metrics.clear()

def log_registration(logger: Optional[RegistrationLogger] = None):
    """
    Decorator for logging registration function calls.

    Parameters
    ----------
    logger : Optional[RegistrationLogger], optional
        Logger instance, by default None

    Returns
    -------
    callable
        Decorated function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create logger if not provided
            nonlocal logger
            if logger is None:
                logger = RegistrationLogger(func.__name__)
                
            # Log function call
            logger.logger.info(f"Starting registration: {func.__name__}")
            
            try:
                # Log parameters if available
                if kwargs.get('parameters'):
                    logger.log_parameters(kwargs['parameters'])
                    
                # Execute registration
                result = func(*args, **kwargs)
                
                # Log result
                if isinstance(result, dict):
                    logger.log_result(result)
                    
                return result
                
            except Exception as e:
                logger.logger.error(f"Registration failed: {str(e)}", exc_info=True)
                raise
                
        return wrapper
    return decorator

def setup_logger(
    name: str,
    log_dir: Optional[Union[str, Path]] = None,
    level: int = logging.INFO
) -> RegistrationLogger:
    """
    Set up registration logger.

    Parameters
    ----------
    name : str
        Logger name
    log_dir : Optional[Union[str, Path]], optional
        Directory for log files, by default None
    level : int, optional
        Logging level, by default logging.INFO

    Returns
    -------
    RegistrationLogger
        Configured logger instance
    """
    return RegistrationLogger(name, log_dir, level)

def log_registration_params(
    logger: RegistrationLogger,
    params: Dict[str, Any]
) -> None:
    """
    Log registration parameters.

    Parameters
    ----------
    logger : RegistrationLogger
        Logger instance
    params : Dict[str, Any]
        Registration parameters
    """
    logger.log_parameters(params)

def log_registration_result(
    logger: RegistrationLogger,
    result: Dict[str, Any]
) -> None:
    """
    Log registration result.

    Parameters
    ----------
    logger : RegistrationLogger
        Logger instance
    result : Dict[str, Any]
        Registration result
    """
    logger.log_result(result)