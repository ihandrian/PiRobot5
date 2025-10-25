"""
Error Handler for PiRobot5
Handles errors, logging, and recovery mechanisms
"""

import logging
import traceback
import sys
from typing import Optional, Callable, Any
from enum import Enum, auto
from dataclasses import dataclass
from datetime import datetime


class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()
    CRITICAL = auto()


@dataclass
class ErrorInfo:
    """Error information container"""
    timestamp: datetime
    severity: ErrorSeverity
    error_type: str
    message: str
    traceback: str
    context: dict


class ErrorHandler:
    """Centralized error handling for PiRobot"""
    
    def __init__(self, logger_name: str = "PiRobot.ErrorHandler"):
        self.logger = logging.getLogger(logger_name)
        self.error_callbacks: list[Callable[[ErrorInfo], None]] = []
        self.error_history: list[ErrorInfo] = []
        self.max_history = 100
        
    def register_callback(self, callback: Callable[[ErrorInfo], None]) -> None:
        """Register an error callback function"""
        self.error_callbacks.append(callback)
        
    def handle_error(self, 
                    error: Exception, 
                    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                    context: Optional[dict] = None) -> None:
        """Handle an error with logging and callbacks"""
        
        error_info = ErrorInfo(
            timestamp=datetime.now(),
            severity=severity,
            error_type=type(error).__name__,
            message=str(error),
            traceback=traceback.format_exc(),
            context=context or {}
        )
        
        # Add to history
        self.error_history.append(error_info)
        if len(self.error_history) > self.max_history:
            self.error_history.pop(0)
            
        # Log the error
        log_level = {
            ErrorSeverity.LOW: logging.INFO,
            ErrorSeverity.MEDIUM: logging.WARNING,
            ErrorSeverity.HIGH: logging.ERROR,
            ErrorSeverity.CRITICAL: logging.CRITICAL
        }[severity]
        
        self.logger.log(log_level, f"Error: {error_info.message}")
        if severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
            self.logger.error(f"Traceback: {error_info.traceback}")
            
        # Call registered callbacks
        for callback in self.error_callbacks:
            try:
                callback(error_info)
            except Exception as e:
                self.logger.error(f"Error in callback: {e}")
                
    def get_recent_errors(self, count: int = 10) -> list[ErrorInfo]:
        """Get recent errors from history"""
        return self.error_history[-count:]
        
    def get_errors_by_severity(self, severity: ErrorSeverity) -> list[ErrorInfo]:
        """Get errors filtered by severity"""
        return [error for error in self.error_history if error.severity == severity]
        
    def clear_history(self) -> None:
        """Clear error history"""
        self.error_history.clear()
        
    def safe_execute(self, 
                    func: Callable, 
                    *args, 
                    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                    context: Optional[dict] = None,
                    **kwargs) -> Any:
        """Safely execute a function with error handling"""
        try:
            return func(*args, **kwargs)
        except Exception as e:
            self.handle_error(e, severity, context)
            return None


# Global error handler instance
error_handler = ErrorHandler()


def handle_error(error: Exception, 
                severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                context: Optional[dict] = None) -> None:
    """Convenience function for global error handling"""
    error_handler.handle_error(error, severity, context)


def safe_execute(func: Callable, 
                *args, 
                severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                context: Optional[dict] = None,
                **kwargs) -> Any:
    """Convenience function for safe execution"""
    return error_handler.safe_execute(func, *args, severity=severity, context=context, **kwargs)
