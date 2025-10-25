# PiRobot V.4 API Documentation

## Overview

This document provides detailed API documentation for the PiRobot V.4 system. It covers all major components, their interfaces, and usage examples.

## Core Components

### MotorController

The `MotorController` class handles motor control operations with safety features and encoder feedback.

```python
class MotorController:
    """Controls motor operations with safety features and encoder feedback."""
    
    def __init__(self, config: MotorConfig) -> None:
        """Initialize the motor controller.
        
        Args:
            config: Motor configuration parameters
        """
        pass
        
    def move(self, left_speed: float, right_speed: float) -> bool:
        """Set motor speeds with safety checks.
        
        Args:
            left_speed: Left motor speed (-100 to 100)
            right_speed: Right motor speed (-100 to 100)
            
        Returns:
            bool: True if movement command was successful
            
        Raises:
            ValueError: If speeds are outside valid range
            SafetyError: If safety conditions are not met
        """
        pass
        
    def stop(self, emergency: bool = False) -> None:
        """Stop motors with optional emergency stop.
        
        Args:
            emergency: If True, perform emergency stop
        """
        pass
        
    def get_speed(self) -> Tuple[float, float]:
        """Get current motor speeds.
        
        Returns:
            Tuple[float, float]: Current left and right motor speeds
        """
        pass
        
    def auto_tune_pid(self, method: str = 'relay') -> Dict[str, float]:
        """Auto-tune PID parameters.
        
        Args:
            method: Tuning method ('relay', 'genetic', 'pattern')
            
        Returns:
            Dict[str, float]: Tuned PID parameters
        """
        pass
```

### EncoderHandler

The `EncoderHandler` class manages encoder inputs and provides speed feedback.

```python
class EncoderHandler:
    """Handles encoder inputs and provides speed feedback."""
    
    def __init__(self, pin_a: int, pin_b: int, 
                 pulses_per_revolution: int = 20,
                 wheel_diameter_mm: float = 65.0) -> None:
        """Initialize encoder handler.
        
        Args:
            pin_a: GPIO pin for encoder channel A
            pin_b: GPIO pin for encoder channel B
            pulses_per_revolution: Encoder pulses per revolution
            wheel_diameter_mm: Wheel diameter in millimeters
        """
        pass
        
    def get_speed(self) -> float:
        """Get current speed in mm/s.
        
        Returns:
            float: Current speed
        """
        pass
        
    def get_count(self) -> int:
        """Get current encoder count.
        
        Returns:
            int: Current encoder count
        """
        pass
        
    def reset(self) -> None:
        """Reset encoder count."""
        pass
```

### PIDController

The `PIDController` class implements PID control for motor speed regulation.

```python
class PIDController:
    """Implements PID control for motor speed regulation."""
    
    def __init__(self, kp: float, ki: float, kd: float,
                 output_limits: Tuple[float, float] = (-100, 100)) -> None:
        """Initialize PID controller.
        
        Args:
            kp: Proportional gain
            ki: Integral gain
            kd: Derivative gain
            output_limits: Tuple of (min, max) output values
        """
        pass
        
    def compute(self, setpoint: float, measurement: float) -> float:
        """Compute PID output.
        
        Args:
            setpoint: Target value
            measurement: Current value
            
        Returns:
            float: PID output
        """
        pass
        
    def reset(self) -> None:
        """Reset PID controller state."""
        pass
```

### TemperatureMonitor

The `TemperatureMonitor` class monitors motor temperatures and implements protection.

```python
class TemperatureMonitor:
    """Monitors motor temperatures and implements protection."""
    
    def __init__(self, adc_channels: Dict[str, int],
                 max_temp: float = 80.0,
                 warning_temp: float = 70.0) -> None:
        """Initialize temperature monitor.
        
        Args:
            adc_channels: Dictionary of motor IDs to ADC channels
            max_temp: Maximum allowed temperature
            warning_temp: Temperature for warning
        """
        pass
        
    def get_temperature(self, motor_id: str) -> float:
        """Get current temperature for a motor.
        
        Args:
            motor_id: Motor identifier
            
        Returns:
            float: Current temperature in Celsius
        """
        pass
        
    def is_overheating(self, motor_id: str) -> bool:
        """Check if motor is overheating.
        
        Args:
            motor_id: Motor identifier
            
        Returns:
            bool: True if motor is overheating
        """
        pass
```

### SafetyMonitor

The `SafetyMonitor` class implements safety features and monitoring.

```python
class SafetyMonitor:
    """Implements safety features and monitoring."""
    
    def __init__(self, config: SafetyConfig) -> None:
        """Initialize safety monitor.
        
        Args:
            config: Safety configuration parameters
        """
        pass
        
    def check_safety(self) -> SafetyLevel:
        """Check current safety level.
        
        Returns:
            SafetyLevel: Current safety level
        """
        pass
        
    def add_warning(self, source: str, message: str,
                   level: SafetyLevel = SafetyLevel.WARNING) -> None:
        """Add a safety warning.
        
        Args:
            source: Warning source
            message: Warning message
            level: Warning level
        """
        pass
        
    def clear_warning(self, source: str) -> None:
        """Clear a safety warning.
        
        Args:
            source: Warning source to clear
        """
        pass
```

## Navigation Components

### LaneDetection

The `LaneDetection` class implements lane detection algorithms.

```python
class LaneDetection:
    """Implements lane detection algorithms."""
    
    def __init__(self, camera_id: int = 0,
                 frame_size: Tuple[int, int] = (640, 480)) -> None:
        """Initialize lane detection.
        
        Args:
            camera_id: Camera device ID
            frame_size: Frame size (width, height)
        """
        pass
        
    def detect_lanes(self, frame: np.ndarray) -> List[Line]:
        """Detect lanes in frame.
        
        Args:
            frame: Input frame
            
        Returns:
            List[Line]: Detected lane lines
        """
        pass
        
    def get_steering_angle(self) -> float:
        """Get recommended steering angle.
        
        Returns:
            float: Steering angle in degrees
        """
        pass
```

### WaypointNavigator

The `WaypointNavigator` class handles waypoint-based navigation.

```python
class WaypointNavigator:
    """Handles waypoint-based navigation."""
    
    def __init__(self, gps_port: str = '/dev/ttyUSB0',
                 baud_rate: int = 9600) -> None:
        """Initialize waypoint navigator.
        
        Args:
            gps_port: GPS serial port
            baud_rate: GPS baud rate
        """
        pass
        
    def add_waypoint(self, latitude: float, longitude: float) -> None:
        """Add a waypoint.
        
        Args:
            latitude: Waypoint latitude
            longitude: Waypoint longitude
        """
        pass
        
    def get_next_waypoint(self) -> Tuple[float, float]:
        """Get next waypoint.
        
        Returns:
            Tuple[float, float]: Next waypoint (latitude, longitude)
        """
        pass
        
    def update_position(self) -> None:
        """Update current position from GPS."""
        pass
```

## Utility Components

### ErrorHandler

The `ErrorHandler` class manages error handling and logging.

```python
class ErrorHandler:
    """Manages error handling and logging."""
    
    def __init__(self, log_file: str = 'robot.log') -> None:
        """Initialize error handler.
        
        Args:
            log_file: Log file path
        """
        pass
        
    def handle_error(self, error: Exception,
                    level: str = 'ERROR') -> None:
        """Handle an error.
        
        Args:
            error: Exception to handle
            level: Error level
        """
        pass
        
    def log_message(self, message: str,
                   level: str = 'INFO') -> None:
        """Log a message.
        
        Args:
            message: Message to log
            level: Log level
        """
        pass
```

### PIDTuner

The `PIDTuner` class provides PID parameter tuning capabilities.

```python
class PIDTuner:
    """Provides PID parameter tuning capabilities."""
    
    def __init__(self, measurement_func: Callable,
                 output_func: Callable) -> None:
        """Initialize PID tuner.
        
        Args:
            measurement_func: Function to get measurements
            output_func: Function to set output
        """
        pass
        
    def relay_auto_tune(self, setpoint: float,
                       relay_amplitude: float = 10.0) -> Dict[str, float]:
        """Perform relay auto-tuning.
        
        Args:
            setpoint: Target value
            relay_amplitude: Relay amplitude
            
        Returns:
            Dict[str, float]: Tuned PID parameters
        """
        pass
        
    def genetic_optimize(self, setpoint: float,
                        population_size: int = 50,
                        generations: int = 20) -> Dict[str, float]:
        """Optimize PID parameters using genetic algorithm.
        
        Args:
            setpoint: Target value
            population_size: Population size
            generations: Number of generations
            
        Returns:
            Dict[str, float]: Optimized PID parameters
        """
        pass
```

## Web Interface

### WebServer

The `WebServer` class provides web interface functionality.

```python
class WebServer:
    """Provides web interface functionality."""
    
    def __init__(self, host: str = '0.0.0.0',
                 port: int = 8080) -> None:
        """Initialize web server.
        
        Args:
            host: Server host
            port: Server port
        """
        pass
        
    def start(self) -> None:
        """Start web server."""
        pass
        
    def stop(self) -> None:
        """Stop web server."""
        pass
        
    def get_status(self) -> Dict[str, Any]:
        """Get server status.
        
        Returns:
            Dict[str, Any]: Server status information
        """
        pass
```

## Configuration

### Configuration Format

The system uses a JSON-based configuration format:

```json
{
    "motor": {
        "left_calibration": 1.0,
        "right_calibration": 1.0,
        "max_speed": 100,
        "acceleration_limit": 50,
        "pid": {
            "kp": 1.0,
            "ki": 0.1,
            "kd": 0.05
        }
    },
    "encoder": {
        "pulses_per_revolution": 20,
        "wheel_diameter_mm": 65.0,
        "wheel_base_mm": 150.0,
        "sample_time": 0.1
    },
    "safety": {
        "motor": {
            "max_current": 10.0,
            "current_threshold": 8.0,
            "overcurrent_time": 1.0
        },
        "battery": {
            "min_voltage": 11.0,
            "warning_voltage": 11.5,
            "max_current": 20.0,
            "update_rate": 10
        }
    }
}
```

## Error Codes

### Error Categories

1. **Hardware Errors (1xxx)**
   - 1001: Motor error
   - 1002: Encoder error
   - 1003: Sensor error
   - 1004: GPIO error

2. **Software Errors (2xxx)**
   - 2001: Configuration error
   - 2002: Communication error
   - 2003: Processing error
   - 2004: Memory error

3. **Safety Errors (3xxx)**
   - 3001: Overcurrent
   - 3002: Overheating
   - 3003: Low battery
   - 3004: Obstacle detected

### Error Severity Levels

1. **DEBUG**
   - Informational messages
   - Development details
   - Diagnostic information

2. **INFO**
   - Normal operation
   - Status updates
   - Configuration changes

3. **WARNING**
   - Potential issues
   - Performance degradation
   - Resource constraints

4. **ERROR**
   - Operation failures
   - System errors
   - Hardware issues

5. **CRITICAL**
   - System failures
   - Safety violations
   - Emergency conditions

## Best Practices

### Coding Guidelines

1. **Style**
   - Follow PEP 8
   - Use type hints
   - Write docstrings
   - Keep functions focused

2. **Error Handling**
   - Use specific exceptions
   - Provide error context
   - Log errors properly
   - Handle cleanup

3. **Performance**
   - Optimize critical paths
   - Use appropriate data structures
   - Minimize object creation
   - Profile code regularly

4. **Safety**
   - Validate inputs
   - Check resource limits
   - Implement timeouts
   - Handle edge cases

### Usage Examples

1. **Motor Control**
   ```python
   # Initialize motor controller
   config = MotorConfig(
       max_speed=100,
       acceleration=50,
       pid_params={"kp": 1.0, "ki": 0.1, "kd": 0.05}
   )
   controller = MotorController(config)
   
   # Move robot
   controller.move(left_speed=50, right_speed=50)
   
   # Stop robot
   controller.stop()
   ```

2. **Safety Monitoring**
   ```python
   # Initialize safety monitor
   safety = SafetyMonitor(config)
   
   # Check safety
   level = safety.check_safety()
   if level == SafetyLevel.WARNING:
       safety.add_warning("motor", "High temperature")
   
   # Clear warning
   safety.clear_warning("motor")
   ```

3. **Navigation**
   ```python
   # Initialize navigation
   navigator = WaypointNavigator()
   
   # Add waypoints
   navigator.add_waypoint(latitude=37.7749, longitude=-122.4194)
   navigator.add_waypoint(latitude=37.7833, longitude=-122.4167)
   
   # Update position
   navigator.update_position()
   
   # Get next waypoint
   next_lat, next_lon = navigator.get_next_waypoint()
   ```

4. **Web Interface**
   ```python
   # Initialize web server
   server = WebServer(host='0.0.0.0', port=8080)
   
   # Start server
   server.start()
   
   # Get status
   status = server.get_status()
   
   # Stop server
   server.stop()
   ``` 