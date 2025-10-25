# PiRobot V.4 Configuration Guide

## Overview

This guide explains how to configure PiRobot V.4 for optimal performance. The system uses a JSON-based configuration file located at `config/config.txt`.

## Configuration Structure

The configuration file is organized into several sections:

```json
{
    "motor": {
        // Motor control settings
    },
    "navigation": {
        // Navigation settings
    },
    "safety": {
        // Safety settings
    },
    "network": {
        // Network settings
    },
    "logging": {
        // Logging settings
    }
}
```

## Motor Configuration

### Basic Settings

```json
"motor": {
    "left_calibration": 1.0,      // Left motor calibration factor
    "right_calibration": 1.0,     // Right motor calibration factor
    "max_speed": 100,             // Maximum speed (0-100)
    "acceleration_limit": 50,     // Maximum acceleration (%/s)
    "pid": {
        "kp": 1.0,                // Proportional gain
        "ki": 0.1,                // Integral gain
        "kd": 0.05                // Derivative gain
    }
}
```

### Encoder Settings

```json
"encoder": {
    "pulses_per_revolution": 20,  // Encoder pulses per revolution
    "wheel_diameter_mm": 65.0,    // Wheel diameter in millimeters
    "wheel_base_mm": 150.0,       // Distance between wheels
    "sample_time": 0.1            // Speed calculation interval
}
```

### Temperature Settings

```json
"temperature": {
    "max_temp": 80.0,            // Maximum temperature (°C)
    "warning_temp": 70.0,        // Warning temperature (°C)
    "temp_coefficient": 0.01,    // ADC to temperature conversion
    "adc_channels": {
        "left": 0,               // Left motor ADC channel
        "right": 1               // Right motor ADC channel
    }
}
```

## Navigation Configuration

### Lane Detection

```json
"lane_detection": {
    "camera_id": 0,              // Camera device ID
    "frame_width": 640,          // Frame width
    "frame_height": 480,         // Frame height
    "roi_height": 0.6,           // Region of interest height ratio
    "min_line_length": 100,      // Minimum line length
    "max_line_gap": 50           // Maximum line gap
}
```

### Waypoint Navigation

```json
"waypoint": {
    "radius": 5.0,               // Waypoint radius (meters)
    "max_curvature": 0.5,        // Maximum path curvature
    "update_rate": 10,           // Navigation update rate (Hz)
    "gps": {
        "port": "/dev/ttyUSB0",  // GPS serial port
        "baud_rate": 9600,       // GPS baud rate
        "timeout": 1.0           // GPS timeout (seconds)
    }
}
```

## Safety Configuration

### Motor Safety

```json
"safety": {
    "motor": {
        "max_current": 10.0,     // Maximum motor current (A)
        "current_threshold": 8.0, // Current warning threshold
        "overcurrent_time": 1.0   // Overcurrent time limit (s)
    }
}
```

### Battery Safety

```json
"safety": {
    "battery": {
        "min_voltage": 11.0,     // Minimum battery voltage
        "warning_voltage": 11.5,  // Warning voltage
        "max_current": 20.0,     // Maximum current draw
        "update_rate": 10        // Battery check rate (Hz)
    }
}
```

### Obstacle Safety

```json
"safety": {
    "obstacle": {
        "distance_threshold": 0.5,  // Minimum safe distance (m)
        "warning_distance": 1.0,    // Warning distance (m)
        "update_rate": 20           // Obstacle check rate (Hz)
    }
}
```

## Network Configuration

### Web Interface

```json
"network": {
    "web": {
        "host": "0.0.0.0",       // Web server host
        "port": 8080,            // Web server port
        "debug": false,          // Debug mode
        "ssl": {
            "enabled": false,    // SSL enabled
            "cert_file": "",     // SSL certificate file
            "key_file": ""       // SSL key file
        }
    }
}
```

### Logging

```json
"logging": {
    "level": "INFO",            // Logging level
    "file": "robot.log",        // Log file
    "max_size": 10485760,       // Maximum log size (bytes)
    "backup_count": 5,          // Number of backup files
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
}
```

## Example Configuration

Here's a complete example configuration:

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
    "temperature": {
        "max_temp": 80.0,
        "warning_temp": 70.0,
        "temp_coefficient": 0.01,
        "adc_channels": {
            "left": 0,
            "right": 1
        }
    },
    "navigation": {
        "lane_detection": {
            "camera_id": 0,
            "frame_width": 640,
            "frame_height": 480,
            "roi_height": 0.6,
            "min_line_length": 100,
            "max_line_gap": 50
        },
        "waypoint": {
            "radius": 5.0,
            "max_curvature": 0.5,
            "update_rate": 10,
            "gps": {
                "port": "/dev/ttyUSB0",
                "baud_rate": 9600,
                "timeout": 1.0
            }
        }
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
        },
        "obstacle": {
            "distance_threshold": 0.5,
            "warning_distance": 1.0,
            "update_rate": 20
        }
    },
    "network": {
        "web": {
            "host": "0.0.0.0",
            "port": 8080,
            "debug": false,
            "ssl": {
                "enabled": false,
                "cert_file": "",
                "key_file": ""
            }
        }
    },
    "logging": {
        "level": "INFO",
        "file": "robot.log",
        "max_size": 10485760,
        "backup_count": 5,
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    }
}
```

## Configuration Best Practices

1. **Backup Your Configuration**
   - Always keep a backup of your working configuration
   - Use version control for configuration files
   - Document any changes made

2. **Testing**
   - Test new configurations in a safe environment
   - Start with conservative values
   - Gradually adjust parameters
   - Monitor system behavior

3. **Safety First**
   - Never disable safety features
   - Set appropriate limits for your hardware
   - Regular safety checks
   - Monitor system logs

4. **Performance Tuning**
   - Start with default values
   - Adjust one parameter at a time
   - Document the effects of changes
   - Use the auto-tuning features

5. **Maintenance**
   - Regular configuration reviews
   - Update settings based on usage patterns
   - Monitor system performance
   - Keep documentation up to date

## Troubleshooting

### Common Issues

1. **Motor Issues**
   - Check calibration values
   - Verify PID parameters
   - Monitor temperature settings
   - Check current limits

2. **Navigation Problems**
   - Verify GPS settings
   - Check camera configuration
   - Adjust waypoint parameters
   - Monitor obstacle settings

3. **Safety Alerts**
   - Review safety thresholds
   - Check battery settings
   - Verify temperature limits
   - Monitor current limits

### Getting Help

If you encounter issues:
1. Check the logs
2. Review the documentation
3. Consult the troubleshooting guide
4. Contact support if needed 