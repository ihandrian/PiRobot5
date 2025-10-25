# Data Collector Documentation

## Overview

The Data Collector is a graphical interface for collecting training data for the PiRobot V.4 system. It captures camera frames and sensor data in real-time, providing a user-friendly way to gather training data for the room mapping and navigation systems.

## Features

- Real-time camera feed display
- Manual frame capture
- Automatic data collection
- Sensor data validation
- Emergency stop functionality
- Statistics tracking
- Configurable settings

## Configuration

The data collector is configured using `config/data_collector.yaml`. Key settings include:

```yaml
# Data storage settings
data_dir: "training/collected_data"

# Camera settings
camera:
  device_id: 0
  width: 640
  height: 480
  fps: 30

# Sensor settings
sensors:
  ultrasonic:
    count: 4
    min_distance: 0.02  # meters
    max_distance: 4.0   # meters
  position:
    update_rate: 10     # Hz
  motor:
    update_rate: 20     # Hz

# Data collection settings
collection:
  max_frames: 10000     # Maximum number of frames to collect
  frame_skip: 1         # Process every Nth frame
  save_interval: 100    # Save data every N frames
  compression: 80       # JPEG compression quality (0-100)
```

## Usage

### Starting the Collector

```bash
python src/core/data_collector.py
```

### Interface Controls

1. **Start/Stop Recording Button**
   - Toggles data collection
   - Changes color to indicate recording state
   - Updates status display

2. **Save Current Frame Button**
   - Manually saves the current frame and sensor data
   - Only works when recording is active
   - Creates both image and JSON files

3. **Emergency Stop Button**
   - Immediately stops all operations
   - Red button for visibility
   - Updates status to "Emergency Stop"

4. **Status Display**
   - Shows current system state
   - Updates in real-time
   - Indicates any errors or warnings

5. **Statistics Panel**
   - Shows number of frames collected
   - Shows number of sensor readings
   - Updates in real-time

### Data Format

#### Image Files
- Saved as JPEG format
- Filename format: `frame_[timestamp].jpg`
- Compression quality configurable

#### Sensor Data (JSON)
```json
{
    "timestamp": 1234567890.123,
    "frame_path": "path/to/frame.jpg",
    "sensor_data": {
        "ultrasonic": [0.0, 0.0, 0.0, 0.0],
        "position": [0.0, 0.0],
        "motor_commands": [0.0, 0.0]
    }
}
```

## Error Handling

The data collector includes comprehensive error handling:

1. **Initialization Errors**
   - Configuration file validation
   - Camera initialization
   - Directory creation
   - UI setup

2. **Runtime Errors**
   - Camera feed errors
   - File system errors
   - Sensor data validation
   - Memory management

3. **User Interface**
   - Error messages displayed in status panel
   - Emergency stop functionality
   - Graceful shutdown

## Best Practices

1. **Data Collection**
   - Collect data in various lighting conditions
   - Include different room layouts
   - Vary robot speeds and movements
   - Include obstacle scenarios

2. **Storage Management**
   - Monitor disk space usage
   - Regularly backup collected data
   - Clean up old sessions if needed
   - Use appropriate compression settings

3. **Performance**
   - Adjust frame skip rate if needed
   - Monitor system resources
   - Use appropriate resolution settings
   - Validate sensor data regularly

## Troubleshooting

### Common Issues

1. **Camera Not Detected**
   - Check camera connection
   - Verify device ID in config
   - Check camera permissions

2. **Storage Issues**
   - Verify directory permissions
   - Check available disk space
   - Validate file paths

3. **Performance Problems**
   - Adjust frame skip rate
   - Reduce resolution if needed
   - Check system resources

### Error Messages

- `Camera initialization error`: Check camera connection and config
- `Configuration error`: Verify YAML file format and settings
- `Storage error`: Check disk space and permissions
- `Sensor validation error`: Verify sensor connections and data format

## Development

### Adding New Features

1. **New Sensors**
   - Add sensor configuration
   - Implement data collection
   - Update validation
   - Modify data format

2. **UI Enhancements**
   - Add new controls
   - Modify layout
   - Update statistics
   - Add new displays

3. **Data Processing**
   - Add preprocessing
   - Implement filters
   - Add data augmentation
   - Modify storage format

### Testing

1. **Unit Tests**
   - Sensor data validation
   - File operations
   - UI components
   - Error handling

2. **Integration Tests**
   - Camera feed
   - Data collection
   - Storage operations
   - UI responsiveness

3. **Performance Tests**
   - Memory usage
   - CPU utilization
   - Storage speed
   - UI refresh rate 