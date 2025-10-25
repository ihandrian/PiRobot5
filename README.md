# PiRobot V.4 - Raspberry Pi 5 Edition

An advanced autonomous robot system with enhanced safety features, collision detection, and room mapping capabilities, optimized for Raspberry Pi 5.

## Features

- Enhanced safety monitoring with real-time sensor feedback
- Improved collision detection using ultrasonic sensors
- Room mapping and autonomous navigation
- Data collection and training system for machine learning
- Real-time camera feed and sensor data visualization
- Emergency stop functionality
- Configurable settings via YAML configuration
- Optimized for Raspberry Pi 5 performance with ARM64 support
- Leverages Pi 5's improved CPU, GPU, and memory capabilities

## Project Structure

```
PiRobot5/
├── config/
│   └── data_collector.yaml    # Configuration settings
├── src/
│   └── core/
│       ├── data_collector.py  # Data collection interface
│       ├── motor_controller.py # Motor control and safety
│       └── safety_monitor.py  # Safety monitoring system
├── training/
│   ├── mapping/
│   │   ├── room_mapper.py     # Room mapping implementation
│   │   ├── dataset.py         # Training dataset handling
│   │   └── train.py           # Training script
│   └── requirements.txt       # Training dependencies
├── docs/
│   ├── data_collector.md      # Data collector documentation
│   └── training.md            # Training system documentation
├── setup_pi.sh                # Raspberry Pi setup script
├── LICENSE                    # GPL-3.0 License
└── README.md                  # This file
```

## Installation

### Quick Setup (Recommended)

1. Clone the repository:
   ```bash
   git clone https://github.com/ihandrian/PiRobot5.git
   cd PiRobot5
   ```

2. Run the setup script (as root):
   ```bash
   chmod +x setup_pi.sh
   sudo ./setup_pi.sh
   ```

3. Reboot your Raspberry Pi:
   ```bash
   sudo reboot
   ```

4. After reboot, activate the virtual environment:
   ```bash
   source PiRobot/bin/activate
   ```

5. Verify the installation:
   ```bash
   python test_installation.py
   ```

### Manual Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/ihandrian/PiRobot5.git
   cd PiRobot5
   ```

2. Create and activate virtual environment:
   ```bash
   python3 -m venv PiRobot
   source PiRobot/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r training/requirements.txt
   ```

4. Configure the system:
   ```bash
   mkdir -p config
   # Edit config/data_collector.yaml as needed
   ```

## Usage

### Data Collection
1. Start the data collector:
   ```bash
   python src/core/data_collector.py
   ```

2. Use the interface to:
   - Start/stop recording
   - Save frames manually
   - View real-time sensor data
   - Monitor system status

### Training
1. Collect training data using the data collector
2. Upload data to Google Colab
3. Run the training script:
   ```bash
   python training/mapping/train.py
   ```

### Room Mapping
1. Deploy the trained model
2. Start the room mapper:
   ```bash
   python training/mapping/room_mapper.py
   ```

## Safety Features

- Real-time monitoring of:
  - Battery voltage
  - Motor temperature
  - Obstacle distances
  - Emergency stop button
- Automatic shutdown on critical conditions
- Hardware watchdog timer
- Thread-safe operations
- Optimized resource management for Raspberry Pi 3B

## Requirements

- Raspberry Pi 5 (optimized for)
- Python 3.9+ (Pi 5 supports newer Python versions)
- Camera module (Pi 5 supports dual cameras)
- Ultrasonic sensors
- Motor drivers
- Emergency stop button
- Battery monitoring circuit
- Minimum 1GB swap space (optimized for Pi 5's 4GB+ RAM)
- 128MB GPU memory (optimized for Pi 5's VideoCore VII)
- 5V 4A power supply (Pi 5 specific requirement)

1. Fork the repository at https://github.com/ihandrian/PiRobot5
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request
## System Optimizations

The setup script (`setup_pi.sh`) includes several optimizations for Raspberry Pi 5:
- Configured 1GB swap space (optimized for Pi 5's 4GB+ RAM)
- Optimized GPU memory allocation (128MB for VideoCore VII)
- Performance-oriented CPU governor with Pi 5 specific overclocking
- Hardware-accelerated OpenCV with ARM64 optimizations
- ARM64-compatible PyTorch version with improved performance
- Optional Coral TPU support
- Pi 5 specific CPU and GPU frequency optimizations
- Enhanced I/O scheduling for better real-time performance

## Contributing


## License

This project is licensed under the GPL-3.0 License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Free Software Foundation for the GPL-3.0 License
- Raspberry Pi Foundation for hardware support
- OpenCV and PyTorch communities for software libraries
- Google Coral team for TPU support

## Security Considerations

### 🔒 Security Best Practices

- **Network Security**: The robot runs web services on ports 5002 and 5003. Ensure your network is secure.
- **Access Control**: Change default passwords and use strong authentication.
- **Firewall**: Configure firewall rules to restrict access to robot services.
- **Updates**: Regularly update the system and dependencies for security patches.
- **Sensitive Data**: Never commit passwords, API keys, or personal data to the repository.

### 🛡️ Safety Features

- **Emergency Stop**: Hardware emergency stop button for immediate robot shutdown
- **Collision Detection**: Ultrasonic sensors prevent collisions
- **Temperature Monitoring**: Automatic shutdown on overheating
- **Battery Monitoring**: Low battery protection and warnings

### ⚠️ Important Security Notes

- This robot has physical movement capabilities - use responsibly
- Ensure proper physical barriers when testing
- Monitor robot behavior and have emergency stop accessible
- Keep firmware and software updated
- Use secure network connections

## Support

- Paypal: https://paypal.me/IrfanHandrian
- Buy me Coffee: https://buymeacoffee.com/handrianirv
# PiRobot5
