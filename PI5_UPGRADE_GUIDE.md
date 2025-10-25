# PiRobot V.4 - Raspberry Pi 5 Upgrade Guide

This guide will help you upgrade your PiRobot V.4 from Raspberry Pi 3B to Raspberry Pi 5, taking advantage of the improved performance and new features.

## Overview of Changes

### Hardware Improvements
- **CPU**: Upgraded from 1.2GHz Cortex-A53 to 2.4GHz Cortex-A76 (2x performance)
- **GPU**: Upgraded from VideoCore IV to VideoCore VII with OpenGL ES 3.1 and Vulkan 1.2
- **RAM**: Increased from 1GB to 4GB/8GB options
- **Connectivity**: Dual-band Wi-Fi, Bluetooth 5.0, Gigabit Ethernet, USB 3.0
- **Power**: Requires 5V 4A power supply (vs 2.5A for Pi 3B)

### Software Updates
- Updated Python packages for ARM64 compatibility
- Optimized system configurations for Pi 5
- Enhanced performance monitoring
- Improved camera support (dual cameras)
- Better real-time performance

## Pre-Upgrade Checklist

### 1. Backup Your Data
```bash
# Create backup of your current setup
sudo tar -czf pirobot_backup_$(date +%Y%m%d).tar.gz /home/pi/PiRobot5/
```

### 2. Check Hardware Compatibility
- Ensure your motor drivers and sensors are Pi 5 compatible
- Verify camera modules work with Pi 5
- Check that your power supply can provide 5V 4A

### 3. Prepare Pi 5
- Flash latest Raspberry Pi OS (64-bit recommended)
- Enable SSH and configure network
- Update system packages

## Upgrade Steps

### Step 1: Clone Updated Repository
```bash
# Remove old version (after backup)
rm -rf /home/pi/PiRobot5

# Clone updated version
git clone https://github.com/ihandrian/PiRobot-V.4.git /home/pi/PiRobot5
cd /home/pi/PiRobot5
```

### Step 2: Run Pi 5 Setup Script
```bash
# Make setup script executable
chmod +x setup_pi.sh

# Run setup (as root)
sudo ./setup_pi.sh
```

### Step 3: Test Installation
```bash
# Run Pi 5 specific tests
python test_installation.py
```

### Step 4: Configure Hardware
```bash
# Copy Pi 5 hardware configuration
cp config/hardware.yaml config/hardware.yaml

# Edit configuration for your specific hardware
nano config/hardware.yaml
```

### Step 5: Start Robot
```bash
# Use Pi 5 optimized startup script
sudo ./start_robot.sh
```

## Configuration Changes

### System Configuration (`/boot/config.txt`)
The setup script automatically adds these Pi 5 optimizations:
```
# Pi 5 specific optimizations
arm_freq=2400
over_voltage=2
gpu_freq=800
sdram_freq=5000
dtparam=pciex1
camera_auto_detect=1
```

### Service Configuration
The systemd service has been updated with Pi 5 optimizations:
- Higher process priority
- Optimized I/O scheduling
- Multi-threading environment variables

### Hardware Configuration
- Updated GPIO pin assignments for Pi 5 compatibility
- Enhanced camera support for dual cameras
- Improved power management settings

## Performance Improvements

### Expected Performance Gains
- **CPU Performance**: 2-3x faster processing
- **Memory**: 4-8x more RAM available
- **Camera**: Support for higher resolutions and dual cameras
- **Real-time**: Better real-time performance for motor control
- **AI/ML**: Faster inference with optimized PyTorch

### Monitoring Performance
```bash
# Monitor system performance
htop

# Check temperature
cat /sys/class/thermal/thermal_zone0/temp

# Monitor GPU usage
vcgencmd measure_temp
vcgencmd get_mem gpu
```

## Troubleshooting

### Common Issues

#### 1. High Temperature
**Symptoms**: Robot shuts down or throttles performance
**Solution**: 
- Install active cooling (Pi 5 active cooler recommended)
- Check thermal paste application
- Monitor with: `watch -n 1 cat /sys/class/thermal/thermal_zone0/temp`

#### 2. Power Issues
**Symptoms**: Random shutdowns or instability
**Solution**:
- Use official Pi 5 power supply (5V 4A)
- Check power supply quality
- Monitor voltage: `vcgencmd measure_volts`

#### 3. Camera Not Detected
**Symptoms**: No camera feed or detection errors
**Solution**:
- Check camera connections
- Enable camera in raspi-config
- Test with: `libcamera-hello --list-cameras`

#### 4. GPIO Issues
**Symptoms**: Motor control or sensor errors
**Solution**:
- Check GPIO pin assignments
- Verify hardware connections
- Test GPIO: `python -c "import RPi.GPIO as GPIO; GPIO.setmode(GPIO.BCM)"`

### Performance Optimization

#### 1. CPU Optimization
```bash
# Set CPU governor to performance
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Check current frequency
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq
```

#### 2. Memory Optimization
```bash
# Check memory usage
free -h

# Optimize swap usage
sudo swapon --show
```

#### 3. I/O Optimization
```bash
# Set I/O scheduler
echo deadline | sudo tee /sys/block/mmcblk0/queue/scheduler

# Check I/O performance
sudo iotop
```

## Advanced Features

### Dual Camera Support
Pi 5 supports dual cameras for enhanced vision:
```python
# Primary camera
camera_primary = cv2.VideoCapture(0)

# Secondary camera  
camera_secondary = cv2.VideoCapture(1)
```

### M.2 SSD Support
Pi 5 supports M.2 NVMe SSDs for faster storage:
```bash
# Check if M.2 is detected
lsblk

# Mount M.2 SSD
sudo mkdir /mnt/nvme
sudo mount /dev/nvme0n1p1 /mnt/nvme
```

### Enhanced Networking
Pi 5 has improved networking capabilities:
- 5GHz Wi-Fi support
- Gigabit Ethernet
- Bluetooth 5.0

## Migration Checklist

- [ ] Backup current Pi 3B setup
- [ ] Prepare Pi 5 with latest OS
- [ ] Clone updated repository
- [ ] Run Pi 5 setup script
- [ ] Test all hardware components
- [ ] Configure for your specific hardware
- [ ] Run performance tests
- [ ] Start robot and verify operation
- [ ] Monitor performance and temperature
- [ ] Document any custom configurations

## Support

If you encounter issues during the upgrade:

1. Check the troubleshooting section above
2. Review the test output from `test_installation.py`
3. Check system logs: `journalctl -u pirobot -f`
4. Verify hardware connections and compatibility
5. Ensure proper power supply (5V 4A)

## Performance Monitoring

### Key Metrics to Monitor
- CPU temperature (should stay below 80°C)
- CPU usage (should be reasonable for your workload)
- Memory usage (4GB+ available on Pi 5)
- GPU temperature and usage
- I/O performance
- Network connectivity

### Monitoring Commands
```bash
# System overview
htop

# Temperature monitoring
watch -n 1 'cat /sys/class/thermal/thermal_zone0/temp'

# Memory usage
free -h

# Disk usage
df -h

# Network status
ip addr show
```

## Conclusion

The upgrade to Raspberry Pi 5 provides significant performance improvements and new capabilities for your PiRobot. The enhanced CPU, GPU, and memory capabilities will allow for more sophisticated AI/ML processing, better real-time performance, and support for advanced features like dual cameras.

Follow this guide carefully, and your PiRobot will be running optimally on the Pi 5 platform.
