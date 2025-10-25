#!/bin/bash

# PiRobot5 - Raspberry Pi 5 Startup Script
# Optimized for Pi 5 performance and features

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print status messages
print_status() {
    echo -e "${GREEN}[*] $1${NC}"
}

# Function to print error messages
print_error() {
    echo -e "${RED}[!] $1${NC}"
}

# Function to print warning messages
print_warning() {
    echo -e "${YELLOW}[!] $1${NC}"
}

# Function to print info messages
print_info() {
    echo -e "${BLUE}[i] $1${NC}"
}

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    print_error "Please run as root for optimal Pi 5 performance"
    exit 1
fi

print_header() {
    echo -e "${BLUE}"
    echo "============================================================"
    echo " $1"
    echo "============================================================"
    echo -e "${NC}"
}

print_header "PiRobot5 - Raspberry Pi 5 Startup"

# Check if running on Pi 5
print_info "Checking system compatibility..."
if ! grep -q "Pi 5" /proc/device-tree/model 2>/dev/null; then
    print_warning "Not running on Pi 5 - some optimizations may not apply"
fi

# Check if virtual environment exists
if [ ! -d "PiRobot" ]; then
    print_error "Virtual environment not found. Please run setup_pi.sh first."
    exit 1
fi

# Activate virtual environment
print_status "Activating virtual environment..."
source PiRobot/bin/activate

# Check Python version
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
if [[ $python_version < "3.9" ]]; then
    print_error "Python 3.9 or higher is required for Pi 5. Current version: $python_version"
    exit 1
fi

print_status "Python version: $python_version ✓"

# Pi 5 specific optimizations
print_status "Applying Pi 5 specific optimizations..."

# Set CPU governor to performance
echo "performance" | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor > /dev/null 2>&1

# Set real-time priority
print_info "Setting real-time priority for robot processes..."

# Check required packages
print_status "Checking required packages..."
required_packages=("torch" "cv2" "numpy" "flask" "RPi.GPIO" "psutil")
missing_packages=()

for package in "${required_packages[@]}"; do
    if ! python3 -c "import $package" 2>/dev/null; then
        missing_packages+=("$package")
    fi
done

if [ ${#missing_packages[@]} -ne 0 ]; then
    print_error "Missing required packages: ${missing_packages[*]}"
    print_info "Please run: pip install ${missing_packages[*]}"
    exit 1
fi

print_status "All required packages installed ✓"

# Check system resources
print_status "Checking system resources..."
memory_gb=$(free -g | awk '/^Mem:/{print $2}')
if [ $memory_gb -lt 4 ]; then
    print_warning "Less than 4GB RAM detected. Pi 5 typically has 4GB or 8GB."
fi

# Check temperature
temp=$(cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null)
if [ ! -z "$temp" ]; then
    temp_c=$((temp/1000))
    print_info "CPU Temperature: ${temp_c}°C"
    if [ $temp_c -gt 80 ]; then
        print_warning "High temperature detected. Consider active cooling."
    fi
fi

# Check camera availability
print_status "Checking camera availability..."
cameras_found=0
for i in {0..4}; do
    if [ -e "/dev/video$i" ]; then
        cameras_found=$((cameras_found + 1))
    fi
done

if [ $cameras_found -eq 0 ]; then
    print_warning "No cameras detected. Camera functionality may be limited."
else
    print_status "Found $cameras_found camera(s) ✓"
fi

# Check GPIO availability
print_status "Checking GPIO availability..."
if [ ! -d "/sys/class/gpio" ]; then
    print_warning "GPIO interface not available. Hardware control may be limited."
else
    print_status "GPIO interface available ✓"
fi

# Pi 5 specific performance tuning
print_status "Applying Pi 5 performance tuning..."

# Set process priority
renice -n -5 $$ 2>/dev/null

# Optimize memory settings
echo 1 > /proc/sys/vm/drop_caches 2>/dev/null

# Set I/O scheduler for better performance
echo deadline > /sys/block/mmcblk0/queue/scheduler 2>/dev/null

# Check if Pi 5 specific optimizations are enabled
print_info "Checking Pi 5 optimizations..."
if grep -q "arm_freq=2400" /boot/config.txt 2>/dev/null; then
    print_status "Pi 5 CPU overclocking enabled ✓"
else
    print_warning "Pi 5 CPU overclocking not detected. Consider running setup_pi.sh"
fi

# Start the robot
print_header "Starting PiRobot5"
print_status "Launching robot with Pi 5 optimizations..."

# Set environment variables for Pi 5
export PYTHONOPTIMIZE=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENCV_VIDEOIO_PRIORITY_MSMF=0

# Run the robot
python main.py

# Handle exit
exit_code=$?
if [ $exit_code -ne 0 ]; then
    print_error "PiRobot failed to start (exit code: $exit_code)"
    print_info "Check logs for details: journalctl -u pirobot -f"
    exit $exit_code
else
    print_status "PiRobot shutdown completed successfully"
fi
