# PiRobot V.4 Wiring Diagram

## Hardware Components
- Raspberry Pi 3B (Main Controller)
- L298N H-Bridge Motor Driver
- HC-SR04 Ultrasonic Sensor
- Air530Z GPS Module
- Logitech C270 Webcam (USB)
- Raspberry Pi Camera Module (CSI)
- Google Coral TPU (USB)

## Power System
```
Main Power Supply (12V, 2.5A)
        |
        +---> L298N Motor Driver (12V)
        |
        +---> 5V Regulator
                |
                +---> Raspberry Pi 3B
                |
                +---> HC-SR04
                |
                +---> Air530Z GPS
                |
                +---> Camera Module
```

## Component Connections

### L298N Motor Driver
```
Raspberry Pi 3B                     L298N H-Bridge
+----------------+                 +----------------+
|                |                 |                |
| GPIO2 (PWM)    |----------------| ENA (Enable A) |
| GPIO3 (PWM)    |----------------| ENB (Enable B) |
| GPIO4 (Dir)    |----------------| IN1 (Motor A)  |
| GPIO17 (Dir)   |----------------| IN2 (Motor A)  |
|                |                 |                |
+----------------+                 +----------------+

L298N Power Connections:
+----------------+                 +----------------+
|                |                 |                |
| 5V             |----------------| +5V (Logic)    |
| 12V            |----------------| +12V (Motor)   |
| GND            |----------------| GND            |
|                |                 |                |
+----------------+                 +----------------+
```

### HC-SR04 Ultrasonic Sensor
```
Raspberry Pi 3B                     HC-SR04
+----------------+                 +----------------+
|                |                 |                |
| GPIO23 (Trig)  |----------------| TRIG           |
| GPIO24 (Echo)  |----------------| ECHO           |
| 5V             |----------------| VCC            |
| GND            |----------------| GND            |
|                |                 |                |
+----------------+                 +----------------+

Voltage Divider for Echo:
+----------------+                 +----------------+
|                |                 |                |
| Echo (5V)      |----[1kΩ]-------| GPIO24 (3.3V) |
|                |        |        |                |
|                |       [2kΩ]     |                |
|                |        |        |                |
| GND            |--------+--------| GND            |
+----------------+                 +----------------+
```

### Air530Z GPS Module
```
Raspberry Pi 3B                     Air530Z GPS
+----------------+                 +----------------+
|                |                 |                |
| GPIO14 (TXD)   |----------------| RXD            |
| GPIO15 (RXD)   |----------------| TXD            |
| 5V             |----------------| VCC            |
| GND            |----------------| GND            |
|                |                 |                |
+----------------+                 +----------------+
```

### Camera Setup
```
Raspberry Pi 3B                     Cameras
+----------------+                 +----------------+
|                |                 |                |
| CSI Port       |----------------| Pi Camera      |
| USB Port 1     |----------------| Logitech C270  |
| USB Port 2     |----------------| Coral TPU      |
|                |                 |                |
+----------------+                 +----------------+
```

## Safety Features
```
Raspberry Pi 3B                     Safety Components
+----------------+                 +----------------+
|                |                 |                |
| GPIO27 (IN)    |----------------| Emergency Stop |
| GPIO5 (PWM)    |----------------| Status LED     |
| GPIO6 (PWM)    |----------------| Warning LED    |
|                |                 |                |
+----------------+                 +----------------+
```

## Protection Components
```
Power Protection:
+----------------+                 +----------------+
|                |                 |                |
| 12V Input      |----[2A Fuse]---| L298N Power    |
| 5V Input       |----[1A Fuse]---| Logic Power    |
|                |                 |                |
+----------------+                 +----------------+

Motor Protection:
+----------------+                 +----------------+
|                |                 |                |
| Motor A        |----[Diode]-----| L298N OUT1     |
| Motor B        |----[Diode]-----| L298N OUT2     |
|                |                 |                |
+----------------+                 +----------------+
```

## Grounding Scheme
```
Star Ground Point
        |
        +---> Raspberry Pi GND
        |
        +---> L298N Logic GND
        |
        +---> L298N Motor GND
        |
        +---> Sensor GND
        |
        +---> USB Device GND
```

## Notes
1. All power connections should use appropriate wire gauges:
   - 14-16 AWG for motor power
   - 18-22 AWG for logic power
   - 22-24 AWG for signal wires

2. Protection components:
   - Add 100nF capacitors near ICs
   - Use Schottky diodes for motor protection
   - Implement TVS diodes for ESD protection

3. Signal integrity:
   - Use shielded cables for sensitive signals
   - Keep motor wires away from signal wires
   - Implement proper cable management

4. Power sequencing:
   - Power up logic circuits first
   - Enable motor power after system check
   - Implement proper shutdown sequence 