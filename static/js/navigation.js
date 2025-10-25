class NavigationController {
    constructor() {
        this.currentMode = 'manual';
        this.setupEventListeners();
    }

    setupEventListeners() {
        document.querySelectorAll('.mode-btn').forEach(btn => {
            btn.addEventListener('click', () => this.handleModeChange(btn));
        });
    }

    async sendMotorCommand(direction, speed) {
        try {
            const response = await fetch('/motor', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/x-www-form-urlencoded',
                },
                body: `direction=${direction}&speed=${speed}`
            });
            const data = await response.json();
            if (!data.status === 'ok') {
                console.error('Motor command failed');
            }
        } catch (e) {
            console.error('Failed to send motor command:', e);
        }
    }

    async calibrateMotors() {
        try {
            const response = await fetch('/calibrate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/x-www-form-urlencoded',
                },
                body: 'action=start'
            });
            const data = await response.json();
            if (data.status === 'ok') {
                alert('Calibration completed: ' + data.message);
            } else {
                alert('Calibration failed: ' + data.message);
            }
        } catch (e) {
            console.error('Calibration failed:', e);
            alert('Calibration failed: ' + e.message);
        }
    }

    // ... rest of the navigation functions ...
}

const navigationController = new NavigationController(); 