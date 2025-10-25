class StatusUpdater {
    constructor() {
        this.updateInterval = 2000;
        this.startUpdates();
    }

    async updateStatus() {
        try {
            const systemInfo = await fetch('/system_info').then(r => r.json());
            this.updateSystemInfo(systemInfo);
            this.updateGPSInfo(systemInfo.gps);
        } catch (e) {
            console.error('Status update failed:', e);
        }
    }

    // ... rest of the status update functions ...
}

const statusUpdater = new StatusUpdater(); 