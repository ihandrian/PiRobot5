class MapController {
    constructor() {
        this.map = null;
        this.markers = {};
    }

    initialize(containerId) {
        this.map = L.map(containerId).setView([0, 0], 2);
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '© OpenStreetMap contributors'
        }).addTo(this.map);
        return this.map;
    }

    // ... rest of the map functions ...
}

const mapController = new MapController(); 