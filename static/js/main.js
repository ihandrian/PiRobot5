document.addEventListener('DOMContentLoaded', () => {
    // WebSocket connection
    const socket = new WebSocket(`ws://${location.hostname}:5003`);
    
    socket.onopen = () => {
        console.log('WebSocket connected');
        new JoystickController(
            document.getElementById('joystick'),
            (x, y) => {
                const speed = document.getElementById('speed').value / 100;
                socket.send(JSON.stringify({
                    type: 'joystick',
                    x: x,
                    y: y,
                    speed: speed
                }));
            }
        );
    };

    // Initialize all controllers
    navigationController.init();
    statusUpdater.init();
    
    // ... rest of the initialization code ...
}); 