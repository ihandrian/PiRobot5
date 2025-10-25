class JoystickController {
    constructor(container, onChange) {
        this.container = container;
        this.knob = container.querySelector('#knob');
        this.bounds = container.getBoundingClientRect();
        this.center = {
            x: this.bounds.width / 2,
            y: this.bounds.height / 2
        };
        this.active = false;
        this.onChange = onChange;
        this.setupEvents();
    }

    // ... rest of the joystick code ...
} 