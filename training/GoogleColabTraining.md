# Google Colab Training Guide for PiRobot V.4

This guide explains how to train your PiRobot for autonomous navigation using Google Colab. The process involves three main steps:
1. Data Collection (Manual Mode)
2. Training in Google Colab
3. Deploying the Trained Model

## 1. Data Collection (Manual Mode)

### Prerequisites
- Raspberry Pi 3B with PiRobot V.4 installed
- USB Camera connected
- Ultrasonic sensors properly configured
- Sufficient storage space for data collection

### Setting Up Data Collection

1. **Start the Data Collection Interface**:
   ```bash
   python src/core/data_collector.py
   ```

2. **Data Collection UI Controls**:
   - **Start/Stop Recording**: Toggle button to start/stop data collection
   - **Save Current Frame**: Button to manually save important frames
   - **Emergency Stop**: Button to immediately stop all operations
   - **Status Display**: Shows current collection status and statistics

3. **Manual Navigation**:
   - Use the manual control interface to navigate the robot
   - The system will automatically collect:
     - Camera images
     - Ultrasonic sensor readings
     - Position data
     - Motor commands
     - Timestamps

4. **Data Storage**:
   - Data is stored in `training/collected_data/`
   - Each session creates a new timestamped folder
   - Data is saved in JSON format for easy processing

### Best Practices for Data Collection
- Navigate through different room layouts
- Include various obstacle configurations
- Collect data at different times of day
- Include both successful and failed navigation attempts
- Ensure good lighting conditions
- Collect at least 30 minutes of data per room layout

## 2. Training in Google Colab

### Setting Up Google Colab

1. **Create a New Colab Notebook**:
   - Go to [Google Colab](https://colab.research.google.com)
   - Create a new notebook
   - Set runtime type to GPU (Runtime > Change runtime type > GPU)

2. **Upload Data**:
   ```python
   from google.colab import files
   uploaded = files.upload()  # Upload your collected data
   ```

3. **Install Dependencies**:
   ```python
   !pip install -r requirements.txt
   ```

4. **Run Training**:
   ```python
   from training.mapping.train import main
   
   # Configure training parameters
   config = {
       'data_dir': 'collected_data',
       'model_save_path': 'models/room_mapper.pth',
       'map_save_path': 'maps/room_map.json',
       'epochs': 50,  # Adjust based on your data size
       'batch_size': 32,
       'learning_rate': 0.001,
       'map_size': (100, 100),
       'resolution': 0.1
   }
   
   main(config)
   ```

### Monitoring Training Progress
- Use TensorBoard to monitor training:
  ```python
  %load_ext tensorboard
  %tensorboard --logdir runs
  ```
- Check loss curves and validation metrics
- Monitor GPU utilization
- Save checkpoints periodically

### Evaluating the Model
```python
from training.mapping.room_mapper import RoomMapper
import matplotlib.pyplot as plt

# Load trained model
mapper = RoomMapper()
mapper.load_model('models/room_mapper.pth')

# Test on sample data
test_path = mapper.find_path(start=(0, 0), goal=(5, 5))
mapper.visualize_map(path=test_path)
```

## 3. Deploying the Trained Model

### Downloading Results
1. From Google Colab:
   ```python
   from google.colab import files
   files.download('models/room_mapper.pth')
   files.download('maps/room_map.json')
   ```

2. Save to Raspberry Pi:
   - Copy files to `src/models/` directory
   - Ensure proper permissions:
     ```bash
     chmod 644 src/models/room_mapper.pth
     chmod 644 src/models/room_map.json
     ```

### Testing on Raspberry Pi
1. **Load the Model**:
   ```python
   from src.core.room_mapper import RoomMapper
   
   mapper = RoomMapper()
   mapper.load_model('src/models/room_mapper.pth')
   mapper.load_map('src/models/room_map.json')
   ```

2. **Start Autonomous Mode**:
   ```bash
   python src/core/autonomous_navigator.py
   ```

### Troubleshooting
- If the model performs poorly:
  1. Collect more training data
  2. Adjust training parameters
  3. Check data quality
  4. Verify sensor calibration

- Common Issues:
  - Memory errors: Reduce batch size
  - Slow performance: Optimize model architecture
  - Poor navigation: Collect more diverse data

## Additional Resources
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [OpenCV Documentation](https://docs.opencv.org/4.x/)
- [Raspberry Pi GPIO Documentation](https://www.raspberrypi.org/documentation/usage/gpio/)

## Support
For issues or questions:
1. Check the [GitHub Issues](https://github.com/your-repo/issues)
2. Join the [Discord Community](your-discord-link)
3. Email support@pirobot.com 