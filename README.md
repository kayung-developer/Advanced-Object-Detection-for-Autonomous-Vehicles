# Advanced Object Detection for Autonomous Vehicles

An Andvance object detection and tracking techniques to process video or image feeds for autonomous vehicle systems. It integrates a YOLOv4 model for object detection and DeepSORT for object tracking.

## Features

- Real-time object detection from video or image.
- Object tracking using DeepSORT.
- Customizable confidence threshold for detection.
- Export detection logs as CSV files.
- Toggle between Dark and Light themes for the UI.

## Requirements

To set up this application, you need the following dependencies:

- Python 3.7+
- `opencv-python`
- `customtkinter`
- `Pillow`
- `pandas`
- `numpy`
- `deepsort`
- `tensorflow` or `pytorch` (for YOLOv4, depending on your preference)

### Installing Dependencies

You can install the required libraries using `pip`:

```bash
pip install opencv-python customtkinter Pillow pandas numpy deepsort tensorflow
```
Additionally, you will need to download the YOLOv4 model files for the object detection:
since its already available in the models folder.
- `yolov4.weights`
- `yolov4.cfg`
These can be obtained from the official YOLOv4 website or other trusted sources.

### Folder Structure

`
├── assets
│   ├── open-camera.png
│   ├── no-camera.png
│   ├── video-recoder.png
│   ├── image-download.png
│   └── download.png
├── models
│   ├── yolov4.weights
│   └── yolov4.cfg
├── main.py
└── README.md
`

- Place the `yolov4.weights` and `yolov4.cfg` files in the models folder.

### Usage
1. Starting the Application: Run the Python script main.py:
```bash
python main.py
```

2. UI Controls:

- Start Detection: Begin object detection using the camera.
- Stop Detection: Stop object detection.
- Detect from Video: Choose a video file for object detection.
- Upload Image: Select an image file to detect objects.
- Export Logs: Save the detection logs in CSV format.
- Toggle Theme: Switch between Light and Dark themes.

### Confidence Threshold: 
You can adjust the confidence threshold for detection using the slider. The default is 0.5.

### Running the Detection
- Real-Time Camera Feed: When you click the "Start Detection" button, the app will begin processing the feed from your webcam. The objects detected in the feed will be highlighted with bounding boxes.
- Video File Detection: Click "Detect from Video" to select a video file for object detection. The application will process the video frame by frame.
- Image Detection: Click "Upload Image" to upload a static image, and the app will process the image for object detection.

### DeepSORT Tracking
This application uses the DeepSORT algorithm to track the objects detected. This allows the system to track multiple objects over time, even if they move in and out of the frame.

### Exporting Logs
You can export the detection logs as a CSV file by clicking the Export Logs button. The logs contain details about the detected objects, including the class labels and detection confidence scores.

### Troubleshooting
- **Model Files Missing:** Ensure that the `yolov4.weights` and `yolov4.cfg` files are placed correctly in the models folder.
- **Low FPS:** If you experience slow performance, consider reducing the resolution of the input feed or using a more powerful machine.


### License
This project is licensed under the MIT License - see the LICENSE file for details.


Note: This app is intended for research and development in autonomous vehicle systems and object detection. It requires proper hardware for real-time video processing, especially for large-scale deployments.



### Explanation:

1. **Introduction**: Describes the purpose of the application.
2. **Requirements**: Lists the necessary Python libraries and dependencies.
3. **Installation**: Details how to install the dependencies via `pip`.
4. **Folder Structure**: Provides the expected directory structure of the project.
5. **Usage**: Provides step-by-step instructions on how to use the app.
6. **DeepSORT Tracking**: Explains the usage of the DeepSORT tracker in the application.
7. **Exporting Logs**: Details how to export detection logs as CSV files.
8. **Troubleshooting**: Common issues users might encounter.
9. **License**: Mentions the MIT License for the project.

Feel free to update it as necessary for your specific setup or environment!
