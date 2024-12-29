import cv2
import customtkinter as ctk
from tkinter import filedialog
import threading
import pandas as pd
from datetime import datetime
import os
from PIL import Image
from customtkinter import CTkImage
import numpy as np
from deepsort import DeepSortTracker  # Import DeepSORT tracker
import time


class ObjectDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Object Detection for Autonomous Vehicles")
        self.root.geometry("950x650")

        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("blue")

        self.video_capture = None
        self.detection_running = False
        self.detection_log = []
        self.frame_count = 0
        self.start_time = None
        self.confidence_threshold = 0.5  # Default confidence threshold

        self.deep_sort_tracker = DeepSortTracker()  # Initialize DeepSORT tracker

        self.setup_ui()
        self.download_model()

    def setup_ui(self):
        # Left Frame - Video Feed
        self.left_frame = ctk.CTkFrame(self.root, width=700, height=600)
        self.left_frame.grid(row=0, column=0, padx=10, pady=10, rowspan=3)

        self.video_label = ctk.CTkLabel(self.left_frame, text="", width=700, height=500)
        self.video_label.pack(pady=10)

        # Right Frame - Controls
        self.right_frame = ctk.CTkFrame(self.root, width=300, height=600)
        self.right_frame.grid(row=0, column=1, padx=10, pady=10)

        # Load images for buttons
        self.start_img = CTkImage(light_image=Image.open("assets/open-camera.png"), dark_image=Image.open("assets/open-camera.png"), size=(60, 60))
        self.stop_img = CTkImage(light_image=Image.open("assets/no-camera.png"), dark_image=Image.open("assets/no-camera.png"), size=(60, 60))
        self.video_img = CTkImage(light_image=Image.open("assets/video-recoder.png"), dark_image=Image.open("assets/video-recoder.png"), size=(60, 60))
        self.image_img = CTkImage(light_image=Image.open("assets/image-download.png"), dark_image=Image.open("assets/image-download.png"), size=(60, 60))
        self.export_img = CTkImage(light_image=Image.open("assets/download.png"), dark_image=Image.open("assets/download.png"), size=(60, 60))
        self.theme_img = CTkImage(light_image=Image.open("assets/settings.png"), dark_image=Image.open("assets/settings.png"), size=(60, 60))

        # Buttons in 3x2 grid
        self.start_button = ctk.CTkButton(self.right_frame, image=self.start_img, text="", command=self.start_detection)
        self.start_button.grid(row=0, column=0, padx=10, pady=10)

        self.stop_button = ctk.CTkButton(self.right_frame, image=self.stop_img, text="", command=self.stop_detection, state="disabled")
        self.stop_button.grid(row=1, column=0, padx=10, pady=10)

        self.detect_video_button = ctk.CTkButton(self.right_frame, image=self.video_img, text="", command=self.detect_from_video)
        self.detect_video_button.grid(row=2, column=0, padx=10, pady=10)

        self.detect_image_button = ctk.CTkButton(self.right_frame, image=self.image_img, text="", command=self.upload_image)
        self.detect_image_button.grid(row=3, column=0, padx=10, pady=10)

        self.export_button = ctk.CTkButton(self.right_frame, image=self.export_img, text="", command=self.export_logs)
        self.export_button.grid(row=4, column=0, padx=10, pady=10)

        self.theme_button = ctk.CTkButton(self.right_frame, image=self.theme_img, text="", command=self.toggle_theme)
        self.theme_button.grid(row=5, column=0, padx=10, pady=10)

        # Confidence Threshold Slider
        self.confidence_label = ctk.CTkLabel(self.right_frame, text="Confidence Threshold:")
        self.confidence_label.grid(row=6, column=0, padx=10, pady=10)

        self.confidence_slider = ctk.CTkSlider(self.right_frame, from_=0, to=1, number_of_steps=10, command=self.update_confidence_threshold)
        self.confidence_slider.set(self.confidence_threshold)  # Set the default value to 0.5
        self.confidence_slider.grid(row=7, column=0, padx=5, pady=5)

    def update_confidence_threshold(self, value):
        self.confidence_threshold = float(value)

    def download_model(self):
        model_dir = "models"
        os.makedirs(model_dir, exist_ok=True)

        self.weights_path = os.path.join(model_dir, "yolov4.weights")
        self.cfg_path = os.path.join(model_dir, "yolov4.cfg")

        if os.path.exists(self.weights_path) and os.path.exists(self.cfg_path):
            print("YOLO model files loaded.")
        else:
            print("Error: Model files not found.")
            raise FileNotFoundError("Required YOLO model files are missing.")

    def start_detection(self):
        self.detection_running = True
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.video_capture = cv2.VideoCapture(0)

        self.detection_thread = threading.Thread(target=self.run_detection)
        self.detection_thread.start()

    def stop_detection(self):
        self.detection_running = False
        self.start_button.configure(state="normal")
        self.stop_button.configure(state="disabled")

        if self.video_capture:
            self.video_capture.release()

    def detect_from_video(self):
        file_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi")])
        if not file_path:
            return

        self.video_capture = cv2.VideoCapture(file_path)
        self.detection_running = True
        self.run_detection()

    def upload_image(self):
        # Open a file dialog to choose an image
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.png;*.jpeg")])

        if file_path:
            image = cv2.imread(file_path)  # Read the image
            self.detect_objects(image)

    def detect_objects(self, image):
        # Your detection logic here
        # For example, using a YOLO model:

        net = cv2.dnn.readNet(self.weights_path, self.cfg_path)
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
        classes = ["traffic_sign", "pedestrian", "car", "truck", "cyclist", "person"]  # Replace with your classes

        height, width, _ = image.shape
        blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outputs = net.forward(output_layers)

        boxes = []
        confidences = []
        class_ids = []

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = int(scores.argmax())
                confidence = scores[class_id]
                if confidence > self.confidence_threshold:
                    center_x, center_y, w, h = (detection[:4] * [width, height, width, height]).astype(int)
                    x, y = int(center_x - w / 2), int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Apply non-maxima suppression
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, 0.4)

        if len(indexes) > 0:
            for i in indexes[0].flatten():  # Fix: Access the first element of the tuple
                x, y, w, h = boxes[i]
                label = classes[class_ids[i]]
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Convert to PIL Image and display in Tkinter
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(image)
        img_tk = ctk.CTkImage(light_image=img, dark_image=img, size=(700, 500))
        self.video_label.configure(image=img_tk, text="")
        self.video_label.image = img_tk

    def run_detection(self):
        net = cv2.dnn.readNet(self.weights_path, self.cfg_path)
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
        classes = ["traffic_sign", "pedestrian", "car", "truck", "cyclist", "person"]
        trackers = []  # Initialize the trackers list
        self.start_time = time.time()
        while self.detection_running:
            ret, frame = self.video_capture.read()
            if not ret:
                break

            self.frame_count += 1
            self.process_frame(frame, net, output_layers, classes)

            # Calculate FPS every 30 frames
            if self.frame_count % 30 == 0:
                elapsed_time = time.time() - self.start_time
                fps = self.frame_count / elapsed_time
                self.root.title(f"FPS: {fps:.2f} - Advanced Object Detection")

    def process_frame(self, frame, net=None, output_layers=None, classes=None, trackers=None):
        height, width, _ = frame.shape
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outputs = net.forward(output_layers)

        boxes = []  # List to store the bounding boxes of detected objects
        confidences = []  # List to store the confidence of the detections
        class_ids = []  # List to store the class IDs of the detections

        # Iterate over the detections
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = int(scores.argmax())
                confidence = scores[class_id]
                if confidence > self.confidence_threshold:  # Use the current confidence threshold
                    center_x, center_y, w, h = (detection[:4] * [width, height, width, height]).astype(int)
                    x, y = int(center_x - w / 2), int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Apply non-maxima suppression to reduce duplicate boxes
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, 0.4)

        # Track the objects
        if len(boxes) > 0:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                label = classes[class_ids[i]]

                # Update the DeepSORT tracker
                tracker = self.deep_sort_tracker.update(frame, boxes)
                trackers.append(tracker)

                # Draw the bounding boxes and labels
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        img_tk = CTkImage(light_image=img, dark_image=img, size=(700, 500))
        self.video_label.configure(image=img_tk, text="")
        self.video_label.image = img_tk

    def export_logs(self):
        if not self.detection_log:
            return

        file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if file_path:
            df = pd.DataFrame(self.detection_log)
            df.to_csv(file_path, index=False)

    def toggle_theme(self):
        current_mode = ctk.get_appearance_mode()
        ctk.set_appearance_mode("Light" if current_mode == "Dark" else "Dark")


if __name__ == "__main__":
    root = ctk.CTk()
    app = ObjectDetectionApp(root)
    root.resizable(False, False)
    root.mainloop()
