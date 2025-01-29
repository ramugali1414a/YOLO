import streamlit as st
from ultralytics import YOLO
import random
import cv2
import numpy as np
import tempfile

# Load YOLO model
model = YOLO("weights/yolov8n.pt", "v8")

# Generate random colors for the class list
with open(r"C:\Users\galir\Downloads\24th-  ultralytics- yolo\24th-  ultralytics- yolo\8. YOLO\utils\coco.txt", "r") as file:
    class_list = file.read().split("\n")

detection_colors = []
for _ in range(len(class_list)):
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    detection_colors.append((b, g, r))

# Streamlit App
st.title("YOLO Object Detection")
st.markdown("Upload an image or video to detect objects using YOLOv8 model.")

# Upload image or video
uploaded_file = st.file_uploader("Upload Image or Video", type=["jpg", "jpeg", "png", "mp4", "avi"])

if uploaded_file is not None:
    # Create temporary file for the uploaded content
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    # Check if it's an image or video
    if uploaded_file.type.startswith("image"):
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        image = cv2.imread(tfile.name)
        detect_params = model.predict(source=[image], conf=0.45)
        
        # Process detections
        detect_results = detect_params[0].numpy()
        if len(detect_results) != 0:
            for i in range(len(detect_params[0])):
                boxes = detect_params[0].boxes
                box = boxes[i]
                clsID = box.cls.numpy()[0]
                conf = box.conf.numpy()[0]
                bb = box.xyxy.numpy()[0]
                
                # Draw bounding box and label
                cv2.rectangle(image, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), detection_colors[int(clsID)], 3)
                font = cv2.FONT_HERSHEY_SIMPLEX
                label = f"{class_list[int(clsID)]} {round(conf, 3) * 100}%"
                cv2.putText(image, label, (int(bb[0]), int(bb[1]) - 10), font, 1, (255, 255, 255), 2)
            
            # Display the result
            st.image(image, caption="Detection Result", use_column_width=True)
    
    elif uploaded_file.type.startswith("video"):
        st.video(uploaded_file)
        
        # Load video and display detections
        cap = cv2.VideoCapture(tfile.name)
        
        # Create a temporary file for saving the processed frames
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        out = cv2.VideoWriter(temp_file.name, cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (640, 480))
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Perform detection on each frame
            detect_params = model.predict(source=[frame], conf=0.45)
            detect_results = detect_params[0].numpy()

            if len(detect_results) != 0:
                for i in range(len(detect_params[0])):
                    boxes = detect_params[0].boxes
                    box = boxes[i]
                    clsID = box.cls.numpy()[0]
                    conf = box.conf.numpy()[0]
                    bb = box.xyxy.numpy()[0]
                    
                    # Draw bounding box and label
                    cv2.rectangle(frame, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), detection_colors[int(clsID)], 3)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    label = f"{class_list[int(clsID)]} {round(conf, 3) * 100}%"
                    cv2.putText(frame, label, (int(bb[0]), int(bb[1]) - 10), font, 1, (255, 255, 255), 2)
                
            # Write the processed frame to the output video file
            out.write(frame)
        
        cap.release()
        out.release()

        # Display processed video
        st.video(temp_file.name)

# Error handling for missing input
else:
    st.warning("Please upload an image or video file.")
