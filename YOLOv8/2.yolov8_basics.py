from ultralytics import YOLO
import numpy

# load a pretrained YOLOv8n model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='path/to/yolov8n.pt')


# predict on an image
detection_output = model.predict(source=r"C:\Users\galir\Downloads\24th-  ultralytics- yolo\24th-  ultralytics- yolo\8. YOLO\img\1.jpg", conf=0.25, save=True)

# Display tensor array
print(detection_output)

# Display numpy array
print(detection_output[0].numpy())