from ultralytics import YOLO
import random
import cv2
import numpy as np

# load a pretrained YOLOv2 model
model = YOLO("yolov2.pt", "v2")

# predict on an image
detection_output = model.predict(source=r"C:\Users\galir\Downloads\24th-  ultralytics- yolo\24th-  ultralytics- yolo\8. YOLO\img\1.jpg", conf=0.25, save=True)

# Display tensor array
print(detection_output)

# Display numpy array
print(detection_output[0].numpy())



# opening the file in read mode
my_file = open(r"C:\Users\galir\Downloads\24th-  ultralytics- yolo\24th-  ultralytics- yolo\8. YOLO\utils\coco.txt", "r")
# reading the file
data = my_file.read()
# replacing end splitting the text | when newline ('\n') is seen.
class_list = data.split("\n")
my_file.close()

# print(class_list)

# Generate random colors for class list
detection_colors = []
for i in range(len(class_list)):
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    detection_colors.append((b, g, r))

# load a pretrained YOLOv2 model
#model = YOLO("weights/yolov2.pt", "v2")

# Vals to resize video frames | small frame optimise the run
frame_wid = 640
frame_hyt = 480

# cap = cv2.VideoCapture(1)
cap = cv2.VideoCapture(r"C:\Users\galir\Downloads\24th-  ultralytics- yolo\24th-  ultralytics- yolo\8. YOLO\video\video_sample1.mp4")

if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    #  resize the frame | small frame optimise the run
    # frame = cv2.resize(frame, (frame_wid, frame_hyt))

    # Predict on image
    detect_params = model.predict(source=[frame], conf=0.45, save=True)

    # Convert tensor array to numpy
    DP = detect_params[0].numpy()
    print(DP)

    if len(DP) != 0:
        for i in range(len(detect_params[0])):
            print(i)

            boxes = detect_params[0].boxes
            box = boxes[i]  # returns one box
            clsID = box.cls.numpy()[0]
            conf = box.conf.numpy()[0]
            bb = box.xyxy.numpy()[0]

            cv2.rectangle(
                frame,
                (int(bb[0]), int(bb[1])),
                (int(bb[2]), int(bb[3])),
                detection_colors[int(clsID)],
                3,
            )

            # Display class name and confidence
            font = cv2.FONT_HERSHEY_COMPLEX
            cv2.putText(
                frame,
                class_list[int(clsID)] + " " + str(round(conf, 3)) + "%",
                (int(bb[0]), int(bb[1]) - 10),
                font,
                1,
                (255, 255, 255),
                2,
            )

    # Display the resulting frame
    cv2.imshow("ObjectDetection", frame)

    # Terminate run when "Q" pressed
    if cv2.waitKey(1) == ord("q"):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()