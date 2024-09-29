# Import All the Required Libraries
import json
import cv2
from ultralytics import YOLO
import numpy as np
import re
import os
from datetime import datetime
from paddleocr import PaddleOCR

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# Create a Video Capture Object
cap = cv2.VideoCapture(r"resources\traffic_car2.mp4")
# Initialize the YOLOv10 Model
model = YOLO("weights\\best.pt")
# Initialize the Paddle OCR
ocr = PaddleOCR(use_angle_cls=True, use_gpu=False)

# Set confidence threshold and persistence duration
CONFIDENCE_THRESHOLD = 0.8
PERSISTENCE_FRAMES = 5

# Dictionary to track detected license plates and their persistence
tracked_plates = {}
plate_history = {}  # History of detected plates
unique_plates_data = []  # To store unique plates with timestamps
seen_plates = set()  # To track unique plates during the session

# Define the Area of Interest (example: rectangle coordinates)
area_of_interest = (95, 555, 916, 700)  # Example coordinates, modify as needed

def draw_area_of_interest(frame):
    x1, y1, x2, y2 = area_of_interest
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green rectangle

def paddle_ocr(frame, x1, y1, x2, y2):
    frame = frame[y1:y2, x1:x2]
    result = ocr.ocr(frame, det=False, rec=True, cls=False)
    text = ""
    for r in result:
        scores = r[0][1]
        if np.isnan(scores):
            scores = 0
        else:
            scores = int(scores * 100)
        if scores > 60:
            text = r[0][0]
    pattern = re.compile('[\W]')
    text = pattern.sub('', text)
    text = text.replace("???", "").replace("O", "0").replace("粤", "")
    return str(text) if is_valid_plate(text) else ""

def is_valid_plate(plate):
    # Adjust this function to match your expected license plate format
    return bool(re.match(r'^[A-Z0-9]+$', plate)) and 6 <= len(plate) <= 10

def save_to_jsonl(data):
    with open("output.jsonl", 'a') as f:
        for entry in data:
            f.write(json.dumps(entry) + '\n')

def compute_iou(boxA, boxB):
    xA, yA, xB, yB = boxA
    xC, yC, xD, yD = boxB
    interX1 = max(xA, xC)
    interY1 = max(yA, yC)
    interX2 = min(xB, xD)
    interY2 = min(yB, yD)
    interArea = max(0, interX2 - interX1) * max(0, interY2 - interY1)
    boxAArea = (xB - xA) * (yB - yA)
    boxBArea = (xD - xC) * (yD - yC)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

# Directory for saving images
os.makedirs("saved_images", exist_ok=True)

startTime = datetime.now()

while True:
    ret, frame = cap.read()
    if ret:
        currentTime = datetime.now()
        draw_area_of_interest(frame)

        results = model.predict(frame, conf=0.45)
        current_detected_plates = set()

        for result in results:
            boxes = result.boxes
            for box in boxes:
                conf = box.conf[0].item()
                if conf >= CONFIDENCE_THRESHOLD:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

                    # Check if the detected box is within the area of interest
                    if (x1 < area_of_interest[2] and x2 > area_of_interest[0] and
                        y1 < area_of_interest[3] and y2 > area_of_interest[1]):
                        label = paddle_ocr(frame, x1, y1, x2, y2)

                        if label:
                            current_detected_plates.add(label)

                            # Save unique plates with timestamp if not already seen
                            if label not in seen_plates:
                                seen_plates.add(label)

                                # Save images of the license plate and the car
                                license_plate_img_path = f"saved_images/{label}_license_plate.jpg"
                                car_img_path = f"saved_images/{label}_car.jpg"

                                # Save the license plate image
                                cv2.imwrite(license_plate_img_path, frame[y1:y2, x1:x2])
                                # Save the full car image (you might want to adjust this if you want a specific area)
                                cv2.imwrite(car_img_path, frame)

                                unique_plates_data.append({
                                    "timestamp": currentTime.isoformat(),
                                    "license_plate": label,
                                    "license_plate_img": license_plate_img_path,
                                    "car_img": car_img_path
                                })

                            # Update plate history
                            if label not in plate_history:
                                plate_history[label] = []
                            plate_history[label].append((x1, y1, x2, y2))
                            if len(plate_history[label]) > PERSISTENCE_FRAMES:
                                plate_history[label].pop(0)

                            # Track plates
                            if label not in tracked_plates:
                                tracked_plates[label] = {'persistence': PERSISTENCE_FRAMES}
                            else:
                                tracked_plates[label]['last_box'] = (x1, y1, x2, y2)

                            # Draw the bounding box and label
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                            textSize = cv2.getTextSize(label, 0, fontScale=0.5, thickness=2)[0]
                            c2 = x1 + textSize[0], y1 - textSize[1] - 3
                            cv2.rectangle(frame, (x1, y1), c2, (255, 0, 0), -1)
                            cv2.putText(frame, label, (x1, y1 - 2), 0, 0.5, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)

        # Update persistence for tracked plates
        for plate in list(tracked_plates.keys()):
            if plate not in current_detected_plates:
                tracked_plates[plate]['persistence'] -= 1
                if tracked_plates[plate]['persistence'] <= 0:
                    del tracked_plates[plate]

        # Print tracked plates for debugging
        print("Tracked Plates:", tracked_plates)

        # Save unique plates data to JSONL every 20 seconds
        if (currentTime - startTime).seconds >= 20:
            save_to_jsonl(unique_plates_data)
            startTime = currentTime
            unique_plates_data.clear()  # Clear the list after saving

        cv2.imshow("Video", frame)
        if cv2.waitKey(1) & 0xFF == ord('1'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
