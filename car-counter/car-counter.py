from ultralytics import YOLO
import cv2
import cvzone
import math
from utils.sort import *

# Load the video
cap = cv2.VideoCapture("./assets/video/traffic_flow.mp4")

# Load the model
model = YOLO("./yolo/yolov8l.pt")

# Class names to detect
classNames = ["car", "motorbike", "bus", "truck"]

# Load the mask
mask = cv2.imread("assets/image/mask.png")

# Tracking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# Tracking line coordinates [x1, y1, x2, y2]
limits = [351, 400, 926, 400]

# Total count of vehicles
totalCount = []

# # Load the counter UI
# counterUI = cv2.imread("assets/image/graphics.png", cv2.IMREAD_UNCHANGED)
#
# # Resize the counter UI
# counterUI = cv2.resize(counterUI, (int(counterUI.shape[1] // 1.5), int(counterUI.shape[0] // 1.5)))

while True:
    # Read the frame
    success, img = cap.read()
    # Apply the mask
    imgRegion = cv2.bitwise_and(img, mask)

    # img = cvzone.overlayPNG(img, counterUI, (0, 0))

    # Get the results from the model
    results = model(imgRegion, stream=True)

    # Array to store the detections
    detections = np.empty((0, 5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Width and Height of the bounding box
            w, h = x2 - x1, y2 - y1

            # Confidence of the detection
            conf = math.ceil((box.conf[0] * 100)) / 100

            # Get the class name
            cls = int(box.cls[0])

            # Prevent the index out of range error
            if (cls >= len(classNames)):
                continue

            currentClass = classNames[cls]

            # Check if the class is a vehicle and the confidence is greater than 0.3
            if classNames.count(currentClass) != 0 and conf > 0.3:
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    # Update the tracker
    resultsTracker = tracker.update(detections)

    # Draw the tracking line
    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (90, 40, 40), 5)

    for result in resultsTracker:
        # Get the coordinates of the bounding box
        x1, y1, x2, y2, id = result
        # Convert the coordinates to integers
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        # Width and Height of the bounding box
        w, h = x2 - x1, y2 - y1

        print(result)

        # Draw the bounding box around the vehicle
        cvzone.cornerRect(img, (x1, y1, w, h), l=0, t=1, rt=2, colorR=(0, 255, 0), colorC=(0, 255, 0))

        # Draw the ID of the vehicle
        cvzone.putTextRect(img, f'{int(id)}', (max(0, x1), max(35, y1)),
                           scale=2, thickness=2, offset=8, colorR=(40, 110, 40))

        # Draw the center of the vehicle
        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(img, (cx, cy), 3, (0, 255, 0), cv2.FILLED)

        # Check if the center of the vehicle is within the tracking line
        if limits[0] < cx < limits[2] and limits[1] - 15 < cy < limits[1] + 15:
            if totalCount.count(id) == 0:
                totalCount.append(id)
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (40, 140, 40), 5)

    # Draw the total count of vehicles
    # cv2.putText(img, str(len(totalCount)), (162, 70), cv2.FONT_HERSHEY_PLAIN, 4, (255, 70, 50), 5)
    cvzone.putTextRect(img, f' Count: {len(totalCount)} ', thickness=2, pos=(50, 75), colorR=(90, 40, 40))

    cv2.imshow("Image", img)
    cv2.imshow("ImageRegion", imgRegion) # Uncomment this line to see the masked region
    cv2.waitKey(1)