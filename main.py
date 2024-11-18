import torch
import cv2
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

def detect_waste(image_path, confidence_threshold=0.2):
    model.conf = confidence_threshold

    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape

    quadrants = {
        "Top-Left": [0, 0, w // 2, h // 2],
        "Top-Right": [w // 2, 0, w, h // 2],
        "Bottom-Left": [0, h // 2, w // 2, h],
        "Bottom-Right": [w // 2, h // 2, w, h]
    }

    results = model(img_rgb)
    detections = results.pandas().xyxy[0]

    garbage_labels = [
        "bottle", "can", "plastic bag", "cup", "fork", "knife", "spoon",
        "bowl", "box", "bucket", "laptop", "umbrella", "chair",
        "cell phone", "backpack", "toilet", "book", "banana", "apple",
        "orange", "broccoli", "carrot", "plastic", "dustbin", "metal cans",
        "paper", "cardboard"
    ]

    quadrant_counts = {key: 0 for key in quadrants}
    total_items = 0

    for _, row in detections.iterrows():
        xmin, ymin, xmax, ymax, label, confidence = row[['xmin', 'ymin', 'xmax', 'ymax', 'name', 'confidence']]

        if label in garbage_labels:
            total_items += 1
            color = (0, 255, 0)
            cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 3)
            cv2.putText(img, f'{label}: {confidence:.2f}', (int(xmin), int(ymin) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            for quadrant, coords in quadrants.items():
                x1, y1, x2, y2 = coords
                if xmin >= x1 and ymin >= y1 and xmax <= x2 and ymax <= y2:
                    quadrant_counts[quadrant] += 1
                    break

    font = cv2.FONT_HERSHEY_SIMPLEX
    for quadrant, count in quadrant_counts.items():
        if total_items > 0 and (count / total_items) >= 0.1:
            x1, y1, x2, y2 = quadrants[quadrant]
            color = (0, 0, 255)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)

            percentage = (count / total_items) * 100
            cv2.putText(img, f'{percentage:.2f}%', (x1 + 10, y1 + 30), font, 1, (255, 0, 0), 2)

            cv2.putText(img, f'Garbage Pile Detected', (x1 + 10, y1 + 60), font, 1, (255, 0, 0), 2)

            for _, row in detections.iterrows():
                xmin, ymin, xmax, ymax, label, confidence = row[['xmin', 'ymin', 'xmax', 'ymax', 'name', 'confidence']]
                if label in garbage_labels:
                    if xmin >= x1 and ymin >= y1 and xmax <= x2 and ymax <= y2:
                        cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 3)
                        cv2.putText(img, f'{label}: {confidence:.2f}', (int(xmin), int(ymin) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    max_width, max_height = 800, 600
    if w > max_width or h > max_height:
        scale = min(max_width / w, max_height / h)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))

    cv2.namedWindow('Waste Detection', cv2.WINDOW_NORMAL)
    cv2.imshow('Waste Detection', img)
    print(f"Detected items: {total_items}, Quadrants with at least 10% items: {quadrant_counts}")
    print("Press any key to close the window...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

l = [
    "C:\\Users\\Raj Chaurasiya\\Documents\\Coding\\projects\\waste_detection\\samples\\a.jpg",
    "C:\\Users\\Raj Chaurasiya\\Documents\\Coding\\projects\\waste_detection\\samples\\b.jpeg",
    "C:\\Users\\Raj Chaurasiya\\Documents\\Coding\\projects\\waste_detection\\samples\\c.jpg",
    "C:\\Users\\Raj Chaurasiya\\Documents\\Coding\\projects\\waste_detection\\samples\\d.jpg",
    "C:\\Users\\Raj Chaurasiya\\Documents\\Coding\\projects\\waste_detection\\samples\\e.jpeg",
    "C:\\Users\\Raj Chaurasiya\\Documents\\Coding\\projects\\waste_detection\\samples\\f.jpg",
    "C:\\Users\\Raj Chaurasiya\\Documents\\Coding\\projects\\waste_detection\\samples\\g.jpeg",
    "C:\\Users\\Raj Chaurasiya\\Documents\\Coding\\projects\\waste_detection\\samples\\h.webp",
    "C:\\Users\\Raj Chaurasiya\\Documents\\Coding\\projects\\waste_detection\\samples\\i.jpg",
    "C:\\Users\\Raj Chaurasiya\\Documents\\Coding\\projects\\waste_detection\\samples\\j.jpg",
    "C:\\Users\\Raj Chaurasiya\\Documents\\Coding\\projects\\waste_detection\\samples\\k.jpg",
    "C:\\Users\\Raj Chaurasiya\\Documents\\Coding\\projects\\waste_detection\\samples\\l.webp",
    "C:\\Users\\Raj Chaurasiya\\Documents\\Coding\\projects\\waste_detection\\samples\\m.jpg",
    "C:\\Users\\Raj Chaurasiya\\Documents\\Coding\\projects\\waste_detection\\samples\\n.jpg",
    "C:\\Users\\Raj Chaurasiya\\Documents\\Coding\\projects\\waste_detection\\samples\\o.jpg",]

for i in l:
    detect_waste(i)
