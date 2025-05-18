import numpy as np
import torch
from PIL import ImageDraw
import os
from ultralytics import YOLO
# COCO dataset classes
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Define colors for different classes
COLORS = {}
for i, cls in enumerate(COCO_CLASSES):
    COLORS[cls] = [np.random.randint(0, 255) for _ in range(3)]

# Model directory
MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)

# Global variable to store the model
detection_model = None
# Load pre-trained model - YOLO 
def load_model():
    try:    
        # Load yolov11 model from ultralytics
        detection_model = YOLO('yolo11n.pt')
        # Use gpu if available
        detection_model.to('cuda' if torch.cuda.is_available() else 'cpu')
        
        print("YOLOv11 model loaded successfully")
        return detection_model
    
    except Exception as e:
        print(f"Error loading YOLOv11 model: {e}")
        print("Please ensure you have an internet connection for the first run")
        raise

def get_prediction(image):
    # Get predictions from the YOLO model for the given image"
    global detection_model
    
    # Load model if not already loaded
    if detection_model is None:
        detection_model = load_model()
        # Set confidence threshold
        # Change confidence threshold (higher = fewer detections but more confident)
        detection_model.conf = 0.45  
        # Change IoU threshold for Non-Maximum Suppression
        detection_model.iou = 0.50   
        # Change input image size (larger = more accurate but slower)
        detection_model.imgsz = 1280
    # Ensure model is in evaluation mode
    detection_model.eval()
    
    # Run inference
    # Process results - YOLO returns list of Results objects
    results = detection_model(image)
    
    # Extract boxes, labels, and scores from the results
    boxes = []
    scores = []
    labels = []
    
    # Process detections - results[0] contains predictions for the first image
    if len(results) > 0:
        # Convert to numpy for consistency
        result_array = results[0].boxes.data.cpu().numpy()
        
        # Extract boxes (first 4 columns are x1, y1, x2, y2)
        boxes = result_array[:, :4]
        
        # Extract confidence scores (5th column)
        scores = result_array[:, 4]
        
        # Extract class labels (6th column)
        labels = result_array[:, 5].astype(int)
    
    return np.array(boxes), np.array(labels), np.array(scores)

def draw_boxes(image, boxes, labels, scores):
    # Draw bounding boxes and labels on the image
    draw_image = image.copy()
    draw = ImageDraw.Draw(draw_image)
    
    # Draw each bounding box
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box.astype(int)
        
        # Get class name and color
        class_idx = labels[i]
        class_name = COCO_CLASSES[class_idx] if class_idx < len(COCO_CLASSES) else f"Class {class_idx}"
        color = tuple(COLORS[class_name] if class_name in COLORS else [0, 250, 0])
        
        # Define your custom font and size

        # Draw rectangle
        draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=2)
        
        # Display class name and confidence score
        score = scores[i]
        # Create the text string
        text = f"{class_name}: {score:.2f}"

        # Get text dimensions (approximate)
        text_bbox = draw.textbbox((0, 0), text)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        # Position the text inside the bounding box (at the top)
        text_x = x1 + 2  # 2 pixels margin from left edge
        text_y = y1 + 2  # 2 pixels margin from top edge

        # Draw text background (optional)
        draw.rectangle([(text_x, text_y), (text_x + text_width, text_y + text_height +2)], fill = color)

        # Draw the text
        draw.text((text_x, text_y), text, fill=(255, 255, 255)) # White text
    
    return draw_image 