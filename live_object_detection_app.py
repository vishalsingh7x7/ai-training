import streamlit as st
import torch
import torchvision
from torchvision.transforms import functional as F
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Live Object Detection", layout="wide")
st.title("🎥 Live Object Detection")

# Load the model
@st.cache_resource
def load_model():
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    return model

model = load_model()

# COCO labels
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
    'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
    'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
    'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Detection function
def detect_objects(img, threshold=0.5):
    img_tensor = F.to_tensor(img)
    with torch.no_grad():
        predictions = model([img_tensor])[0]
    
    boxes = predictions['boxes']
    labels = predictions['labels']
    scores = predictions['scores']

    # Filter by score threshold
    keep = scores > threshold
    boxes = boxes[keep]
    labels = labels[keep]
    scores = scores[keep]

    return boxes, labels, scores

# Draw boxes
def draw_boxes(img, boxes, labels, scores):
    img = np.array(img.copy())

    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = [int(v.item()) for v in box]
        label_name = COCO_INSTANCE_CATEGORY_NAMES[int(label)]

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img,
            f"{label_name} {score:.2f}",
            (x1, max(y1 - 5, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            2,
        )

    return img



# Stream webcam
FRAME_WINDOW = st.image([])
camera = cv2.VideoCapture(0)

run = st.checkbox("▶️ Start Camera")

if run:
    while True:
        success, frame = camera.read()
        if not success:
            st.error("Failed to read from webcam")
            break

        # Convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)

        # Detect
        boxes, labels, scores = detect_objects(pil_image)

        # Draw
        result_img = draw_boxes(pil_image, boxes, labels, scores)

        FRAME_WINDOW.image(result_img, channels="RGB")

else:
    st.write("✅ Camera paused.")
    camera.release()
