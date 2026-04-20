"""
detector.py
============
Vehicle detection and tracking using YOLOv7.
Handles frame-by-frame detection, classification, and counting.
"""

import cv2
import numpy as np
import time


# Vehicle classes from COCO dataset that YOLOv7 is trained on
VEHICLE_CLASSES = {
    2: 'car',
    3: 'motorcycle',
    5: 'bus',
    7: 'truck'
}

COLORS = {
    'car':        (0, 255, 0),
    'motorcycle': (255, 165, 0),
    'bus':        (0, 0, 255),
    'truck':      (255, 0, 255),
}


class VehicleDetector:
    """
    Detects and classifies vehicles in video frames using YOLOv7.

    Parameters
    ----------
    weights_path : str
        Path to YOLOv7 weights file (.pt converted to ONNX or use torch hub)
    conf_threshold : float
        Minimum confidence to count a detection
    nms_threshold : float
        Non-maximum suppression threshold
    input_size : tuple
        Frame resize dimensions for the model
    """

    def __init__(self, weights_path='yolov7.weights',
                 conf_threshold=0.5, nms_threshold=0.4,
                 input_size=(640, 640)):
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.input_size = input_size
        self.model = self._load_model(weights_path)

    def _load_model(self, weights_path):
        """Load YOLOv7 model via OpenCV DNN or torch hub."""
        try:
            # Option 1: OpenCV DNN (ONNX export of YOLOv7)
            net = cv2.dnn.readNet(weights_path)
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            print("✅ YOLOv7 model loaded via OpenCV DNN.")
            return net
        except Exception as e:
            print(f"⚠️  Could not load weights from {weights_path}: {e}")
            print("    Running in DEMO MODE with simulated detections.")
            return None

    def detect(self, frame):
        """
        Run detection on a single frame.

        Returns
        -------
        detections : list of dict
            Each dict has keys: label, confidence, bbox (x, y, w, h)
        """
        if self.model is None:
            return self._demo_detections(frame)

        blob = cv2.dnn.blobFromImage(
            frame, 1/255.0, self.input_size,
            swapRB=True, crop=False
        )
        self.model.setInput(blob)
        outputs = self.model.forward(self._get_output_layers())
        return self._parse_outputs(outputs, frame.shape)

    def _get_output_layers(self):
        layer_names = self.model.getLayerNames()
        return [layer_names[i - 1]
                for i in self.model.getUnconnectedOutLayers().flatten()]

    def _parse_outputs(self, outputs, frame_shape):
        h, w = frame_shape[:2]
        boxes, confidences, class_ids = [], [], []

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = int(np.argmax(scores))
                confidence = float(scores[class_id])

                if class_id not in VEHICLE_CLASSES:
                    continue
                if confidence < self.conf_threshold:
                    continue

                cx, cy, bw, bh = (detection[:4] * np.array([w, h, w, h])).astype(int)
                x = int(cx - bw / 2)
                y = int(cy - bh / 2)
                boxes.append([x, y, bw, bh])
                confidences.append(confidence)
                class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(
            boxes, confidences, self.conf_threshold, self.nms_threshold)

        detections = []
        if len(indices) > 0:
            for i in indices.flatten():
                detections.append({
                    'label': VEHICLE_CLASSES[class_ids[i]],
                    'confidence': confidences[i],
                    'bbox': boxes[i]
                })
        return detections

    def _demo_detections(self, frame):
        """Simulated detections for demo/testing without weights."""
        np.random.seed(int(time.time()) % 100)
        n = np.random.randint(2, 8)
        h, w = frame.shape[:2]
        detections = []
        for _ in range(n):
            label = np.random.choice(list(VEHICLE_CLASSES.values()))
            x = np.random.randint(0, w - 100)
            y = np.random.randint(0, h - 60)
            detections.append({
                'label': label,
                'confidence': round(np.random.uniform(0.6, 0.99), 2),
                'bbox': [x, y, np.random.randint(60, 120), np.random.randint(40, 80)]
            })
        return detections

    def draw_detections(self, frame, detections):
        """Draw bounding boxes and labels on the frame."""
        for det in detections:
            x, y, w, h = det['bbox']
            label = det['label']
            conf = det['confidence']
            color = COLORS.get(label, (200, 200, 200))

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = f"{label} {conf:.0%}"
            cv2.putText(frame, text, (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        return frame
