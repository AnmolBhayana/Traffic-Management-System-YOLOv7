"""
app.py
=======
Flask web application for the Traffic Management System.
Serves the dashboard and streams live video feed.
"""

from flask import Flask, render_template, Response, jsonify
import cv2
import threading

from detector import VehicleDetector
from pipeline import TrafficPipeline

app = Flask(__name__)

# ─────────────────────────────────────────────
# Global state
# ─────────────────────────────────────────────
detector = VehicleDetector(weights_path='yolov7.weights')
pipeline = TrafficPipeline(history_window=60)

# Video source: 0 = webcam, or path to video file
VIDEO_SOURCE = 0
cap = None
frame_lock = threading.Lock()
latest_frame = None
running = False


def capture_loop():
    """Background thread: captures frames, runs detection, updates pipeline."""
    global latest_frame, running, cap
    cap = cv2.VideoCapture(VIDEO_SOURCE)

    if not cap.isOpened():
        print(f"⚠️  Could not open video source: {VIDEO_SOURCE}")
        print("    Generating demo frames instead.")
        import numpy as np
        running = True
        while running:
            # Create a blank demo frame
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "DEMO MODE — No Camera", (120, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)
            detections = detector.detect(frame)
            pipeline.process(detections)
            annotated = detector.draw_detections(frame.copy(), detections)
            _, buffer = cv2.imencode('.jpg', annotated)
            with frame_lock:
                latest_frame = buffer.tobytes()
        return

    running = True
    while running:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        detections = detector.detect(frame)
        pipeline.process(detections)
        annotated = detector.draw_detections(frame.copy(), detections)

        _, buffer = cv2.imencode('.jpg', annotated)
        with frame_lock:
            latest_frame = buffer.tobytes()

    cap.release()


def generate_frames():
    """Generator for MJPEG stream."""
    while True:
        with frame_lock:
            if latest_frame is None:
                continue
            frame = latest_frame
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/metrics')
def metrics():
    """JSON endpoint — dashboard polls this every second."""
    return jsonify(pipeline.get_dashboard_data())

@app.route('/api/reset', methods=['POST'])
def reset():
    pipeline.reset()
    return jsonify({'status': 'reset'})


# ─────────────────────────────────────────────
# Start
# ─────────────────────────────────────────────

if __name__ == '__main__':
    thread = threading.Thread(target=capture_loop, daemon=True)
    thread.start()
    print("🚦 Traffic Management System running at http://localhost:5000")
    app.run(debug=False, host='0.0.0.0', port=5000)
