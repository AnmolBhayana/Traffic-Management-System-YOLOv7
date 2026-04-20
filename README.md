# 🚦 Real-Time Traffic Management System using YOLOv7

A real-time highway traffic monitoring system built during my internship at **Innovate Technologies**. The system uses YOLOv7 to detect, classify, and track vehicles from live video feeds, and displays live traffic insights through a web-based dashboard.

---

## 📌 Overview

Traffic monitoring is a critical part of smart city infrastructure. This project automates vehicle detection and counting from live camera feeds using a state-of-the-art object detection model, eliminating the need for manual monitoring.

---

## 🎯 Features

- ✅ Real-time vehicle detection and classification from live video feeds
- ✅ Tracks multiple vehicle types (cars, trucks, motorcycles, buses)
- ✅ Python backend pipeline for real-time data processing
- ✅ Live traffic density calculation per lane
- ✅ Web-based dashboard displaying vehicle counts and traffic status
- ✅ High detection accuracy using YOLOv7 pretrained weights

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Object Detection | YOLOv7 |
| Backend | Python |
| Video Processing | OpenCV |
| Frontend Dashboard | HTML, CSS, JavaScript |
| Data Handling | NumPy, Pandas |

---

## 🏗️ System Architecture

```
Live Video Feed
      ↓
YOLOv7 Detection Model
      ↓
Python Backend Pipeline
  ├── Vehicle Classification
  ├── Count Aggregation
  └── Density Calculation
      ↓
Web Dashboard (Real-Time Display)
```

---

## ⚙️ How It Works

1. **Video Input** — Live feed is captured frame by frame using OpenCV
2. **Detection** — Each frame is passed through YOLOv7 which draws bounding boxes around detected vehicles
3. **Classification** — Vehicles are categorised by type (car, truck, bus, motorcycle)
4. **Backend Processing** — Python pipeline aggregates counts, calculates density, and flags high-traffic conditions
5. **Dashboard** — Results are pushed to a web interface updating in real time

---

## 🚀 Getting Started

### Prerequisites
```bash
Python 3.8+
pip install -r requirements.txt
```

### Installation
```bash
# Clone the repository
git clone https://github.com/AnmolBhayana/Traffic-Management-System.git
cd Traffic-Management-System

# Install dependencies
pip install -r requirements.txt

# Download YOLOv7 weights
wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt
```

### Run
```bash
python app.py
```
Then open `http://localhost:5000` in your browser.

---

## 📊 Results

- Achieved high vehicle detection accuracy on highway footage
- System processes video frames with minimal latency
- Dashboard updates vehicle counts and density metrics in real time

---

## 🔮 Future Improvements

- Add automatic incident detection (stopped vehicles, wrong-way driving)
- Integrate traffic signal control based on density data
- Deploy on cloud with support for multiple camera feeds

---

## 👤 Author

**Anmol Bhayana**
[LinkedIn](https://linkedin.com/in/a-721a2) • [GitHub](https://github.com/AnmolBhayana)

---

> Built during internship at Innovate Technologies, Lucknow — Mar 2025 to Jun 2025
