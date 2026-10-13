# 🎥 Camera Detector

A modern, real-time computer vision desktop application built with **CustomTkinter**, **OpenCV**, and **MediaPipe**. It detects faces, eye landmarks, hands, and motion — all through a sleek dark-themed GUI with live statistics, snapshots, and smooth 30+ FPS performance.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.22-green)
![OpenCV](https://img.shields.io/badge/OpenCV-4.9+-red)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

---

## ✨ Features

| Feature | Description |
|---|---|
| 🧑 **Face Detection** | Real-time face bounding boxes via MediaPipe FaceMesh |
| 👁 **Eye / Landmark Tracking** | Live eye-contour overlays using 468-point face mesh |
| ✋ **Hand Tracking** | 21-point skeleton for up to 2 hands simultaneously |
| 📸 **Motion Detection** | Frame-differencing with contour bounding boxes |
| 🪞 **Mirror View** | Webcam-style horizontal flip |
| 📷 **Snapshots** | Save annotated frames as timestamped PNGs |
| 📊 **Live Stats** | Smoothed FPS, face/eye/hand counts |
| 🎛 **Toggle Modules** | Enable/disable each detector on the fly |
| 🎚 **Camera Selector** | Switch between cameras 0–3 without restarting |
| 🌗 **Dark UI** | Modern CustomTkinter interface, fully resizable |

---


---

## 📋 Requirements

- **Python 3.12** ⚠️ *(MediaPipe does NOT support Python 3.13)*
- Webcam (built-in or USB)
- Windows / Linux / macOS

> **Why Python 3.12?** MediaPipe 0.10.x only ships prebuilt wheels for Python 3.9–3.12. The latest MediaPipe releases (0.10.31+) removed the legacy `mp.solutions` API entirely, which this project relies on.

---

---

## 📋 Requirements

- **Python 3.12** ⚠️ *(MediaPipe does NOT support Python 3.13)*
- Webcam (built-in or USB)
- Windows / Linux / macOS

> **Why Python 3.12?** MediaPipe 0.10.x only ships prebuilt wheels for Python 3.9–3.12. The latest MediaPipe releases (0.10.31+) removed the legacy `mp.solutions` API entirely, which this project relies on.

---

## 🚀 Installation

### Option A — Using `uv` (recommended)

```bash
# Clone or navigate to the project
cd "Camera Detector"

# Create a Python 3.12 virtual environment
uv venv --python 3.12

# Activate it
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt

python camera_detetion.py
```

### Option B — Using standard pip

```bash
# Ensure you have Python 3.12
python --version   # should print 3.12.x

python -m venv .venv

# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

pip install -r requirements.txt

python camera_detetion.py
```
