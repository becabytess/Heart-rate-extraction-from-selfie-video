# Remote Photoplethysmography (rPPG) from Selfie Videos

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)](https://opencv.org/)

Non-invasive measurement of **Blood Volume Pulse (BVP)** signals and **Heart Rate (BPM)** from standard camera video feeds using facial ROI extraction, continuous Butterworth bandpass filtering, and a temporal Transformer sequence model.

---

## 🎬 Real-Time Inference Demo

The production pipeline tracks facial skin ROIs using YuNet face detection, reconstructs the underlying cardiac blood volume pulse wave using a Transformer neural network, and renders real-time visual overlays:

- **Soft Red Face Glow**: Dynamic facial contour overlay (`cv2.convexHull` + Gaussian edge blur) pulsing in rhythm with each cardiac beat.
- **5-Second Rolling Pulse Wave**: Real-time graph displaying the reconstructed rPPG waveform.
- **Live Heart Rate Display**: Real-time heart rate (BPM) computed via Welch Power Spectral Density.

![rPPG Live Demo](final/output_video.gif)

| Input Selfie Video (7.0 MB) | Full Output Video (25.8 MB) |
| :---: | :---: |
| [`final/sample_video_2.mp4`](final/sample_video_2.mp4) | [`final/output_video.mp4`](final/output_video.mp4) |

---

## 📊 Held-Out Validation Benchmark (Seed 42)

The Transformer model (`rPPGModel`) was evaluated on a **held-out 20% validation split (13,761 sequences, seed 42)** from the **UBFC-RPPG Dataset**, comparing predicted pulse signals against ground-truth finger-clip BVP sensors:

| Evaluation Metric | Held-Out Validation Benchmark |
| :--- | :---: |
| **Mean Pearson Correlation ($r$)** | **`+0.9080`** |
| **Median Pearson Correlation ($r$)** | **`+0.9135`** |
| **Mean Validation Loss (MSE + Pearson)** | **`0.1733`** |

---

## 🏗️ Repository Layout

```text
.
├── final/                  # Production model, preprocessing & inference pipeline
│   ├── best_model.pth      # Trained Transformer checkpoint weights
│   ├── model.py            # PyTorch Transformer Architecture (rPPGModel)
│   ├── datasets.py         # UBFC-RPPG Dataset loader (Continuous Filtering Pass)
│   ├── train.py            # Model training script (Modal Cloud T4 GPU)
│   ├── test.py             # Inference pipeline & video renderer
│   ├── sample_video_2.mp4  # Lightweight input sample video (7.0 MB)
│   ├── output_video.gif    # Auto-playing README demo animation (6.0 MB)
│   └── output_video.mp4    # Rendered video output (25.8 MB)
├── experiments/            # Exploratory research, notebooks & baseline models
├── data/                   # UBFC-RPPG Dataset cache
├── yunet.onnx              # Lightweight YuNet face detection model
└── README.md               # Project documentation
```

---

## ⚙️ How It Works

1. **Face & ROI Extraction**: YuNet (`yunet.onnx`) detects the face and tracks 3 skin regions: **Forehead**, **Left Cheek**, and **Right Cheek**.
2. **Spatial Feature Reduction**: Spatial BGR color averaging across each ROI produces a 9-dimensional frame feature vector `[Forehead (BGR), Left Cheek (BGR), Right Cheek (BGR)]`.
3. **Continuous Signal Preprocessing**:
   - Detrending removes slow illumination drifts and posture shifts.
   - Butterworth bandpass filtering ($0.7\text{ Hz} - 2.5\text{ Hz} \equiv 42 - 150\text{ BPM}$) isolates physiological pulse signals.
4. **Windowed Transformer Inference**:
   - Sliding 300-frame windows ($10\text{ seconds}$ at $30\text{ FPS}$) are normalized per-window via z-score scaling.
   - Self-attention encoder layers reconstruct long-range temporal cardiac pulse waveforms.
   - Overlapping predictions are stitched via linear window averaging.
5. **Sliding Window Heart Rate Estimation**:
   - Computes the median Welch Power Spectral Density peak over sliding 10-second windows to isolate physiological cardiac pulse signals from ambient low-frequency noise.

---

## 🚀 Quick Start

### 1. Installation

```bash
pip install torch numpy opencv-python scipy matplotlib modal
```

### 2. Run Inference & Video Rendering

Execute `test.py` under `final/` to run inference on the sample video:

```bash
cd final
python test.py
```

This generates:
- `output_video.mp4`: Rendered video with pulsing face glow overlay and real-time pulse graph.

### 3. Train on Cloud GPU (Modal)

To retrain or fine-tune the model on Modal infrastructure:

```bash
cd final
modal run train.py
```

---

## 📜 Citation & License

Trained and evaluated on the **UBFC-RPPG Dataset** (*Bobbia et al.*).
