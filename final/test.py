import os
import cv2
import urllib.request
import numpy as np
import torch
from scipy.signal import butter, sosfiltfilt, detrend, welch
from model import rPPGModel

# ---------------------------
# Download YuNet face detector model if missing
# ---------------------------
YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
YUNET_PATH = "yunet.onnx"

if not os.path.exists(YUNET_PATH):
    print("Downloading YuNet face detector model...")
    urllib.request.urlretrieve(YUNET_URL, YUNET_PATH)
    print("Done.")

# ---------------------------
# Signal Processing Utilities
# ---------------------------
def bandpass_filter(data, lowcut=0.85, highcut=2.5, fs=30.0, order=2):
    nyq = 0.5 * fs
    sos = butter(order, [lowcut / nyq, highcut / nyq], btype='band', output='sos')
    return sosfiltfilt(sos, data, axis=0)

def smooth_signal(signal_data, window=7):
    kernel = np.ones(window) / window
    return np.convolve(signal_data, kernel, mode='same')

def extract_roi_means_and_landmarks(video_path):
    """
    Extract facial ROI color means and facial contours using YuNet face detector.
    Returns:
      roi_colors: (T, 9) array containing [forehead (BGR), left_cheek (BGR), right_cheek (BGR)]
      face_hulls: List of convex hull polygons per frame for visual overlay rendering
      fps: Video frame rate
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    detector = cv2.FaceDetectorYN.create(YUNET_PATH, "", (w, h), 0.6, 0.3, 5000)

    roi_colors = []
    face_hulls = []

    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            break

        detector.setInputSize((img.shape[1], img.shape[0]))
        _, faces = detector.detect(img)

        if faces is not None and len(faces) > 0:
            face = faces[0]
            x1, y1, fw, fh = face[:4].astype(int)

            forehead_x1 = max(0, x1 + int(0.2 * fw))
            forehead_x2 = min(w, x1 + int(0.8 * fw))
            forehead_y1 = max(0, y1)
            forehead_y2 = min(h, y1 + int(0.25 * fh))

            left_cheek_x1 = max(0, x1)
            left_cheek_x2 = min(w, x1 + int(0.35 * fw))
            left_cheek_y1 = max(0, y1 + int(0.35 * fh))
            left_cheek_y2 = min(h, y1 + int(0.75 * fh))

            right_cheek_x1 = max(0, x1 + int(0.65 * fw))
            right_cheek_x2 = min(w, x1 + fw)
            right_cheek_y1 = max(0, y1 + int(0.35 * fh))
            right_cheek_y2 = min(h, y1 + int(0.75 * fh))

            forehead_roi = img[forehead_y1:forehead_y2, forehead_x1:forehead_x2]
            left_cheek_roi = img[left_cheek_y1:left_cheek_y2, left_cheek_x1:left_cheek_x2]
            right_cheek_roi = img[right_cheek_y1:right_cheek_y2, right_cheek_x1:right_cheek_x2]

            f_c = np.mean(forehead_roi, axis=(0, 1)) if forehead_roi.size > 0 else np.zeros(3)
            l_c = np.mean(left_cheek_roi, axis=(0, 1)) if left_cheek_roi.size > 0 else np.zeros(3)
            r_c = np.mean(right_cheek_roi, axis=(0, 1)) if right_cheek_roi.size > 0 else np.zeros(3)

            merged = np.concatenate([f_c, l_c, r_c])
            roi_colors.append(merged)

            box_pts = np.array([
                [x1, y1], [x1 + fw, y1],
                [x1 + fw, y1 + fh], [x1, y1 + fh]
            ], dtype=np.int32)
            face_hulls.append(box_pts)
        else:
            if len(roi_colors) > 0:
                roi_colors.append(roi_colors[-1])
                face_hulls.append(face_hulls[-1])
            else:
                roi_colors.append(np.zeros(9))
                face_hulls.append(None)

    cap.release()
    return np.array(roi_colors), face_hulls, fps

def estimate_bpm(sig, fs=30.0, lowcut=0.85, highcut=2.5, window_sec=10.0, stride_sec=2.0):
    """
    Robust Heart Rate (BPM) Estimator.
    Computes spectral peaks over sliding 10-second windows and calculates the median 
    to isolate cardiac pulse signals from ambient low-frequency noise.
    """
    win_len = int(window_sec * fs)
    stride = int(stride_sec * fs)

    if len(sig) < win_len:
        f, pxx = welch(sig, fs=fs, nperseg=len(sig))
        mask = (f >= lowcut) & (f <= highcut)
        return float(f[mask][np.argmax(pxx[mask])] * 60.0)

    window_bpms = []
    for start in range(0, len(sig) - win_len + 1, stride):
        window = sig[start:start + win_len]
        f, pxx = welch(window, fs=fs, nperseg=len(window))
        mask = (f >= lowcut) & (f <= highcut)
        if np.any(mask):
            bpm = f[mask][np.argmax(pxx[mask])] * 60.0
            window_bpms.append(bpm)

    if len(window_bpms) == 0:
        return 75.0

    return float(np.median(window_bpms))

def normalize_window(window):
    """Per-ROI per-channel z-score normalization matching training data."""
    w = window.reshape(window.shape[0], 3, 3)
    mean = w.mean(axis=0, keepdims=True)
    std = w.std(axis=0, keepdims=True)
    w = (w - mean) / (std + 1e-6)
    return w.reshape(window.shape[0], 9)

def run_model_windowed(model, colors_filtered, seq_len=300, stride=150):
    """Run model inference over sliding 300-frame windows with overlap-averaging."""
    T = colors_filtered.shape[0]
    orig_T = T

    if T < seq_len:
        pad = seq_len - T
        colors_filtered = np.pad(colors_filtered, ((0, pad), (0, 0)), mode='edge')
        T = seq_len

    starts = list(range(0, T - seq_len + 1, stride))
    if starts[-1] != T - seq_len:
        starts.append(T - seq_len)

    pred_sum = np.zeros(T, dtype=np.float64)
    pred_count = np.zeros(T, dtype=np.float64)

    model.eval()
    with torch.no_grad():
        for start in starts:
            window = colors_filtered[start:start + seq_len]
            window = normalize_window(window)
            x = torch.tensor(window, dtype=torch.float32).unsqueeze(0)
            pred = model(x)[0, :, 0].numpy()

            pred_sum[start:start + seq_len] += pred
            pred_count[start:start + seq_len] += 1

    preds = pred_sum / np.maximum(pred_count, 1)
    return preds[:orig_T]

def render_output_video(video_path, preds, face_hulls, model_bpm, out_video_path="output_video.mp4", window_sec=5.0):
    """Render output video with pulsing soft face glow and 5s rolling pulse wave graph."""
    print(f"Rendering output video: {out_video_path}...")
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(out_video_path,
                          cv2.VideoWriter_fourcc(*'mp4v'),
                          fps, (w, h + 140))

    idx = 0
    history = []
    max_pts = int(fps * window_sec)  # 5 seconds window

    preds_vis = (preds - np.mean(preds)) / (np.std(preds) + 1e-6)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if idx < len(preds_vis):
            val = preds_vis[idx]
            history.append(val)

            alpha = max(0.05, min(0.45, 0.25 + 0.18 * val))

            if idx < len(face_hulls) and face_hulls[idx] is not None:
                hull = face_hulls[idx]

                face_mask = np.zeros_like(frame)
                cv2.fillConvexPoly(face_mask, hull, (0, 0, 240))
                face_mask = cv2.GaussianBlur(face_mask, (25, 25), 0)

                frame = cv2.addWeighted(frame, 1.0, face_mask, alpha, 0)

        graph = np.zeros((140, w, 3), dtype=np.uint8)
        cv2.line(graph, (0, 70), (w, 70), (50, 50, 50), 1)

        if len(history) > 1:
            recent_pts = history[-max_pts:]
            n_pts = len(recent_pts)
            x_coords = np.linspace(0, w - 1, n_pts).astype(int)

            y_coords = 70 - (np.array(recent_pts) * 35).astype(int)
            y_coords = np.clip(y_coords, 15, 125)

            for i in range(1, n_pts):
                cv2.line(graph,
                         (x_coords[i - 1], y_coords[i - 1]),
                         (x_coords[i], y_coords[i]),
                         (0, 255, 0), 2)

        cv2.putText(frame, f"BPM: {int(round(model_bpm))}",
                    (25, 55),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (0, 255, 0), 3)

        cv2.putText(graph, "rPPG Pulse Wave (Last 5 sec)",
                    (15, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (200, 200, 200), 1)

        combined = np.vstack((frame, graph))
        out.write(combined)

        idx += 1

    cap.release()
    out.release()
    print(f"Saved -> {out_video_path}")

# ---------------------------
# MAIN INFERENCE PIPELINE
# ---------------------------
if __name__ == "__main__":
    video_path = "sample_video_2.mp4"
    model_path = "best_model.pth"
    output_video_path = "output_video.mp4"

    print(f"Processing video: {video_path}...")
    roi_colors, face_hulls, fps = extract_roi_means_and_landmarks(video_path)

    # Detrend & bandpass filter full video recording using actual video FPS
    colors_filtered = detrend(roi_colors, axis=0)
    colors_filtered = bandpass_filter(colors_filtered, 0.7, 2.5, fs=fps, order=2)

    # Load trained rPPG Transformer model
    model = rPPGModel(input_size=9, d_model=64, nhead=4, ff_hidden_size=128, num_layers=2, output_size=1)
    chkpt = torch.load(model_path, map_location="cpu")
    model.load_state_dict(chkpt["model_state_dict"])
    model.eval()

    # Windowed model inference
    preds = run_model_windowed(model, colors_filtered, seq_len=300, stride=150)
    preds = bandpass_filter(preds, 0.85, 2.5, fs=fps, order=2)
    preds = smooth_signal(preds, 9)

    # Estimate Heart Rate (BPM) using sliding 10-second window median
    estimated_bpm = estimate_bpm(preds, fs=fps)
    print(f"Estimated Heart Rate (BPM): {estimated_bpm:.2f}")

    # Render Output Video
    render_output_video(video_path, preds, face_hulls=face_hulls, model_bpm=estimated_bpm, out_video_path=output_video_path)