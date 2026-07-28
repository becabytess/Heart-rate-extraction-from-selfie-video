import os
import torch
import numpy as np
from scipy.signal import butter, filtfilt, detrend
from torch.utils.data import Dataset

class UBFC_Dataset(Dataset):
    """
    Dataset loader for UBFC-RPPG dataset.
    Loads raw ROI colors (9 channels) and ground-truth pulse signals (BVP).
    Applies continuous detrending and bandpass filtering per subject prior to slicing 300-frame windows.
    """

    def __init__(self, data_path, subjects, seq_len=300,
                 lowcut=0.7, highcut=2.5, fs=30, order=2,
                 filter_target=False):
        self.data_path = data_path
        self.subjects = subjects
        self.seq_len = seq_len
        self.lowcut = lowcut
        self.highcut = highcut
        self.fs = fs
        self.order = order
        self.filter_target = filter_target

        self.possible_ranges = []
        self.filtered_colors = {}
        self.filtered_signal = {}

        for subject in subjects:
            signal_path = os.path.join(data_path, subject, 'ground_truth.txt')
            colors_path = os.path.join(data_path, subject, 'roi_colors.txt')

            if not os.path.exists(signal_path) or not os.path.exists(colors_path):
                continue

            with open(signal_path, 'r') as f:
                lines = f.readlines()
                signal = np.array([float(x) for x in lines[0].strip().split()])

            colors = np.loadtxt(colors_path, delimiter=',')

            # Detrend & bandpass filter full continuous subject recording
            colors_clean = detrend(colors, axis=0)
            colors_clean = self.bandpass_filter(colors_clean)

            if self.filter_target:
                signal_clean = detrend(signal)
                signal_clean = self.bandpass_filter(signal_clean)
            else:
                signal_clean = signal

            self.filtered_colors[subject] = colors_clean
            self.filtered_signal[subject] = signal_clean

            num_starts = len(signal) - seq_len
            for i in range(num_starts):
                self.possible_ranges.append((subject, i))

    def __len__(self):
        return len(self.possible_ranges)

    def bandpass_filter(self, data):
        nyq = 0.5 * self.fs
        low = self.lowcut / nyq
        high = self.highcut / nyq
        b, a = butter(self.order, [low, high], btype='band')
        return filtfilt(b, a, data, axis=0)

    def __getitem__(self, index):
        subject, i = self.possible_ranges[index]

        color_seq = self.filtered_colors[subject][i:i + self.seq_len].copy()
        signal_seq = self.filtered_signal[subject][i:i + self.seq_len].copy()

        # Per-ROI per-channel z-score normalization
        color_seq = color_seq.reshape(self.seq_len, 3, 3)
        mean = color_seq.mean(axis=0, keepdims=True)
        std = color_seq.std(axis=0, keepdims=True)
        color_seq = (color_seq - mean) / (std + 1e-6)
        color_seq = color_seq.reshape(self.seq_len, 9)

        color_seq = torch.tensor(color_seq, dtype=torch.float32)
        signal_seq = torch.tensor(signal_seq, dtype=torch.float32)

        return color_seq, torch.unsqueeze(signal_seq, dim=-1)