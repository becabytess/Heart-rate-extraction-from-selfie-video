from torch.utils.data import Dataset, DataLoader
import numpy as np 
import torch 
import os 
from scipy.signal import butter, filtfilt  

# sample_subject = subjects[0]
# sample_roi = np.loadtxt(os.path.join(data_path, sample_subject, 'roi_colors.txt'), delimiter=',')


# sample_subject = subjects[0]
# sample_signal = np.loadtxt(os.path.join(data_path, sample_subject, 'ground_truth.txt'))
# sample_roi  = np.loadtxt(os.path.join(data_path, sample_subject, 'roi_colors.txt'), delimiter=',')
# print(f"Sample ROI shape: {sample_roi.shape}")
# print(f"Sample signal shape: {sample_signal.shape}")

#roi shape: (1547,9(average rgb values from 3 regions: left cheek , right cheek and forehead)), signal shape: (3,1547)
#each frame has 3 labels in the dataset : Normalized Blood Volume Pulse , Heart Rate and Time. We will only train on the first one

class UBFC_Dataset(Dataset):
    def __init__(self, data_path, subjects,seq_len=300):
        self.data_path = data_path 
        self.subjects = subjects 
        self.seq_len = seq_len
        self.possible_ranges = [] 
        for subject in subjects:
            signal_path = os.path.join(data_path,subject, 'ground_truth.txt')
            signal = np.loadtxt(signal_path)
            num_starts = signal.shape[-1] - seq_len 
            for i in range(num_starts): 
                self.possible_ranges.append((subject, i)) 
    def __len__(self):
        return len(self.possible_ranges)
     
    def bandpass_filter(self, data, lowcut, highcut, fs, order=3):
        nyq = 0.5 * fs 
        low = lowcut / nyq 
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        y = filtfilt(b, a, data,axis=0)
        return y

    



    def __getitem__(self, index):
        subject,i = self.possible_ranges[index] 
        signal_path = os.path.join(self.data_path,subject, 'ground_truth.txt')
        colors_path = os.path.join(self.data_path, subject, 'roi_colors.txt')
        signals = np.loadtxt(signal_path)
        colors = np.loadtxt(colors_path, delimiter=',')
        signal_seq = signals[0,i : i + self.seq_len]
        color_seq = colors[i : i + self.seq_len]
        #seperate the 9 channels into 3 regions and normalize each region separately
        color_seq = color_seq.reshape(self.seq_len,3,3)
       
        mean = color_seq.mean(axis=0, keepdims=True)
        std = color_seq.std(axis=0, keepdims=True)
        color_seq = color_seq - mean / (std + 1e-6) #per roi , per channel normalization (z-score)
        
        color_seq = color_seq.reshape(self.seq_len,9) #flatten back to (seq_len, 9)
        signal_seq = self.bandpass_filter(signal_seq, lowcut=0.7, highcut=2.5, fs=30, order=2) #bandpass filter the signal sequence
        signal_seq = torch.tensor(signal_seq.copy(), dtype=torch.float32)

        return torch.tensor(color_seq, dtype=torch.float32), torch.unsqueeze(signal_seq, dim=-1)  

    
