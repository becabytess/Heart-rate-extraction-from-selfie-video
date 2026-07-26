from torch.utils.data import Dataset, DataLoader
import numpy as np 
import torch 
import os 

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

    def __getitem__(self, index):
        subject,i = self.possible_ranges[index] 
        signal_path = os.path.join(self.data_path,subject, 'ground_truth.txt')
        colors_path = os.path.join(self.data_path, subject, 'roi_colors.txt')
        signals = np.loadtxt(signal_path)
        colors = np.loadtxt(colors_path, delimiter=',')
        signal_seq = signals[0,i : i + self.seq_len]
        color_seq = colors[i : i + self.seq_len]
        min_vals = color_seq.min(axis=0, keepdims=True)
        max_vals = color_seq.max(axis=0, keepdims=True)
        color_seq = (color_seq - min_vals) / (max_vals - min_vals + 1e-6)  #we are basically trying to ignore the constant offset in the color values and only focus on the changes in the color values which are indicative of the blood volume pulse
        signal_seq = torch.tensor(signal_seq, dtype=torch.float32)

        return torch.tensor(color_seq, dtype=torch.float32), torch.unsqueeze(signal_seq, dim=-1)  #returning the color sequence and the corresponding signal sequence

    
