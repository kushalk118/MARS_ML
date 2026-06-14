import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np

class AudioDataset2D(Dataset):
    """
    PyTorch Dataset that loads 2D feature matrices for audio files.
    """
    def __init__(self, file_paths, labels, features_2d_list=None):
        self.file_paths = file_paths
        self.labels = labels
        self.features_2d_list = features_2d_list

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # If we pre-extracted the features (highly recommended for speed)
        if self.features_2d_list is not None:
            feat_2d = self.features_2d_list[idx]
        else:
            # Otherwise extract on-the-fly (import here to avoid circular dependencies)
            from src.features import process_file
            _, feat_2d = process_file(self.file_paths[idx])
            
        # Add channel dimension: (1, n_mfcc, max_pad)
        feat_tensor = torch.tensor(feat_2d, dtype=torch.float32).unsqueeze(0)
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.float32)
        
        return feat_tensor, label_tensor

class DeepfakeAudioCNN(nn.Module):
    """
    A lightweight 2D Convolutional Neural Network (CNN) for audio deepfake classification.
    Expects input shape: (batch_size, 1, n_mfcc, time_steps) e.g., (batch_size, 1, 20, 128)
    """
    def __init__(self, n_mfcc=20, time_steps=128):
        super(DeepfakeAudioCNN, self).__init__()
        
        # Block 1: Input (1, 20, 128) -> Output (16, 10, 64)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block 2: Input (16, 10, 64) -> Output (32, 5, 32)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block 3: Input (32, 5, 32) -> Output (64, 2, 16)
        # Note: MaxPool2d with pool_size=2 on height 5 results in height 2 (integer division)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Compute flattened size: 64 channels * 2 height * 16 width
        self.flat_features = 64 * 2 * 16
        
        # Classifier
        self.fc1 = nn.Linear(self.flat_features, 64)
        self.relu4 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 1) # Raw logit output for BCEWithLogitsLoss
        
    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        
        # Flatten
        x = x.view(-1, self.flat_features)
        
        x = self.dropout(self.relu4(self.fc1(x)))
        x = self.fc2(x)
        return x
