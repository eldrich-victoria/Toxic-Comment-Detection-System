import os
import sys

# 1. FIX: Solve the "OMP: Error #15" crash (Must be at the very top)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np

# 2. FIX: Check for GPU/Accelerator
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_pin_memory = torch.cuda.is_available() # Only use pin_memory if GPU is found

print(f"--- System Check ---")
print(f"Device: {device}")
print(f"Pin Memory Enabled: {use_pin_memory}")
print("--------------------")

# --- Dummy Dataset for Demo ---
class ToxicDataset(Dataset):
    def __init__(self, size=100):
        self.data = torch.randn(size, 10)  # Simulated features
        self.labels = torch.randint(0, 2, (size,)) # 0 or 1 (Toxic)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# --- Simple Model ---
class ToxicModel(nn.Module):
    def __init__(self):
        super(ToxicModel, self).__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)

def main():
    # 3. FIX: DataLoader with dynamic pin_memory setting
    dataset = ToxicDataset()
    train_loader = DataLoader(
        dataset, 
        batch_size=16, 
        shuffle=True, 
        pin_memory=use_pin_memory  # This prevents the UserWarning on CPU
    )

    model = ToxicModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print("Starting training loop...")
    for epoch in range(1):
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
    print("Process complete with no errors!")

if __name__ == "__main__":
    main()