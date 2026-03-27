import os
# This MUST be the first thing your script does
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch

# 1. Detect the best hardware
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} for training.")

# 2. Move your model to that hardware
model = ToxicModel().to(device)

# 3. Move data to the hardware inside your training loop
for inputs, labels in dataloader:
    inputs, labels = inputs.to(device), labels.to(device)
    # The training math now happens on your RTX GPU