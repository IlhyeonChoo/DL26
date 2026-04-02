import torch

# 1. device setting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
