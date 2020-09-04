import os
import torch

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"