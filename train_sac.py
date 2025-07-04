import torch

def train_sac():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    