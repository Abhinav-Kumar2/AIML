import torch

config = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "lr": 1e-4,
    "z_dim": 100,               
    "batch_size": 128,
    "num_epochs": 50,
    "log_dir": "runs/DCGAN_MNIST",
    "dataset_path": "dataset/"
}
