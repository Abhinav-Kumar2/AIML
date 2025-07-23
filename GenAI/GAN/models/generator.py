import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.gen = nn.Sequential(
            nn.ConvTranspose2d(z_dim, 128, kernel_size=7, stride=1, padding=0),  # 128 x 7 x 7
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),     # 64 x 14 x 14
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),       # 1 x 28 x 28
            nn.Tanh()
        )

    def forward(self, x):
        return self.gen(x.view(x.size(0), x.size(1), 1, 1))  # reshape z to (N, z_dim, 1, 1)
