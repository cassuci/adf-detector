import torch
import torch.nn as nn
import torch.nn.functional as F

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def forward(self, input, size=640):
        return input.view(input.size(0), size, 1, 1)


class VAE(nn.Module):
    def __init__(self, image_channels=1, h_dim=640, z_dim=512, device=None):
        super().__init__()
        self.device = device
        self.encoder = nn.Sequential(  # 1, 1, 128, 253
            nn.Conv2d(image_channels, 4, kernel_size=4, stride=2),  # 1, 32, 63, 125
            nn.ReLU(),
            nn.Conv2d(4, 8, kernel_size=4, stride=2),  # 1, 64, 30, 61
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=4, stride=2),  # 1, 128, 14, 29
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),  # 1, 256, 6, 13
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),  # 1, 512, 2, 5
            nn.ReLU(),
            Flatten(),
        )

        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)

        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(h_dim, 64, kernel_size=(3,5), dilation=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=5, dilation=(1, 2), stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=5, dilation=(1, 2), stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=5, dilation=(1, 2), stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 4, kernel_size=5, dilation=(1, 2), stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(4, 1, kernel_size=(4, 5), dilation=1, stride=1),
            nn.Sigmoid(),
        )

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size(), device=self.device)
        z = mu + std * esp
        return z

    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

    def decode(self, z):
        z = self.fc3(z)
        z = self.decoder(z)
        return z

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        z = self.decode(z)
        return z, mu, logvar

class Detector(nn.Module):
    def __init__(self, image_channels=1, h_dim=640, z_dim=128, device=None):
        super().__init__()
        self.device = device
        self.vae = VAE(image_channels=image_channels, h_dim=h_dim, z_dim=z_dim, device=device)
        self.vae.load_state_dict(torch.load('models/model_vae_bce_sum_float/epoch_55.pth'))
        self.vae.eval()

        self.relu = nn.ReLU(inplace=True)
        self.softmax = nn.Softmax(dim=1)

        self.fc1 = nn.Linear(z_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.output = nn.Linear(16, 2)

    def forward(self, x):
        z, mu, logvar = self.vae.encode(x)
        x = self.fc1(z)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.output(x)
        x = self.softmax(x)
        return x