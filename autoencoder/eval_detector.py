import pandas as pd
import logging

from dataset_reader import dataset_loader
import argparse
import os
from torch.utils.data import DataLoader
import torch
from torch import Tensor
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Create the argument parser
parser = argparse.ArgumentParser(description="ASVspoof2021 baseline system")

# Dataset
parser.add_argument(
    "--database_path",
    type=str,
    default="/mnt/f/downloads/avs/DF/",
    help="Change this to user's full directory address of LA database (ASVspoof2019- for training & development (used as validation), ASVspoof2021 for evaluation scores). We assume that all three ASVspoof 2019 LA train, LA dev, and ASVspoof2021 LA eval data folders are in the same database_path directory.",
)
parser.add_argument(
    "--protocols_path",
    type=str,
    default="/mnt/f/downloads/avs/protocols_path/",
    help="Change with path to user's LA database protocols directory address",
)

parser.add_argument(
    "--eval_output",
    type=str,
    default="eval.out",
    help="",
)

parser.add_argument(
    "--track", type=str, default="LA", choices=["LA", "PA", "DF"], help="LA/PA/DF"
)
parser.add_argument("--batch_size", type=int, default=1024)


args = parser.parse_args("--track DF --eval_output eval.out".split())

track = args.track
# database
prefix = "ASVspoof_{}".format(track)
prefix_2019 = "ASVspoof2019.{}".format(track)
prefix_2021 = "ASVspoof2021.{}".format(track)


# define model saving path
model_tag = "model_detector"
model_save_path = os.path.join("models", model_tag)

# set model save directory
if not os.path.exists(model_save_path):
    os.mkdir(model_save_path)

# GPU device
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device: {}".format(device))


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def forward(self, input, size=5120):
        return input.view(input.size(0), size, 1, 1)


class VAE(nn.Module):
    def __init__(self, image_channels=1, h_dim=5120, z_dim=128, device=None):
        super(VAE, self).__init__()
        self.device = device
        self.encoder = nn.Sequential(  # 1, 1, 128, 253
            nn.Conv2d(image_channels, 32, kernel_size=4, stride=2),  # 1, 32, 63, 125
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),  # 1, 64, 30, 61
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),  # 1, 128, 14, 29
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2),  # 1, 256, 6, 13
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=4, stride=2),  # 1, 512, 2, 5
            nn.ReLU(),
            Flatten(),
        )

        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)

        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(h_dim, 128, kernel_size=(5, 5), dilation=2, stride=2),  #
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=(5, 5), dilation=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=(5, 6), dilation=2, stride=(2, 3)),
            nn.ReLU(),
            nn.ConvTranspose2d(
                32,
                image_channels,
                kernel_size=(6, 5),
                dilation=(3, 2),
                stride=(2, 3),
                padding=(0, 1),
            ),
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
    def __init__(self, image_channels=1, h_dim=5120, z_dim=128, device=None):
        super().__init__()
        self.device = device
        self.vae = VAE(
            image_channels=image_channels, h_dim=h_dim, z_dim=z_dim, device=device
        )
        self.vae.load_state_dict(torch.load("models/model_vae/epoch_42.pth"))
        self.vae.eval()

        self.relu = nn.ReLU(inplace=True)
        self.softmax = nn.Softmax(dim=1)

        self.fc1 = nn.Linear(z_dim, 32)
        self.fc2 = nn.Linear(32, 32)
        self.output = nn.Linear(32, 2)

    def forward(self, x):
        z, mu, logvar = self.vae.encode(x)
        x = self.fc1(z)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.output(x)
        x = self.softmax(x)
        return x


def produce_evaluation_file(dataset, model, device, save_path):
    data_loader = DataLoader(dataset, batch_size=1024, shuffle=False, drop_last=False)
    num_correct = 0.0
    num_total = 0.0
    model.eval()

    fname_list = []
    key_list = []
    score_list = []

    for batch_x, utt_id in data_loader:
        fname_list = []
        score_list = []
        batch_size = batch_x.size(0)
        batch_x = batch_x.to(device)

        batch_out = model(batch_x)

        batch_score = (batch_out[:, 1]).data.cpu().numpy().ravel()
        # add outputs
        fname_list.extend(utt_id)
        score_list.extend(batch_score.tolist())

        with open(save_path, "a+") as fh:
            for f, cm in zip(fname_list, score_list):
                fh.write("{} {}\n".format(f, cm))
        fh.close()
    print("Scores saved to {}".format(save_path))


model = Detector(image_channels=1, device=device).to(device)
model.load_state_dict(torch.load("models/model_detector/epoch_6.pth"))

file_eval = dataset_loader.genSpoof_list(
    dir_meta=os.path.join(
        args.protocols_path
        + "ASVspoof_{}_cm_protocols/{}.cm.eval.trl.txt".format(track, prefix_2021)
    ),
    is_train=False,
    is_eval=True,
)
print("no. of eval trials", len(file_eval))
eval_set = dataset_loader.Dataset_ASVspoof2021_eval(
    list_IDs=file_eval,
    base_dir=os.path.join(
        args.database_path + "ASVspoof2021_{}_eval/".format(args.track)
    ),
    ae_detector=True,
)
produce_evaluation_file(eval_set, model, device, args.eval_output)
