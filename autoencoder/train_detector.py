import pandas as pd
import logging

from dataset_reader import dataset_loader
import argparse
import os
from torch.utils.data import DataLoader
import torch
from torch import nn
from torch import Tensor
from torch.utils.data import DataLoader
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
    "--track", type=str, default="LA", choices=["LA", "PA", "DF"], help="LA/PA/DF"
)
parser.add_argument("--batch_size", type=int, default=1024)


args = parser.parse_args("--track LA".split())

track = args.track
# database
prefix = "ASVspoof_{}".format(track)
prefix_2019 = "ASVspoof2019.{}".format(track)

# define train dataloader
d_label_trn, file_train = dataset_loader.genSpoof_list(
    dir_meta=os.path.join(
        args.protocols_path
        + "{}_cm_protocols/{}.cm.train.trn.txt".format(prefix, prefix_2019)
    ),
    is_train=True,
    is_eval=False,
)

print("no. of training trials", len(file_train))

train_set = dataset_loader.Dataset_ASVspoof2019_train(
    args,
    list_IDs=file_train,
    labels=d_label_trn,
    base_dir=os.path.join(
        args.database_path
        + "{}_{}_train/".format(prefix_2019.split(".")[0], args.track)
    ),
    algo=0,
    ae_detector=True,
)

train_loader = DataLoader(
    train_set, batch_size=args.batch_size, num_workers=8, shuffle=True, drop_last=True
)

del train_set, d_label_trn


# define validation dataloader

d_label_dev, file_dev = dataset_loader.genSpoof_list(
    dir_meta=os.path.join(
        args.protocols_path
        + "{}_cm_protocols/{}.cm.dev.trl.txt".format(prefix, prefix_2019)
    ),
    is_train=False,
    is_eval=False,
)

print("no. of validation trials", len(file_dev))

dev_set = dataset_loader.Dataset_ASVspoof2019_train(
    args,
    list_IDs=file_dev,
    labels=d_label_dev,
    base_dir=os.path.join(
        args.database_path + "{}_{}_dev/".format(prefix_2019.split(".")[0], args.track)
    ),
    algo=0,
    ae_detector=True,
)
dev_loader = DataLoader(
    dev_set, batch_size=args.batch_size, num_workers=8, shuffle=False
)
del dev_set, d_label_dev

# make experiment reproducible
# set_random_seed(args.seed, args)

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
        self.vae = VAE(image_channels=image_channels, h_dim=h_dim, z_dim=z_dim, device=device)
        self.vae.load_state_dict(torch.load('models/model_vae/epoch_42.pth'))
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


model = Detector(image_channels=1, device=device).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
print("nb_params:", nb_params)


def loss_fn(y_pred, y, weight):
    criterion = nn.CrossEntropyLoss(weight=weight)
    return criterion(y_pred, y)


@torch.no_grad()
def evaluate_accuracy(dev_loader, model, device):
    val_loss = 0.0
    num_total = 0.0
    model.eval()

    all_labels = []
    all_preds = []
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    for batch_x, batch_y in dev_loader:
        batch_size = batch_x.size(0)
        num_total += batch_size
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        batch_out = model(batch_x)
        all_labels.extend(batch_y.cpu().numpy())
        all_preds.extend(batch_out.cpu().numpy()[:, 1])

        batch_loss = loss_fn(batch_out, batch_y, weight)
        val_loss += batch_loss.item() * batch_size

    val_loss /= num_total

    aucroc_e = roc_auc_score(all_labels, all_preds)
    # EER
    fpr, tpr, _ = roc_curve(all_labels, all_preds, pos_label=1)
    fnr = 1 - tpr
    eer_e = fpr[np.nanargmin(np.absolute((fnr - fpr)))]

    return val_loss, aucroc_e, eer_e


def train_epoch(train_loader, model, optim, device):
    running_loss = 0

    num_total = 0.0

    model.train()
    batch = 0
    weight = torch.FloatTensor([0.1, 0.9]).to(device)

    for batch_x, batch_y in train_loader:
        batch_size = batch_x.size(0)
        num_total += batch_size

        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        batch_out = model(batch_x)
        train_loss = loss_fn(batch_out, batch_y, weight)
        running_loss += train_loss.item() * batch_size

        optim.zero_grad()
        train_loss.backward()
        optim.step()
        batch += 1

    running_loss /= num_total

    return running_loss


# Training and validation
num_epochs = 50
writer = SummaryWriter("logs/{}".format(model_tag))

for epoch in range(num_epochs):
    import gc

    gc.collect()
    torch.cuda.empty_cache()
    running_loss = train_epoch(
        train_loader, model, optimizer, device
    )
    writer.add_scalar("loss", running_loss, epoch)

    torch.cuda.empty_cache()
    gc.collect()
    val_loss, aucroc, val_eer = evaluate_accuracy(dev_loader, model, device)
    writer.add_scalar("val_loss", val_loss, epoch)

    torch.cuda.empty_cache()
    gc.collect()
    print("\nloss epoch:{} - train_loss: {} - val_loss: {} ".format(epoch, running_loss, val_loss))
    print("\nmetrics epoch:{} - val_aucroc: {} - val_err: {} ".format(epoch, aucroc, val_eer))
    torch.save(
        model.state_dict(), os.path.join(model_save_path, "epoch_{}.pth".format(epoch))
    )
