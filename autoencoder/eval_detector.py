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
from models import *


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
parser.add_argument("--itw", action='store_true', default=False)


args = parser.parse_args("--track DF --eval_output eval_detector_bigger_itw.out --itw".split())

track = args.track
# database
prefix = "ASVspoof_{}".format(track)
prefix_2019 = "ASVspoof2019.{}".format(track)
prefix_2021 = "ASVspoof2021.{}".format(track)


# define model saving path
model_tag = "model_detector_vae_bigger"
model_save_path = os.path.join("models", model_tag)

# set model save directory
if not os.path.exists(model_save_path):
    os.mkdir(model_save_path)

# GPU device
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device: {}".format(device))


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
model.load_state_dict(torch.load("models/model_detector_vae_bigger/epoch_57.pth"))

if not args.itw:
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
else:
    print('running for itw')
    meta, filelist = dataset_loader.genSpoof_list_inthewild('/mnt/f/downloads/release_in_the_wild/meta.csv')
    print("no. of eval trials", len(filelist))
    eval_set = dataset_loader.Dataset_InTheWild_eval(filelist, meta, '/mnt/f/downloads/release_in_the_wild/', ae_detector=True)

produce_evaluation_file(eval_set, model, device, args.eval_output)
