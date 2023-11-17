import gc
import argparse
import torch.utils.data
from scipy import misc
from torch import optim
from torchvision.utils import save_image
from models import VAE
import numpy as np
import pickle
import time
import random
import os
from dataset_reader import dataset_loader
from torch.utils.data import DataLoader
import logging
from tensorboardX import SummaryWriter
import torch.nn.functional as F

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

def get_data_loader(dataset, args):
    prefix = f"ASVspoof_{args.track}"
    prefix_2019 = f"ASVspoof2019.{args.track}"
    if dataset == "train":
        d_label_trn, file_train = dataset_loader.genSpoof_list(
            dir_meta=os.path.join(
                args.protocols_path
                + "{}_cm_protocols/{}.cm.train.trn.txt".format(prefix, prefix_2019)
            ),
            is_train=True,
            is_eval=False,
        )

        logging.info(f"no. of training trials {len(file_train)}")

        train_set = dataset_loader.Dataset_ASVspoof2019_train(
            args,
            list_IDs=file_train,
            labels=d_label_trn,
            base_dir=os.path.join(
                args.database_path
                + "{}_{}_train/".format(prefix_2019.split(".")[0], args.track)
            ),
            algo=0,
            ae=True,
        )

        train_loader = DataLoader(
            train_set,
            batch_size=args.batch_size,
            num_workers=4,
            shuffle=True,
            drop_last=True,
        )

        del train_set, d_label_trn
        return train_loader

    elif dataset == "dev":
        d_label_dev, file_dev = dataset_loader.genSpoof_list(
            dir_meta=os.path.join(
                args.protocols_path
                + "{}_cm_protocols/{}.cm.dev.trl.txt".format(prefix, prefix_2019)
            ),
            is_train=False,
            is_eval=False,
        )

        logging.info(f"no. of validation trials {len(file_dev)}")

        dev_set = dataset_loader.Dataset_ASVspoof2019_train(
            args,
            list_IDs=file_dev,
            labels=d_label_dev,
            base_dir=os.path.join(
                args.database_path
                + "{}_{}_dev/".format(prefix_2019.split(".")[0], args.track)
            ),
            algo=0,
            ae=True,
        )
        dev_loader = DataLoader(
            dev_set, batch_size=args.batch_size, num_workers=4, shuffle=False
        )
        del dev_set, d_label_dev
        return dev_loader


def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction="sum")
    #BCE = F.mse_loss(recon_x, x, reduction="sum")

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    #KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    kld_loss = -0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp())

    return BCE + 2*kld_loss, BCE, kld_loss


@torch.no_grad()
def evaluate_accuracy(dev_loader, model, device):
    val_loss = 0.0
    num_total = 0.0
    bce_loss = 0
    kld_loss = 0
    model.eval()

    for batch_x, batch_y in dev_loader:
        batch_size = batch_x.size(0)
        num_total += batch_size
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        batch_out, mu, logvar = model(batch_x)

        batch_loss, bce, kld = loss_function(batch_out, batch_y, mu, logvar)
        val_loss += batch_loss.item() * batch_size
        bce_loss += bce.item() * batch_size
        kld_loss += kld.item() * batch_size

    val_loss /= num_total
    bce_loss /= num_total
    kld_loss /= num_total

    return val_loss, bce_loss, kld_loss


def train_epoch(train_loader, model, optim, device):
    running_loss = 0
    bce_loss = 0
    kld_loss = 0

    num_total = 0.0

    model.train()
    for batch_x, batch_y in train_loader:
        batch_size = batch_x.size(0)
        num_total += batch_size

        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        batch_out, mu, logvar = model(batch_x)
        train_loss, bce, kld = loss_function(batch_out, batch_y, mu, logvar)
        running_loss += train_loss.item() * batch_size
        bce_loss += bce.item() * batch_size
        kld_loss += kld.item() * batch_size

        optim.zero_grad()
        train_loss.backward()
        optim.step()

    running_loss /= num_total
    bce_loss /= num_total
    kld_loss /= num_total

    return running_loss, bce_loss, kld_loss


def main():
    # Create the argument parser
    parser = argparse.ArgumentParser(description="ASVspoof2021 baseline system")

    # Dataset
    parser.add_argument(
        "--database_path",
        type=str,
        default="/home/cassuci/repos/avs/DF/",
        help="Change this to user's full directory address of LA database (ASVspoof2019- for training & development (used as validation), ASVspoof2021 for evaluation scores). We assume that all three ASVspoof 2019 LA train, LA dev, and ASVspoof2021 LA eval data folders are in the same database_path directory.",
    )
    parser.add_argument(
        "--protocols_path",
        type=str,
        default="/home/cassuci/repos/avs/protocols_path/",
        help="Change with path to user's LA database protocols directory address",
    )

    parser.add_argument(
        "--track", type=str, default="LA", choices=["LA", "PA", "DF"], help="LA/PA/DF"
    )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument(
        "--model_path", type=str, default=None, help="Model checkpoint to load"
    )

    args = parser.parse_args()

    # DATA FROM HERE
    # get data_loader
    train_loader = get_data_loader("train", args)
    dev_loader = get_data_loader("dev", args)

    # define model saving path
    model_tag = "model_vae_new_out_padding"
    model_save_path = os.path.join("models", model_tag)

    # create models path if doesn't exist
    if not os.path.exists("models"):
        os.mkdir("models")
    # set model save directory
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)

    # MODEL FROM HERE
    batch_size = args.batch_size
    # GPU device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info("Device: {}".format(device))

    # initialize modellogging
    model = VAE(image_channels=1, device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-8)

    if args.model_path:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        logging.info(f"Model loaded : {args.model_path}")

    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    logging.info(f"nb_params: {nb_params}")

    # TRAINING HERE

    # Training and validation
    num_epochs = 150

    with SummaryWriter(f"logs/{model_tag}") as w:
        for epoch in range(num_epochs):
            gc.collect()
            torch.cuda.empty_cache()
            running_loss, train_bce, train_kld = train_epoch(
                train_loader, model, optimizer, device
            )
            w.add_scalar("loss", running_loss, epoch)
            w.add_scalar("train_bce", train_bce, epoch)
            w.add_scalar("train_kld", train_kld, epoch)

            torch.cuda.empty_cache()
            gc.collect()
            val_loss, val_bce, val_kld = evaluate_accuracy(dev_loader, model, device)
            w.add_scalar("val_loss", val_loss, epoch)
            w.add_scalar("val_bce", val_bce, epoch)
            w.add_scalar("val_kld", val_kld, epoch)

            torch.save(
                model.state_dict(), os.path.join(model_save_path, f"epoch_{epoch}.pth")
            )
            logging.info(f'epoch {epoch} ended ')
            logging.info(f"metrics loss {running_loss} val_loss {val_loss}")


if __name__ == "__main__":
    main()
