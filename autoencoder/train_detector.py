import gc
import argparse
import torch.utils.data
from scipy import misc
from torch import optim
from torchvision.utils import save_image
from models import Detector, BiDetector
import os
from dataset_reader import dataset_loader
from torch.utils.data import DataLoader
import logging
from tensorboardX import SummaryWriter
from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np
from torch import nn

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
                args.database_path + "{}_{}_train/".format(prefix_2019.split(".")[0], args.track)
            ),
            algo=0,
            ae_detector=not args.bidetector,
            noise_detector=args.bidetector,
        )

        train_loader = DataLoader(
            train_set,
            batch_size=args.batch_size,
            num_workers=0,
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
                args.database_path + "{}_{}_dev/".format(prefix_2019.split(".")[0], args.track)
            ),
            algo=0,
            ae_detector=not args.bidetector,
            noise_detector=args.bidetector,
        )
        dev_loader = DataLoader(dev_set, batch_size=args.batch_size, num_workers=0, shuffle=False)
        del dev_set, d_label_dev
        return dev_loader


def loss_function(y_pred, y, weight):
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
    for (batch_x1, batch_x2), batch_y in dev_loader:
        batch_size = batch_x1.size(0)
        num_total += batch_size
        batch_x1 = batch_x1.to(device)
        batch_x2 = batch_x2.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        batch_out = model(batch_x1, batch_x2)
        all_labels.extend(batch_y.cpu().numpy())
        all_preds.extend(batch_out.cpu().numpy()[:, 1])

        batch_loss = loss_function(batch_out, batch_y, weight)
        val_loss += batch_loss.item() * batch_size

    val_loss /= num_total

    aucroc_e = roc_auc_score(all_labels, all_preds)
    # EER
    fpr, tpr, _ = roc_curve(all_labels, all_preds, pos_label=1)
    fnr = 1 - tpr
    abs_diffs = np.abs(fpr - fnr)
    min_index = np.argmin(abs_diffs)
    eer_e = np.mean((fpr[min_index], fnr[min_index]))

    return val_loss, aucroc_e, eer_e


def train_epoch(train_loader, model, optim, device):
    running_loss = 0

    num_total = 0.0

    model.train()
    batch = 0
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    for (batch_x1, batch_x2), batch_y in train_loader:
        batch_size = batch_x1.size(0)
        num_total += batch_size

        batch_x1 = batch_x1.to(device)
        batch_x2 = batch_x2.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        batch_out = model(batch_x1, batch_x2)
        train_loss = loss_function(batch_out, batch_y, weight)
        running_loss += train_loss.item() * batch_size

        optim.zero_grad()
        train_loss.backward()
        optim.step()
        batch += 1

    running_loss /= num_total

    return running_loss


def main():
    # Create the argument parser
    parser = argparse.ArgumentParser(description="ASVspoof2021 baseline system")

    # Dataset
    parser.add_argument(
        "--database_path",
        type=str,
        default="/mnt/c/Users/gabri/Desktop/avs/DF/",
        help="Change this to user's full directory address of LA database (ASVspoof2019- for training & development (used as validation), ASVspoof2021 for evaluation scores). We assume that all three ASVspoof 2019 LA train, LA dev, and ASVspoof2021 LA eval data folders are in the same database_path directory.",
    )
    parser.add_argument(
        "--protocols_path",
        type=str,
        default="/mnt/c/Users/gabri/Desktop/avs/protocols_path/",
        help="Change with path to user's LA database protocols directory address",
    )

    parser.add_argument(
        "--track", type=str, default="LA", choices=["LA", "PA", "DF"], help="LA/PA/DF"
    )
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--model_path", type=str, default=None, help="Model checkpoint to load")
    parser.add_argument(
        "--continue_epoch", type=str, default=0, help="Initial epoch to continue from"
    )
    parser.add_argument(
        "--bidetector",
        action="store_true",
        default=False,
        help="Use bi spectrogram for audio detection",
    )

    args = parser.parse_args()

    # DATA FROM HERE
    # get data_loader
    train_loader = get_data_loader("train", args)
    dev_loader = get_data_loader("dev", args)

    # define model saving path
    model_tag = "model_detector_noise_finetune_vae_2"
    model_save_path = os.path.join("models", model_tag)

    # create models path if doesn't exist
    if not os.path.exists("models"):
        os.mkdir("models")
    # set model save directory
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)

    # MODEL FROM HERE
    # GPU device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info("Device: {}".format(device))

    # initialize modellogging
    if args.bidetector:
        print('loaded bidetector')
        model = BiDetector(image_channels=1, device=device).to(device)
    else:
        model = Detector(image_channels=1, device=device).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    if args.model_path:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        logging.info(f"Model loaded : {args.model_path}")

    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    logging.info(f"nb_params: {nb_params}")

    # TRAINING HERE

    # Training and validation
    num_epochs = 150

    with SummaryWriter(f"logs/{model_tag}") as w:
        for epoch in range(args.continue_epoch, num_epochs):
            gc.collect()
            torch.cuda.empty_cache()
            running_loss = train_epoch(train_loader, model, optimizer, device)
            w.add_scalar("loss", running_loss, epoch)

            torch.cuda.empty_cache()
            gc.collect()
            val_loss, val_aucroc, val_err = evaluate_accuracy(dev_loader, model, device)
            w.add_scalar("val_loss", val_loss, epoch)
            w.add_scalar("val_err", val_err, epoch)
            w.add_scalar("val_aucroc", val_aucroc, epoch)

            torch.save(model.state_dict(), os.path.join(model_save_path, f"epoch_{epoch}.pth"))
            logging.info(f"epoch {epoch} ended loss {running_loss} val loss {val_loss}")
            logging.info(f"val metrics err {val_err} aucroc {val_aucroc}")


if __name__ == "__main__":
    main()
