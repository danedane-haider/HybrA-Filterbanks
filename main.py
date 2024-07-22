import argparse
import datetime
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from pesq import pesq_batch

from src.datasets import Chime2
from src.losses import ComplexCompressedMSELoss
from model import HybridfilterbankModel
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio
from tqdm import tqdm

# set seed
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(0)
np.random.seed(0)


def main(args):
    EPOCHS = args.epochs
    VAL_EVERY = args.val_every
    BATCH_SIZE = args.batch_size
    NUM_WORKERS = args.num_workers
    FS = args.fs
    LOGGING_DIR = args.logging_dir
    DATASET = args.dataset
    SIGNAL_LENGTH = args.signal_length
    KAPPA_BETA = args.kappa_beta
    LEARNING_RATE = args.learning_rate

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    model = HybridfilterbankModel()

    print(
        "Number of model parameters: ",
        sum(
            [
                np.prod(p.size())
                for p in filter(lambda p: p.requires_grad, model.parameters())
            ]
        ),
    )

    model = model.to(device)

    # init tensorboard
    writer = SummaryWriter(f"{LOGGING_DIR}")

    if KAPPA_BETA is None:
        loss_fn = ComplexCompressedMSELoss()
    else:
        loss_fn = ComplexCompressedMSELoss(beta=KAPPA_BETA)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    dataset_train = Chime2(
        dataset=DATASET,
        type="train",
        fs=FS,
        signal_length=SIGNAL_LENGTH,
    )
    g = torch.Generator()
    g.manual_seed(0)
    dataloader_train = DataLoader(
        dataset_train,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        generator=g,
    )

    dataset_val = Chime2(
        dataset=DATASET,
        type="dev",
        fs=FS,
        signal_length=SIGNAL_LENGTH,
    )
    dataloader_val = DataLoader(
        dataset_val,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        generator=g,
    )

    epoch = 0

    while epoch < EPOCHS:
        running_loss = 0
        model.train()
        with tqdm(dataloader_train, unit="batch") as tepoch:
            for batch in tepoch:
                tepoch.set_description(f"Epoch {epoch}")

                noisy_signal = batch["noisy"].to(device)
                target_signal = batch["clean"].to(device)

                optimizer.zero_grad()

                enhanced_signal, enhanced_signal_fft = model(noisy_signal)
                target_signal = target_signal[..., : enhanced_signal.shape[-1]]

                target_signal_fft = F.conv1d(
                    target_signal.unsqueeze(1),
                    model.filterbank.auditory_filters_real.to(target_signal.device),
                    stride=model.filterbank.auditory_filters_stride,
                ) + 1j * F.conv1d(
                    target_signal.unsqueeze(1),
                    model.filterbank.auditory_filters_imag.to(target_signal.device),
                    stride=model.filterbank.auditory_filters_stride,
                )

                if KAPPA_BETA is not None:
                    filterbank_weights = model.filterbank.encoder.weight.squeeze(1)
                else:
                    filterbank_weights = None

                loss, loss_ = loss_fn(
                    enhanced_signal_fft, target_signal_fft,
                    filterbank_weights
                )

                running_loss += loss.item()
                if loss_ is not None:
                    loss_.backward()
                else:
                    loss.backward()

                optimizer.step()

                tepoch.set_postfix(loss=loss.item())

        model.eval()
        if epoch % VAL_EVERY == 0:
            running_val_loss = 0
            pesq = 0.0
            sisdr = 0.0

            with torch.no_grad():

                with tqdm(dataloader_val, unit="batch") as tepoch:
                    for batch in tepoch:
                        tepoch.set_description(f"Validation epoch {epoch}")

                        noisy_signal = batch["noisy"].to(device)
                        target_signal = batch["clean"].to(device)

                        enhanced_signal, enhanced_signal_fft = model(noisy_signal)
                        # enhanced signal and target signal should be same length
                        target_signal = target_signal[..., : enhanced_signal.shape[-1]]

                        target_signal_fft = F.conv1d(
                            target_signal.unsqueeze(1),
                            model.filterbank.auditory_filters_real.to(target_signal.device),
                            stride=model.filterbank.auditory_filters_stride,
                        ) + 1j * F.conv1d(
                            target_signal.unsqueeze(1),
                            model.filterbank.auditory_filters_imag.to(target_signal.device),
                            stride=model.filterbank.auditory_filters_stride,
                        )

                        if KAPPA_BETA is not None:
                            filterbank_weights = model.filterbank.encoder.weight.squeeze(1)
                        else:
                            filterbank_weights = None

                        loss, loss_ = loss_fn(
                            enhanced_signal_fft,
                            target_signal_fft,
                            filterbank_weights,
                        )

                        running_val_loss += loss.item()
                        pesq_loop = np.mean(
                            pesq_batch(
                                FS,
                                np.array(target_signal.cpu().detach().numpy()),
                                np.array(enhanced_signal.cpu().detach().numpy()),
                                "nb",
                            )
                        ).item()
                        pesq += pesq_loop
                        sisdr_loop = ScaleInvariantSignalDistortionRatio()(
                            enhanced_signal.cpu().detach(), target_signal.cpu().detach()
                        )
                        sisdr += sisdr_loop

                        tepoch.set_postfix(
                            loss=loss.item(), PESQ=pesq_loop, SISDR=sisdr_loop
                        )

            writer.add_audio(
                "Prediction Val", enhanced_signal[0], epoch, sample_rate=FS
            )
            writer.add_audio("Target Val", target_signal[0], epoch, sample_rate=FS)
            writer.add_audio("Noisy Val", noisy_signal[0], epoch, sample_rate=FS)

            writer.add_scalar("PESQ Val", pesq / len(dataloader_val), epoch)
            writer.add_scalar("SISDR Val", sisdr / len(dataloader_val), epoch)

            writer.add_scalar(
                "Condition Number", model.filterbank.condition_number, epoch
            )

            writer.add_scalars(
                "Loss",
                {
                    "Train": running_loss / len(dataloader_train),
                    "Val": running_val_loss / len(dataloader_val),
                },
                epoch,
            )
        else:
            writer.add_scalars(
                "Loss",
                {
                    "Train": running_loss / len(dataloader_train),
                },
                epoch,
            )

        # check if model save dir exists
        if not os.path.exists(f"{LOGGING_DIR}/models"):
            os.makedirs(f"{LOGGING_DIR}/models")

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            f"{LOGGING_DIR}/models/model_{epoch}.pth",
        )

        epoch += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epochs", type=int, default=301, help="Number of epochs (default: 301)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size (default: 32)"
    )
    parser.add_argument(
        "--val_every",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of workers (default: 4)"
    )
    parser.add_argument(
        "--fs", type=int, default=16000, help="Sampling rate (default: 16000)"
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default=f"{os.getcwd()}/logs/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}",
        help="Logging directory (default: ./logs/<timestamp>)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="datasets/chime2_wsj0",
        help="Path to dataset",
    )
    parser.add_argument(
        "--signal_length",
        type=int,
        default=5,
        help="Signal length in seconds (default: 5)",
    )
    parser.add_argument(
        "--kappa_beta",
        type=float,
        default=None,
        help="Kappa Beta (if None, no kappa beta loss is used, default:  None)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate of the optimizer.",
    )

    main(parser.parse_args())
