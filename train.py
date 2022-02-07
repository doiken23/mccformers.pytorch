#!/usr/bin/env python3

import datetime
import json
import logging
import random
import time
from pathlib import Path

import coloredlogs
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter

import utils
from datasets.cc_dataset import create_dataset
from datasets.original_cmc_dataset import CaptionDataset
from engine import evaluate, train_one_epoch
from models import MCCFormer

logger = logging.getLogger(__name__)
coloredlogs.install(level="INFO", logger=logger)

# set random seed
seed = 23  # It is my favorite number and doesn't mean anything in particular.
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


g = torch.Generator()
g.manual_seed(seed)


def main() -> None:
    # load configuration
    cfg = utils.load_configs()
    logger.info(OmegaConf.to_yaml(cfg))

    # prepare output directory
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    OmegaConf.save(cfg, output_dir.joinpath("config.yaml"))

    # set device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Data loading code
    logger.info("Creating datasets and data loaders")
    if cfg.data.dataset == "rcc_dataset":
        dataset, loader = create_dataset(
            Path.cwd().joinpath(cfg.data.path),
            "train",
            batch_size=cfg.data.batch_size,
            seq_per_img=1,
            num_workers=cfg.data.num_workers,
        )
        val_dataset, val_loader = create_dataset(
            Path.cwd().joinpath(cfg.data.path),
            "val",
            batch_size=cfg.data.batch_size,
            seq_per_img=1,
            num_workers=cfg.data.num_workers,
        )
        num_tokens = dataset.vocab_size
        max_seq_length = dataset.max_seq_length
    elif cfg.data.dataset == "original_cmc_dataset":
        dataset = CaptionDataset(
            Path.cwd().joinpath(cfg.data.path),
            cfg.data.data_name,
            "TRAIN",
            cfg.data.captions_per_image,
            "MOSCC",
        )
        val_dataset = CaptionDataset(
            Path.cwd().joinpath(cfg.data.path),
            cfg.data.data_name,
            "TEST",
            cfg.data.captions_per_image,
            "MOSCC",
        )

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=cfg.data.batch_size,
            shuffle=True,
            num_workers=cfg.data.num_workers,
            drop_last=True,
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=cfg.data.batch_size,
            shuffle=False,
            num_workers=cfg.data.num_workers,
            drop_last=False,
        )

        with Path.cwd().joinpath(cfg.data.path, "WORDMAP_{}.json".format(cfg.data.data_name)).open(
            "r"
        ) as f:
            vocab = json.load(f)
        num_tokens = len(vocab)
        max_seq_length = 99

    logger.info("Creating model")
    model = MCCFormer(
        encoder_type=cfg.model.encoder_type,
        num_tokens=num_tokens,
        feature_dim=cfg.data.feature_dim,
        encoder_dim=cfg.model.encoder_dim,
        encoder_nhead=cfg.model.encoder_nhead,
        encoder_transformer_layer_num=cfg.model.encoder_transformer_layer_num,
        decoder_nhead=cfg.model.decoder_nhead,
        decoder_transformer_layer_num=cfg.model.decoder_transformer_layer_num,
        pe_type=cfg.model.pe_type,
        max_len=max_seq_length,
    )
    model.to(device)
    model_without_multi_gpu = model

    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model)

    params = [p for p in model_without_multi_gpu.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=cfg.optim.lr, betas=(cfg.optim.beta1, cfg.optim.beta2))

    if cfg.resume is not None:
        checkpoint = torch.load(Path.cwd().joinpath(cfg.resume), map_location="cpu")
        model_without_multi_gpu.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"] + 1
    else:
        start_epoch = 1

    if cfg.test_only:
        evaluate(
            model,
            cfg.data.dataset,
            val_loader,
            device,
            cfg.optim.epochs,
            cfg.optim.print_freq,
            logger=logger,
        )
        return

    # set tensorboard summary writer
    writer = SummaryWriter(log_dir=str(output_dir))

    # start training
    logger.info("Start training")
    start_time = time.time()
    for epoch in range(start_epoch, cfg.optim.epochs + 1):
        # train 1 epoch
        loss_avg = train_one_epoch(
            model,
            optimizer,
            cfg.data.dataset,
            loader,
            device,
            epoch,
            cfg.optim.print_freq,
            logger=logger,
        )

        # write the epoch loss
        writer.add_scalar("Loss", loss_avg, epoch)

        # evaluate
        acc_avg = evaluate(
            model, cfg.data.dataset, val_loader, device, epoch, cfg.optim.print_freq, logger=logger
        )

        # write the epoch accuracy
        writer.add_scalar("Accuracy", acc_avg, epoch)

        # save snapshot
        if epoch % cfg.optim.snapshot_interval == 0:
            # write the epoch losses
            torch.save(
                {
                    "model": model_without_multi_gpu.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "cfg": cfg,
                    "epoch": epoch,
                },
                output_dir.joinpath("model_{}.pth".format(epoch)),
            )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info("Training time {}".format(total_time_str))


if __name__ == "__main__":
    main()
