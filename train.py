#!/usr/bin/env python3

import datetime
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import hydra
import torch
from hydra.core.config_store import ConfigStore
from hydra.utils import get_original_cwd
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter

from datasets.cc_datasets import create_dataset
from engine import evaluate, train_one_epoch
from models import MCCFormer

logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    dataset: str = "rcc_dataset"

    image_size: int = 14
    feature_dim: int = 1024

    batch_size: int = 128
    num_workers: int = 4


@dataclass
class ModelConfig:
    encoder_type: str = "D"  # D or S
    encoder_dim: int = 512
    encoder_nhead: int = 4
    encoder_transformer_layer_num: int = 2

    decoder_nhead: int = 4
    decoder_transformer_layer_num: int = 2
    pe_type: str = "fully_learnable"


@dataclass
class OptimConfig:
    # parameter of optimizer
    lr: float = 0.0001
    momentum: float = 0.9
    beta1: float = 0.9
    beta2: float = 0.99

    # parameter of training steps
    print_freq: int = 100
    snapshot_interval: int = 10
    epochs: int = 40


@dataclass
class Config:
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    optim: OptimConfig = OptimConfig()

    test_only: bool = False
    resume: Optional[str] = None


cs = ConfigStore.instance()
cs.store(name="config", node=Config)


@hydra.main(config_name="config")
def main(cfg):
    logger.info(OmegaConf.to_yaml(cfg))

    # set device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Data loading code
    logger.info("Creating datasets and data loaders")
    dataset, loader = create_dataset(
        Path(get_original_cwd()).joinpath("data"),
        "train",
        batch_size=cfg.data.batch_size,
        seq_per_img=1,
        num_workers=cfg.data.num_workers,
    )
    val_dataset, val_loader = create_dataset(
        Path(get_original_cwd()).joinpath("data"),
        "val",
        batch_size=cfg.data.batch_size,
        seq_per_img=1,
        num_workers=cfg.data.num_workers,
    )

    logger.info("Creating model")
    num_tokens = dataset.vocab_size
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
        max_len=dataset.max_seq_length,
    )
    model.to(device)
    model_without_multi_gpu = model

    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model)

    params = [p for p in model_without_multi_gpu.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=cfg.optim.lr, betas=(cfg.optim.beta1, cfg.optim.beta2))

    if cfg.resume is not None:
        checkpoint = torch.load(Path(get_original_cwd()).joinpath(cfg.resume), map_location="cpu")
        model_without_multi_gpu.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"] + 1
    else:
        start_epoch = 1

    if cfg.test_only:
        evaluate(
            model,
            val_loader,
            device,
            cfg.optim.epochs,
            cfg.optim.print_freq,
            logger=logger,
        )
        return

    # set tensorboard summary writer
    writer = SummaryWriter(log_dir=str(Path.cwd()))

    # start training
    logger.info("Start training")
    start_time = time.time()
    for epoch in range(start_epoch, cfg.optim.epochs + 1):
        # train 1 epoch
        loss_avg = train_one_epoch(
            model, optimizer, loader, device, epoch, cfg.optim.print_freq, logger=logger
        )

        # write the epoch loss
        writer.add_scalar("Loss", loss_avg, epoch)

        # evaluate
        acc_avg = evaluate(model, val_loader, device, epoch, cfg.optim.print_freq, logger=logger)

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
                Path.cwd().joinpath("model_{}.pth".format(epoch)),
            )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info("Training time {}".format(total_time_str))


if __name__ == "__main__":
    main()
