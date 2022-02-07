#!/usr/bin/env python3

import datetime
import json
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
from tqdm import tqdm

import utils
from datasets.cc_datasets import create_dataset
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
    test_dataset, test_loader = create_dataset(
        Path(get_original_cwd()).joinpath("data"),
        "test",
        batch_size=cfg.data.batch_size,
        seq_per_img=1,
        num_workers=cfg.data.num_workers,
    )

    # prepare model
    logger.info("Creating model")
    num_tokens = test_dataset.vocab_size
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
        max_len=test_dataset.max_seq_length,
    )
    model.to(device)
    model.eval()

    # load pre-trained model
    if cfg.resume is None:
        exit()
    checkpoint = torch.load(Path(get_original_cwd()).joinpath(cfg.resume), map_location="cpu")
    model.load_state_dict(checkpoint["model"])

    # predict captions
    result_captions_pos = []
    result_captions_neg = []
    start_time = time.time()
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader), 1):
            (
                d_features,
                n_features,
                q_features,
                targets,
                neg_targets,
                _,
                _,
                _,
                _,
                d_img_paths,
                ncs_img_paths,
                sc_feats,
            ) = data
            d_features = d_features.to(device)
            n_features = n_features.to(device)
            q_features = q_features.to(device)
            batch_size = len(d_features)

            outputs_pos = model(d_features, q_features)
            outputs_neg = model(d_features, n_features)

            for j in range(batch_size):
                caption_pos = utils.decode_seq(outputs_pos[j].tolist(), test_dataset.idx_to_word)
                caption_neg = utils.decode_seq(outputs_neg[j].tolist(), test_dataset.idx_to_word)
                image_id = d_img_paths[j].split("_")[-1]

                result_captions_pos.append({"caption": caption_pos, "image_id": image_id})
                result_captions_neg.append({"caption": caption_neg, "image_id": image_id + "_n"})
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info("Test time {}".format(total_time_str))

    # save results
    output_dir = Path.cwd().joinpath("results", "test")
    output_dir.mkdir(parents=True, exist_ok=True)
    with output_dir.joinpath("sc_results.json").open("w") as f:
        json.dump(result_captions_pos, f)
    with output_dir.joinpath("nsc_results.json").open("w") as f:
        json.dump(result_captions_neg, f)


if __name__ == "__main__":
    main()
