#!/usr/bin/env python3

import datetime
import json
import logging
import time
from pathlib import Path

import coloredlogs
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

import utils
from datasets.cc_dataset import create_dataset
from datasets.cmc_dataset import CLEVRMultiChangeDataset
from datasets.original_cmc_dataset import CaptionDataset
from models import MCCFormer

logger = logging.getLogger(__name__)


def main() -> None:
    # load configuration
    cfg = utils.load_configs()

    # prepare output directory
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    OmegaConf.save(cfg, output_dir.joinpath("config.yaml"))

    # logger
    utils.create_logger(output_dir)
    logger = logging.getLogger(__name__)
    coloredlogs.install(level="INFO", logger=logger)
    logger.info(OmegaConf.to_yaml(cfg))

    # set device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    cpu_device = torch.device("cpu")

    # Data loading code
    logger.info("Creating datasets and data loaders")
    if cfg.data.dataset == "rcc_dataset":
        test_dataset, test_loader = create_dataset(
            Path.cwd().joinpath("data"),
            "test",
            batch_size=1,
            seq_per_img=1,
            num_workers=cfg.data.num_workers,
        )
        num_tokens = test_dataset.vocab_size
        max_seq_length = test_dataset.max_seq_length

    elif cfg.data.dataset == "original_cmc_dataset":
        max_seq_length = 99
        with Path.cwd().joinpath(cfg.data.path, "WORDMAP_{}.json".format(cfg.data.data_name)).open(
            "r"
        ) as f:
            vocab = json.load(f)
        idx_to_word = {v: k for k, v in vocab.items()}
        num_tokens = len(vocab)

        test_dataset = CaptionDataset(
            Path.cwd().joinpath(cfg.data.path),
            cfg.data.data_name,
            "TEST",
            cfg.data.captions_per_image,
            "MOSCC",
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=cfg.data.num_workers,
            drop_last=False,
        )

    elif cfg.data.dataset == "cmc_dataset":
        max_seq_length = 99
        with Path.cwd().joinpath(cfg.data.vocab_path).open("r") as f:
            vocab = json.load(f)
        target_transform = utils.Word2Id(max_seq_length, vocab)
        idx_to_word = {v: k for k, v in vocab.items()}
        num_tokens = len(vocab)

        test_dataset = CLEVRMultiChangeDataset(
            Path.cwd().joinpath(cfg.data.path),
            split="test",
            use_feature=True,
            choose_one_caption=True,
            img_transform=None,
            target_transform=target_transform,
        )

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=cfg.data.num_workers,
            drop_last=False,
        )
    logger.info("Length of test dataset: {}".format(len(test_dataset)))

    # prepare model
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
    model.eval()

    # load pre-trained model
    if cfg.resume is None:
        exit()
    checkpoint = torch.load(Path.cwd().joinpath(cfg.resume), map_location="cpu")
    model.load_state_dict(checkpoint["model"])

    # predict captions
    result_captions_pos = []
    result_captions_neg = []
    start_time = time.time()
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):
            if cfg.data.dataset == "rcc_dataset":
                (
                    d_feature,
                    n_feature,
                    q_feature,
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
                d_feature = d_feature.to(device)
                n_feature = n_feature.to(device)
                q_feature = q_feature.to(device)

                outputs_pos = model(d_feature, q_feature)
                caption_pos = utils.decode_seq(
                    outputs_pos.squeeze().to(cpu_device).tolist(), test_dataset.idx_to_word
                )
                image_id = d_img_paths[0].split("_")[-1]
                result_captions_pos.append({"caption": caption_pos, "image_id": image_id})

                outputs_neg = model(d_feature, n_feature)
                caption_neg = utils.decode_seq(outputs_neg[0].tolist(), test_dataset.idx_to_word)
                result_captions_neg.append({"caption": caption_neg, "image_id": image_id + "_n"})

            elif cfg.data.dataset == "original_cmc_dataset":
                if i % cfg.data.captions_per_image != 0:
                    continue

                d_feature, q_feature, target, _, _ = data
                d_feature = d_feature.to(device)
                q_feature = q_feature.to(device)

                output = model(d_feature, q_feature, start_idx=vocab["<start>"])
                output = output.squeeze().to(cpu_device).tolist()
                output = utils.decode_seq(
                    output,
                    idx_to_word,
                    start_idx=vocab["<start>"],
                    end_idx=vocab["<end>"],
                    pad_idx=vocab["<pad>"],
                )

                result_captions_pos.append({"caption": output, "image_id": i + 1})

            elif cfg.data.dataset == "cmc_dataset":
                d_feature, q_feature, target = data
                d_feature = d_feature.to(device)
                q_feature = q_feature.to(device)

                output = model(d_feature, q_feature, start_idx=vocab.word2idx["<BOS>"])
                output = output.squeeze().to(cpu_device).tolist()
                output = utils.decode_seq(
                    output,
                    idx_to_word,
                    start_idx=vocab["<start>"],
                    end_idx=vocab["<end>"],
                    pad_idx=vocab["<pad>"],
                )

                result_captions_pos.append({"caption": output, "image_id": i + 1})

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info("Test time {}".format(total_time_str))

    # save results
    with output_dir.joinpath("sc_results.json").open("w") as f:
        json.dump(result_captions_pos, f)
    with output_dir.joinpath("nsc_results.json").open("w") as f:
        json.dump(result_captions_neg, f)


if __name__ == "__main__":
    main()
