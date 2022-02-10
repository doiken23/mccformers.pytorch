#!/usr/bin/env python3

import json
import logging
from pathlib import Path

import coloredlogs
from omegaconf import OmegaConf
from tqdm import tqdm

import utils
from datasets.original_cmc_dataset import CaptionDataset


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

    # Data loading code
    logger.info("Creating datasets and data loaders")
    dataset = CaptionDataset(
        Path.cwd().joinpath(cfg.data.path),
        cfg.data.data_name,
        "TEST",
        cfg.data.captions_per_image,
        "MOSCC",
    )

    # Load word dict
    with Path.cwd().joinpath(
        cfg.data.path, "WORDMAP_3dcc_5_cap_per_img_0_min_word_freq.json"
    ).open("r") as f:
        word_to_idx = json.load(f)
    idx_to_word = {v: k for k, v in word_to_idx.items()}

    # Create COCO Caption format
    # craete coco format
    # we refer the implementation of Park:
    # https://github.com/Seth-Park/RobustChangeCaptioning/blob/master/utils/utils.py
    logger.info("Creating COCO Caption format")
    gt_dict = {}
    info_dict = {
        "contributor": "dummy",
        "date_created": "dummy",
        "description": "dummy",
        "url": "dummy",
        "version": "dummy",
        "year": "dummy",
    }
    gt_dict["info"] = info_dict
    gt_dict["licenses"] = info_dict
    gt_dict["type"] = "captions"
    gt_dict["images"] = []
    gt_dict["annotations"] = []
    for image_idx in tqdm(range(len(dataset) // cfg.data.captions_per_image)):
        gt_dict["images"].append({"filename": "{}.png".format(image_idx + 1), "id": image_idx + 1})

        _, _, _, _, all_captions = dataset[cfg.data.captions_per_image * image_idx]
        all_captions = all_captions.tolist()

        for caption_idx, caption in enumerate(all_captions):
            decoded_caption = utils.decode_seq(
                caption,
                idx_to_word,
                start_idx=word_to_idx["<start>"],
                end_idx=word_to_idx["<end>"],
                pad_idx=word_to_idx["<pad>"],
            )
            gt_dict["annotations"].append(
                {
                    "caption": decoded_caption,
                    "id": cfg.data.captions_per_image * image_idx + caption_idx + 1,
                    "image_id": image_idx + 1,
                }
            )

    logger.info("Creating COCO Caption format is finished")
    logger.info("{} images are registered".format(image_idx + 1))

    with Path.cwd().joinpath(cfg.output_dir, "coco_caption.json").open("w") as f:
        json.dump(gt_dict, f)


if __name__ == "__main__":
    main()
