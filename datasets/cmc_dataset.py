import json
import random
import string
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

random.seed(23)

# create id list
_ids = list(range(60000))
random.shuffle(_ids)
_id_dict = {"train": _ids[:40000], "val": _ids[40000:50000], "test": _ids[50000:60000]}


def _process_sentence(sentence: str) -> list[str]:
    """
    Sentence is processed as follows:
        - converted into lower cases
        - removed punctuations
        - tokenized
    """
    return sentence.lower().translate(str.maketrans("", "", string.punctuation)).split()


class CLEVRMultiChangeDataset(Dataset):
    def __init__(
        self,
        root: str = "./data/clevr/",
        split: str = "train",
        use_feature: bool = False,
        choose_one_caption: bool = True,
        img_transform: Optional[object] = None,
        target_transform: Optional[object] = None,
    ) -> None:

        # set paths
        self.root = Path(root)
        self.image_dir = self.root.joinpath("images")
        self.feature_dir = self.root.joinpath("features")
        self.annotation_dir = self.root.joinpath("multiple_change_caption.json")

        # prepare annotations
        with self.annotation_dir.open("r") as f:
            self.anns = json.load(f)

        # train / val split
        assert split in ["train", "val", "test"]
        self.ids = _id_dict[split]

        self.use_feature = use_feature
        self.choose_one_caption = choose_one_caption

        # transforms
        self.img_transform = img_transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(
        self, idx: int
    ) -> tuple[
        Union[Image.Image, np.ndarray, torch.Tensor],
        Union[Image.Image, np.ndarray, torch.Tensor],
        Union[list[str], list[list[str]], torch.Tensor],
    ]:

        idx = self.ids[idx]

        # image
        before_img, after_img = self._get_images(idx)
        if self.img_transform is not None:
            before_img = self.img_transform(before_img)
            after_img = self.img_transform(after_img)

        target = self._get_captions(idx)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return before_img, after_img, target

    def _get_images(self, idx: int) -> tuple[Image.Image, Image.Image]:
        image_name = self.anns[idx]["image_id"]

        if self.use_feature:
            before_path = self.feature_dir.joinpath(image_name.replace(".png", ".npy"))
            if not before_path.exists():
                before_path = self.feature_dir.joinpath(image_name + ".npy")
            after_path = self.feature_dir.joinpath(
                image_name.replace("t1", "t2").replace(".png", ".npy")
            )
            if not after_path.exists():
                after_path = self.feature_dir.joinpath(image_name.replace("t1", "t2") + ".npy")
            before_img = torch.from_numpy(np.load(before_path))
            after_img = torch.from_numpy(np.load(after_path))
        else:
            before_path = self.image_dir.joinpath(image_name)
            after_path = self.image_dir.joinpath(image_name.replace("t1", "t2"))

            before_img = Image.open(before_path).convert("RGB")
            after_img = Image.open(after_path).convert("RGB")

        return before_img, after_img

    def _get_captions(self, idx: int) -> Union[list[str], list[list[str]]]:
        annotation = self.anns[idx]
        if self.choose_one_caption:
            return self._concatenate_captions(
                [_process_sentence(c) for c in random.choice(annotation["change_captions"])]
            )
        else:
            return [
                self._concatenate_captions([_process_sentence(c) for c in caption])
                for caption in annotation["change_captions"]
            ]

    def _concatenate_captions(self, captions: list[list[str]]) -> list[str]:
        """

        Args:
            captions (list[list[str]]): list of tokens

        Returns:
            out_captions (str): concatenated captions

        """
        out_caption = ["<BOS>"]
        for i, caption in enumerate(captions):
            out_caption += caption
            if i < len(captions) - 1:
                out_caption.append("<SEP>")
            else:
                out_caption.append("<EOS>")
        return out_caption
