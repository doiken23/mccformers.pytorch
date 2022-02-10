#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

from pycocoevalcap.eval import COCOEvalCap
from pycocotools.coco import COCO


def main():
    # get arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("gt_path", type=str)
    parser.add_argument("pred_path", type=str)
    args = parser.parse_args()

    # evaluation
    coco = COCO(args.gt_path)
    coco_result = coco.loadRes(args.pred_path)
    coco_eval = COCOEvalCap(coco, coco_result)
    coco_eval.params["image_id"] = coco_result.getImgIds()
    coco_eval.evaluate()

    # save results
    scores = coco_eval.eval
    output_name = (
        "coco_captioning_scores.json"
        if "w_target" not in args.pred_path
        else "coco_captioning_score_w_target.json"
    )
    with Path(args.pred_path).parent.joinpath(output_name).open("w") as f:
        json.dump(scores, f, indent=2)


if __name__ == "__main__":
    main()
