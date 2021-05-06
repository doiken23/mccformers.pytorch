# MCCFormers

Unofficial implementation of MCCFormers proposed in the paper:
- Y. Qiu et al. Describing and Localizing Multiple Changes with Transformers. arXiv, 2021. ([arXiv](https://arxiv.org/abs/2103.14146))

# Requirements

- Python
- Poetry

# Installation

```
git clone git@github.com:doiken/mccformers.pytorch
poetry install
```

# Dataset

1. Download and preprocess CLEVR-Change dataset as described in the README of [Park's repository](https://github.com/Seth-Park/RobustChangeCaptioning).
2. Make a symbolic link from the directory containing the preprocessed dataset to `./data` directory in the repository.

# Training and evaluation

The following commands can be used to train and evaluate the model.

```
poetry run python train.py \
    hydra.run.dir=${OUTPUT_DIR} \
    data.batch_size=${BATCH_SIZE} \
    model.encoder_type=${ENCODER_TYPE}  # encoder type is 'D' or 'S'
```

```
poetry run python test.py \
    hydra.run.dir=${OUTPUT_DIR} \
    resume=${PRETRAINED_MODEL_PATH} \
    model.encoder_type=${ENCODER_TYPE}  # encoder type is 'D' or 'S'
```

Running `test.py` will generate a JSON file that contains the inference results.
By applying the evaluation script provided in the [Park's repository](https://github.com/Seth-Park/RobustChangeCaptioning) to the generated JSON file,
you can measure the performance of caption generation for the trained model.
