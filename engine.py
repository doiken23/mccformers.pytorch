import datetime
import math
import sys
import time

import torch

import utils


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, logger=None):
    model.train()
    header = "[Train] Epoch: [{}]".format(epoch)
    total_loss = 0

    start_time = time.time()
    for i, data in enumerate(data_loader, 1):
        d_feature, n_feature, q_feature, target, neg_target, _, _, _, _, _, _, _ = data
        d_feature = d_feature.to(device)
        n_feature = n_feature.to(device)
        q_feature = q_feature.to(device)
        target = target.squeeze(1).to(device)
        neg_target = neg_target.squeeze(1).to(device)

        # positive pairs
        loss = model(d_feature, q_feature, target).mean()

        loss_value = loss.item()
        total_loss += loss_value

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_value)
            sys.exit(1)

        # optimizer
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # negative pairs
        loss = model(d_feature, n_feature, neg_target).mean()
        loss_value += loss.item()
        total_loss += loss_value

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_value)
            sys.exit(1)

        # optimizer
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # log the iteration losses
        if logger is not None and (i % print_freq == 0 or i == len(data_loader)):
            logger.info("{} (iter {} / {})".format(header, i, len(data_loader)))
            logger.info("{} loss: {}".format(header, loss_value))

    # total epoch time
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    if logger is not None:
        logger.info(
            "{} Total time: {} ({:.4f} s /it)".format(
                header, total_time_str, total_time / len(data_loader)
            )
        )

    # average loss
    loss_avg = total_loss / len(data_loader)
    if logger is not None:
        logger.info("{} Loss average: {:.4f}".format(header, loss_avg))

    return loss_avg


@torch.no_grad()
def evaluate(model, data_loader, device, epoch, print_freq, logger=None):
    model.eval()
    header = "[Evaluate] Epoch: [{}]".format(epoch)
    total_acc = 0

    start_time = time.time()
    for i, data in enumerate(data_loader, 1):
        d_feature, n_feature, q_feature, target, neg_target, _, _, _, _, _, _, _ = data
        d_feature = d_feature.to(device)
        n_feature = n_feature.to(device)
        q_feature = q_feature.to(device)
        target = target.squeeze(1).to(device)
        neg_target = neg_target.squeeze(1).to(device)

        preds = model(d_feature, q_feature, target)

        acc = utils.compute_accuracy(preds, target[:, 1:])
        total_acc += acc

        # log the iteration losses
        if logger is not None and (i % print_freq == 0 or i == len(data_loader)):
            logger.info("{} (iter {} / {})".format(header, i, len(data_loader)))
            logger.info("{} accuracy: {:.2f} %".format(header, acc * 100))

    # total epoch time
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    if logger is not None:
        logger.info(
            "{} Total time: {} ({:.4f} s /it)".format(
                header, total_time_str, total_time / len(data_loader)
            )
        )

    # average accuracy
    acc_avg = total_acc / len(data_loader)
    if logger is not None:
        logger.info("{} Accuracy average: {:.2f}".format(header, acc_avg))

    return acc_avg
