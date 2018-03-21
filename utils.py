import numpy as np
import torch.nn as nn
import torch

# Lera's implementation
from torch import optim
from torch.optim import lr_scheduler

import script


class L2Normalization(nn.Module):
    def __init__(self):
        super(L2Normalization, self).__init__()

    def forward(self, input):
        input = input.squeeze()
        return input.div(torch.norm(input, dim=1).view(-1, 1))

    def __repr__(self):
        return self.__class__.__name__


def get_all_keys(filename):
    keys_urls = script.ParseData(filename)
    all_keys = []
    for i, pair in enumerate(keys_urls):
        all_keys.append(pair[0])
    return all_keys


def get_existing_keys(filename):
    with open(filename, "r") as fin:
        line = fin.readline()
    existing_keys = np.array(list(map(str, line.split())))
    for i, key in enumerate(existing_keys):
        existing_keys[i] = existing_keys[i].replace('.jpg', '')
    return existing_keys


def get_all_keys_and_labels(filename):
    keys_labels = script.ParseDataWithLabels(filename)
    all_keys = []
    all_labels = []
    for i, pair in enumerate(keys_labels):
        all_keys.append(pair[0])
        all_labels.append(int(pair[2]))
    return all_keys, all_labels


def save_checkpoint(network, optimizer, epoch, filename='checkpoint.pth.tar'):
    torch.save({
        'epoch': epoch + 1,
        'state_dict': network.state_dict(),
        'optimizer': optimizer.state_dict()
    }, filename)


def load_network_and_optimizer_from_checkpoint(network, optimizer, epoch, name_prefix_for_saved_model):
    # optionally resume from a checkpoint
    print("=> loading checkpoint")
    checkpoint = torch.load(name_prefix_for_saved_model + '-%d' % epoch)
    network.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    print("=> loaded checkpoint (epoch {%d})" % epoch)
    return network, optimizer


def load_network_from_checkpoint(network, epoch, name_prefix_for_saved_model, stage=None, loss_function_name=''):
    # optionally resume from a checkpoint
    print("=> loading checkpoint '{}'")
    if stage != None:
        checkpoint = torch.load(name_prefix_for_saved_model + '-%d-%d%s' % (epoch, stage, loss_function_name))
    else:
        checkpoint = torch.load(name_prefix_for_saved_model + '-%d' % epoch)
    network.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{%s}' (epoch {%d}) stage = %d" % (name_prefix_for_saved_model, epoch, stage))
    return network


def restore_from_the_epoch(
        network,
        optimizer,
        restore_epoch,
        name_prefix_for_saved_model_for_classification
):
    if restore_epoch > 0:
        print('Restore for classification pre-training')
        network, optimizer = load_network_and_optimizer_from_checkpoint(
            network=network,
            optimizer=optimizer,
            epoch=restore_epoch,
            name_prefix_for_saved_model=name_prefix_for_saved_model_for_classification
        )
        start_epoch = restore_epoch
    else:
        start_epoch = 0
    return network, optimizer, start_epoch


def create_optimizer_and_lr_scheduler(
        learning_rate_decay_coefficient,
        learning_rate_decay_epoch,
        learning_rate_for_classification,
        network
):
    optimizer = optim.Adam(
        network.parameters(),
        lr=learning_rate_for_classification
    )
    print('Create lr_scheduler')
    # Decay LR by a factor of learning_rate_decay_coefficient every learning_rate_decay_epoch epochs
    exp_lr_scheduler = lr_scheduler.StepLR(
        optimizer,
        step_size=learning_rate_decay_epoch,
        gamma=learning_rate_decay_coefficient
    )
    return optimizer, exp_lr_scheduler