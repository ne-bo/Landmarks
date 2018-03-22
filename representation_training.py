import datetime

import torch
import visdom
from torch import nn
from torch.autograd import Variable
from torchvision import models

import utils
import landmark_loader_for_classification
import numpy as np

from utils import restore_from_the_epoch, create_optimizer_and_lr_scheduler
from histogramm_loss import HistogramLoss
from margin_loss_for_similarity import MarginLossForSimilarity

def learning_process(
        train_loader,
        network,
        criterion,
        optimizer,
        start_epoch,
        lr_scheduler,
        name_prefix_for_saved_model,
        number_of_epochs
):
    vis = visdom.Visdom()
    r_loss = []
    iterations = []
    total_iteration = 0

    loss_plot = vis.line(Y=np.zeros(1), X=np.zeros(1))

    for epoch in range(start_epoch, number_of_epochs):  # loop over the dataset multiple times
        lr_scheduler.step(epoch=epoch)
        print('current_learning_rate =', optimizer.param_groups[0]['lr'], ' ',datetime.datetime.now())

        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            # print('inputs ', inputs) # 46x3x224x224

            # wrap them in Variable
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = network(inputs)

            # print('outputs ', outputs) # 46x128 = batch_size x desired_dimensionality
            # print('labels ', labels) # 46
            # input()

            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            # print statistics
            current_batch_loss = loss.data[0]
            if i % 1000 == 0:
                print('[epoch %d, iteration in the epoch %5d] loss: %.30f' % (epoch + 1, i + 1, current_batch_loss))

                r_loss.append(current_batch_loss)
                iterations.append(total_iteration + i)

                options = dict(legend=['loss'])
                loss_plot = vis.line(
                    Y=np.array(r_loss),
                    X=np.array(iterations),
                    win=loss_plot,
                    opts=options
                )

        utils.save_checkpoint(
            network=network,
            optimizer=optimizer,
            filename=name_prefix_for_saved_model + '-%d' % epoch,
            epoch=epoch
        )

        total_iteration = total_iteration + i
        print('total_iteration = ', total_iteration)

    print('Finished Training')


def do_trainig(
        network,
        train_loader,
        learning_rate,
        learning_rate_decay_epoch,
        learning_rate_decay_coefficient,
        number_of_epochs
):
    name_prefix_for_saved_model = 'representation-margin-'
    optimizer, exp_lr_scheduler = create_optimizer_and_lr_scheduler(
        learning_rate_decay_coefficient,
        learning_rate_decay_epoch,
        learning_rate,
        network
    )
    ##################################################################
    #
    # Optional recovering from the saved file
    #
    ##################################################################
    restore_epoch = 0
    network, optimizer, start_epoch = restore_from_the_epoch(
        network,
        optimizer,
        restore_epoch,
        name_prefix_for_saved_model
    )

    ##################################################################
    #
    # Representation training
    #
    ##################################################################
    print('Start representation training')
    learning_process(
        train_loader=train_loader,
        network=network,
        #criterion=HistogramLoss(150),
        criterion=MarginLossForSimilarity(),
        optimizer=optimizer,
        start_epoch=start_epoch,
        lr_scheduler=exp_lr_scheduler,
        name_prefix_for_saved_model=name_prefix_for_saved_model,
        number_of_epochs=number_of_epochs
    )


def train_for_representation(desired_dimensionality):
    ########################################
    #
    # Load data
    #
    ########################################

    # we have to use classification data for training because it has labels
    train_loader, test_loader = landmark_loader_for_classification.download_landmark_for_classification(
        data_folder='../Landmark-classification-resized-256/',
        uniform_sampling=True
    )
    ########################################
    #
    # Create a network
    #
    #######################################
    print('Create a network ')
    # create a network as for classification pretraining
    network = models.resnet50(pretrained=True).cuda()

    num_ftrs = network.fc.in_features
    network.fc = torch.nn.Sequential()
    network.fc.add_module('fc', nn.Linear(num_ftrs, 14951))
    network = network.cuda()
    print(network)
    optimizer, exp_lr_scheduler = utils.create_optimizer_and_lr_scheduler(
        learning_rate_decay_coefficient=0.7,
        learning_rate_decay_epoch=10,
        learning_rate_for_classification=0.01,
        network=network
    )
    # restore classification fine-tuned weights
    network, optimizer, start_epoch = utils.restore_from_the_epoch(
        network,
        optimizer,
        restore_epoch=3,
        name_prefix_for_saved_model_for_classification='classification-'
    )
    print('restored after classification network', network)

    # change the dimensionality of the last layer to the desired dimensionality
    num_ftrs = network.fc.fc.in_features
    network.fc = torch.nn.Sequential()
    network.fc.add_module('fc', nn.Linear(num_ftrs, desired_dimensionality))
    network.fc.add_module('l2normalization', utils.L2Normalization())  # need normalization for histogramm loss
    network = network.cuda()
    print(network)

    ########################################
    #
    # Do training
    #
    ########################################
    do_trainig(
        network=network,
        train_loader=train_loader,
        learning_rate=0.01,
        learning_rate_decay_epoch=10,
        learning_rate_decay_coefficient=0.7,
        number_of_epochs=31
    )


train_for_representation(desired_dimensionality=128)
