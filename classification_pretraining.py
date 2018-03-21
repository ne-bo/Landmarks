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


def test_for_classification(loader, network):
    correct = 0
    total = 0
    for data in loader:
        images, labels = data

        outputs = network(Variable(images).cuda())
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted.cpu() == labels).sum()
    accuracy = (100 * correct / total)

    print('Accuracy of the network on the ', total, ' images: %d %%' % accuracy)
    return accuracy


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

            # wrap them in Variable
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = network(inputs)

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

        #if epoch % 10 == 0:
            # print the train accuracy at every epoch
            # to see if it is enough to start representation training
            # or we should proceed with classification
            #accuracy = test_for_classification(
            #    loader=train_loader,
            #    network=network
            #)

        utils.save_checkpoint(
            network=network,
            optimizer=optimizer,
            filename=name_prefix_for_saved_model + '-%d' % epoch,
            epoch=epoch
        )

        total_iteration = total_iteration + i
        print('total_iteration = ', total_iteration)

    print('Finished Training')


def do_pretrainig(
        network,
        train_loader_for_classification,
        learning_rate_for_classification,
        learning_rate_decay_epoch,
        learning_rate_decay_coefficient,
        number_of_epochs
):
    name_prefix_for_saved_model_for_classification = 'classification-'
    optimizer, exp_lr_scheduler = create_optimizer_and_lr_scheduler(
        learning_rate_decay_coefficient,
        learning_rate_decay_epoch,
        learning_rate_for_classification,
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
        name_prefix_for_saved_model_for_classification
    )

    ##################################################################
    #
    # Classification for pre-training
    #
    ##################################################################
    print('Start classification pretraining')
    learning_process(
        train_loader=train_loader_for_classification,
        network=network,
        criterion=nn.CrossEntropyLoss(),
        optimizer=optimizer,
        start_epoch=start_epoch,
        lr_scheduler=exp_lr_scheduler,
        name_prefix_for_saved_model=name_prefix_for_saved_model_for_classification,
        number_of_epochs=number_of_epochs
    )


def pretrain_on_classification(num_classes):
    ########################################
    #
    # Load data
    #
    ########################################

    train_loader, test_loader = landmark_loader_for_classification.download_landmark_for_classification(
        data_folder='../Landmark-classification-resized-256/'
    )
    ########################################
    #
    # Create a network
    #
    #######################################
    print('Create a network ')
    network = models.resnet50(pretrained=True).cuda()

    num_ftrs = network.fc.in_features
    network.fc = torch.nn.Sequential()
    network.fc.add_module('fc', nn.Linear(num_ftrs, num_classes))
    network = network.cuda()
    print(network)
    ########################################
    #
    # Do pretraining
    #
    ########################################
    do_pretrainig(
        network=network,
        train_loader_for_classification=train_loader,
        learning_rate_for_classification=0.01,
        learning_rate_decay_epoch=10,
        learning_rate_decay_coefficient=0.7,
        number_of_epochs=31
    )


pretrain_on_classification(num_classes=14951)
