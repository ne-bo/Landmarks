import gc
import utils
import datetime
import torchvision.models as models
import torch
from torch.autograd import Variable
import numpy as np
import landmark_loader
from utils import L2Normalization
import torch.nn as nn


from sklearn.decomposition import IncrementalPCA
from sklearn.externals import joblib


def learn_PCA_matrix_for_resnetvecs(resnetvecs, desired_dimension):
    print('resnetvecs in learn PCA ', resnetvecs.shape)
    U, S, V = torch.svd(torch.t(resnetvecs))
    print('U.shape ', U.shape)
    print('S.shape ', S.shape)
    return U[:, :desired_dimension], S[:desired_dimension]


def learn_PCA_matrix_for_resnetvecs_with_sklearn(resnetvecs, desired_dimension):
    print('resnetvecs in learn PCA ', resnetvecs.shape)
    pca = IncrementalPCA(n_components=desired_dimension, copy=False)
    pca.fit(resnetvecs)
    joblib.dump(pca, ('pca-%d.pkl' % desired_dimension))
    return pca


# outputs is a Tensor with the shape batch_size x 512 x 37 x 37
# we should return the Tensor of size batch_size x 512
def compute_resnetvec_by_outputs(outputs):
    batch_size = outputs.size(0)
    desired_representation_length = outputs.size(1)
    # sum pooling
    sum_pooled = torch.sum(outputs.view(batch_size, desired_representation_length, -1), dim=2)
    # L2 - normalization
    normalization = L2Normalization()
    resnetvecs = normalization(sum_pooled)
    return resnetvecs


def save_all_resnetvecs_and_labels(loader, network, file_resnetvec, file_labels):
    all_resnetvecs = torch.cuda.FloatTensor()
    all_labels = torch.LongTensor()
    progress = 0
    for data in loader:
        progress = progress + 1
        if progress % 1000 == 0:
            print('progress ', progress, ' ', datetime.datetime.now())
        if progress > 10000:
            images, labels = data
            outputs = network(Variable(images).cuda())
            # print('outputs.shape ', outputs.data.shape)
            all_resnetvecs, all_labels = compute_resnetvecs(
                all_labels,
                all_resnetvecs,
                file_labels,
                file_resnetvec,
                labels,
                outputs,
                progress
            )


# len(new_train_dataset.train_images)  1093759
# len(new_test_dataset.test_images)  115977

def compute_resnetvecs(all_labels, all_resnetvecs, file_labels, file_resnetvec, labels, outputs, progress):
    resnetvecs = compute_resnetvec_by_outputs(outputs)
    all_resnetvecs = torch.cat((all_resnetvecs, resnetvecs.data), dim=0)
    all_labels = torch.cat((all_labels, labels), dim=0)
    if progress % 1000 == 0:
        print('progress ', progress, ' ', datetime.datetime.now())
        print('labels in batch ', labels.numpy())
        # print('resnetvecs ', resnetvecs)
        print('all_resnetvecs ', all_resnetvecs)
    if progress % 2000 == 0 or 115976 in labels:  # labels in batch  [1089995 1089996 1089997 1089998 1089999]
        print('all_resnetvecs', torch.mean(all_resnetvecs), ' ', datetime.datetime.now())
        print('all_labels', all_labels)
        torch.save(all_resnetvecs, '%s-%d' % (file_resnetvec, progress))
        torch.save(all_labels, '%s-%d' % (file_labels, progress))
        # flush big tensors after intermediate saving
        all_resnetvecs = torch.cuda.FloatTensor()
        all_labels = torch.LongTensor()
        gc.collect()
    return all_resnetvecs, all_labels


def read_resnetvecs_and_labels(file_resnetvec, file_labels):
    all_resnetvecs = torch.FloatTensor()
    all_labels = torch.LongTensor()
    if 'train' in file_resnetvec:
        last = 108000
        total = 55
        remaining = 1376
    else:
        last = 10000
        total = 6
        remaining = 1598

    i = 0
    for k in range(total):
        if i < last:
            i = i + 2000
        else:
            print('i = ', i)
            i = i + remaining

        # for 2048-dimensional vectors my 6Gb GPU is not enough
        print('file', file_resnetvec + ('-%d' % i))
        resnetvecs = torch.load(file_resnetvec + ('-%d' % i)).cpu()
        #labels = torch.load(file_labels + ('-%d' % i))
        all_resnetvecs = torch.cat((all_resnetvecs, resnetvecs), dim=0)
        #all_labels = torch.cat((all_labels, labels), dim=0)
        print('all_resnetvecs', all_resnetvecs.shape)
        # print('all_labels', all_labels)
    return all_resnetvecs.numpy(), all_labels


def get_resnetvec():
    ########################################
    #
    # Load data
    #
    ########################################
    train_loader, test_loader = landmark_loader.download_landmark(data_folder='/media/natasha/Data/Landmark Kaggle/')

    ########################################
    #
    # Compute resnetvecs and save them
    #
    ########################################

    if True:
        print('Create a network ')
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

        network, optimizer, start_epoch = utils.restore_from_the_epoch(
            network,
            optimizer,
            restore_epoch=3,
            name_prefix_for_saved_model_for_classification='classification-'
        )

        print('full resnet ', network)

        representation_network = nn.Sequential(*list(network.children())[:8])

        # should be batch_size x 2048 x ??? x ???
        print('next(representation_network ', representation_network)

        network = None
        gc.collect()

        save_all_resnetvecs_and_labels(
            train_loader,
            representation_network,
            'all_resnetvecs_file_train',
            'all_labels_file_train'
        )

        save_all_resnetvecs_and_labels(
            test_loader,
            representation_network,
            'all_resnetvecs_file_test',
            'all_labels_file_test'
        )

    all_resnetvecs_train, all_labels_train = read_resnetvecs_and_labels('all_resnetvecs_file_train',
                                                                        'all_labels_file_train')

    print('all_resnetvecs_train ', all_resnetvecs_train)
    print('all_labels_train ', all_labels_train)
    print('We managed to read it! ', datetime.datetime.now())

    ########################################
    #
    # Learn PCA
    #
    ########################################

    # PCA
    # PCA_matrix, singular_values = learn_PCA_matrix_for_resnetvecs(all_resnetvecs_train, 256)
    # torch.save(PCA_matrix, 'PCA_matrix')
    # torch.save(singular_values, 'singular_values')

    # pca = learn_PCA_matrix_for_resnetvecs_with_sklearn(all_resnetvecs_train, 1024)
    pca = joblib.load('pca-1024.pkl')
    print('We managed to fit PCA! ', datetime.datetime.now())
    ########################################
    #
    # Reduce dimensionality and normalize
    #
    ########################################

    #all_resnetvecs_train = torch.div(torch.mm(all_resnetvecs_train, PCA_matrix), singular_values)
    #all_resnetvecs_test = torch.div(torch.mm(all_resnetvecs_test, PCA_matrix), singular_values)

    # L2 - normalization
    normalization = L2Normalization()

    if True:
        n = all_resnetvecs_train.shape[0]
        batch_size = n // 30
        all_resnetvecs_train_after_pca = pca.transform(all_resnetvecs_train[: batch_size])
        for i in range(1, 30):
            batch = pca.transform(all_resnetvecs_train[i * batch_size: (i + 1) * batch_size])
            all_resnetvecs_train_after_pca = np.vstack((all_resnetvecs_train_after_pca, batch))
            batch = None
        last_batch = pca.transform(all_resnetvecs_train[30 * batch_size:])
        all_resnetvecs_train = np.vstack((all_resnetvecs_train_after_pca, last_batch))
        print('all_resnetvecs_train_after_pca', all_resnetvecs_train.shape)
        last_batch = None
        gc.collect()

        all_resnetvecs_train = torch.from_numpy(all_resnetvecs_train)#.cuda()
        print('all_resnetvecs_train_after_pca', all_resnetvecs_train.shape)
        all_resnetvecs_train = normalization(Variable(all_resnetvecs_train)).data  # https://yadi.sk/d/WY1cwaI83TGypw
        torch.save(all_resnetvecs_train, 'new_all_resnetvecs_file_train_after_pca')
        all_resnetvecs_train = None
        gc.collect()


    all_resnetvecs_test, all_labels_test = read_resnetvecs_and_labels('all_resnetvecs_file_test',
                                                                      'all_labels_file_test')
    print('all_resnetvecs_test ', all_resnetvecs_test)
    print('all_labels_test ', all_labels_test)
    all_resnetvecs_test = pca.transform(all_resnetvecs_test)
    all_resnetvecs_test = torch.from_numpy(all_resnetvecs_test)#.cuda()
    # L2 - normalization
    all_resnetvecs_test = normalization(Variable(all_resnetvecs_test)).data  # https://yadi.sk/d/Tgz4XdEk3TH2eN
    torch.save(all_resnetvecs_test, 'new_all_resnetvecs_file_test_after_pca')


get_resnetvec()
