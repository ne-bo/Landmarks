import gc

import datetime
import torchvision.models as models
import torch
from torch.autograd import Variable

import landmark_loader
from utils import L2Normalization
import torch.nn as nn
import numpy as np
from sklearn.decomposition import PCA


def learn_PCA_matrix_for_spocs(spocs, desired_dimension):
    print('spocs in learn PCA ', spocs.shape)
    U, S, V = torch.svd(torch.t(spocs))
    print('U.shape ', U.shape)
    print('S.shape ', S.shape)
    return U[:, :desired_dimension], S[:desired_dimension]


def learn_PCA_matrix_for_spocs_with_sklearn(spocs, desired_dimension):
    print('spocs in learn PCA ', spocs.shape)
    pca = PCA(n_components=desired_dimension)
    pca.fit(spocs.cpu().numpy())
    return pca


# outputs is a Tensor with the shape batch_size x 512 x 37 x 37
# we should return the Tensor of size batch_size x 256
def compute_spoc_by_outputs(outputs):
    batch_size = outputs.size(0)
    desired_representation_length = outputs.size(1)
    # sum pooling
    sum_pooled = torch.sum(outputs.view(batch_size, desired_representation_length, -1), dim=2)
    # L2 - normalization
    normalization = L2Normalization()
    spocs = normalization(sum_pooled)
    return spocs


def save_all_spocs_and_labels(loader, network, file_spoc, file_labels):
    all_spocs = torch.cuda.FloatTensor()
    all_labels = torch.LongTensor()
    progress = 0
    for data in loader:
        progress = progress + 1
        if progress % 1000 == 0:
            print('progress ', progress, ' ', datetime.datetime.now())
        if progress > 0:
            images, labels = data
            outputs = network(Variable(images).cuda())
            all_spocs, all_labels = compute_spocs(all_labels, all_spocs, file_labels, file_spoc, labels, outputs, progress)


def compute_spocs(all_labels, all_spocs, file_labels, file_spoc, labels, outputs, progress):
    spocs = compute_spoc_by_outputs(outputs)
    all_spocs = torch.cat((all_spocs, spocs.data), dim=0)
    all_labels = torch.cat((all_labels, labels), dim=0)
    if progress % 1000 == 0:
        print('progress ', progress, ' ', datetime.datetime.now())
        print('labels in batch ', labels.numpy())
        # print('spocs ', spocs)
        print('all_spocs ', all_spocs)
    if progress % 2000 == 0: # labels in batch  [1089995 1089996 1089997 1089998 1089999]
        print('all_spocs', all_spocs)
        print('all_labels', all_labels)
        torch.save(all_spocs, '%s-%d' % (file_spoc, progress))
        torch.save(all_labels, '%s-%d' % (file_labels, progress))
        # flush big tensors after intermediate saving
        all_spocs = torch.cuda.FloatTensor()
        all_labels = torch.LongTensor()
        gc.collect()
    return all_spocs, all_labels


def read_spocs_and_labels(file_spoc, file_labels):
    all_spocs = torch.cuda.FloatTensor()
    all_labels = torch.LongTensor()
    if 'train' in file_spoc:
        last = 218000
        total = 110
        remaining = 752
    else:
        last = 22000
        total = 12
        remaining = 1196

    i = 0
    for k in range(total):
        if i < last:
            i = i + 2000
        else:
            i = i + remaining
        spocs = torch.load(file_spoc + ('-%d' % i))
        labels = torch.load(file_labels + ('-%d' % i))
        all_spocs = torch.cat((all_spocs, spocs), dim=0)
        all_labels = torch.cat((all_labels, labels), dim=0)
        print('all_spocs', all_spocs.shape)
        #print('all_labels', all_labels)
    return all_spocs, all_labels


def get_spoc():
    ########################################
    #
    # Load data
    #
    ########################################
    train_loader, test_loader = landmark_loader.download_landmark(data_folder='/media/natasha/Data/Landmark Kaggle/')

    ########################################
    #
    # Compute spocs and save them
    #
    ########################################

    # this magic code allows us to take the network up to the specific layer even if this layer has no it's own name
    # here 29 is a number of the desired level in the initial pretrained network
    vgg = models.vgg16(pretrained=True)
    print('full vgg ', vgg)
    representation_network = nn.Sequential(*list(vgg.features.children())[:29]).cuda()
    vgg = None
    gc.collect()

    # should be batch_size x 512 x 37 x 37
    print('next(representation_network ', representation_network)
    #save_all_spocs_and_labels(train_loader, representation_network,
    #                                                              'all_spocs_file_train', 'all_labels_file_train')

    save_all_spocs_and_labels(test_loader, representation_network,
                                                                'all_spocs_file_test', 'all_labels_file_test')

    all_spocs_train, all_labels_train = read_spocs_and_labels('all_spocs_file_train', 'all_labels_file_train')
    all_spocs_test, all_labels_test = read_spocs_and_labels('all_spocs_file_test', 'all_labels_file_test')

    print('all_spocs_train ', all_spocs_train)
    print('all_labels_train ', all_labels_train)
    print('all_spocs_test ', all_spocs_test)
    print('all_labels_test ', all_labels_test)

    ########################################
    #
    # Learn PCA
    #
    ########################################

    # PCA
    #PCA_matrix, singular_values = learn_PCA_matrix_for_spocs(all_spocs_train, 256)
    #torch.save(PCA_matrix, 'PCA_matrix')
    #torch.save(singular_values, 'singular_values')


    pca = learn_PCA_matrix_for_spocs_with_sklearn(all_spocs_train, 256)

    ########################################
    #
    # Reduce dimensionality and normalize
    #
    ########################################

    #all_spocs_train = torch.div(torch.mm(all_spocs_train, PCA_matrix), singular_values)
    #all_spocs_test = torch.div(torch.mm(all_spocs_test, PCA_matrix), singular_values)

    all_spocs_train = pca.transform(all_spocs_train.cpu().numpy())
    all_spocs_test = pca.transform(all_spocs_test.cpu().numpy())
    print('all_spocs_train_after_pca', all_spocs_train)
    print('all_spocs_train_after_pca', all_spocs_train.shape)

    all_spocs_train = torch.from_numpy(all_spocs_train).cuda()
    all_spocs_test = torch.from_numpy(all_spocs_test).cuda()
    print('all_spocs_train_after_pca', all_spocs_train)
    print('all_spocs_train_after_pca', all_spocs_train.shape)


    # L2 - normalization
    normalization = L2Normalization()
    all_spocs_train = normalization(Variable(all_spocs_train)).data # https://yadi.sk/d/WY1cwaI83TGypw
    all_spocs_test = normalization(Variable(all_spocs_test)).data # https://yadi.sk/d/Tgz4XdEk3TH2eN

    torch.save(all_spocs_train, 'new_all_spocs_file_train_after_pca')
    torch.save(all_spocs_test, 'new_all_spocs_file_test_after_pca')


get_spoc()
