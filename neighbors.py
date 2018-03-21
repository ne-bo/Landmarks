# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD+Patents license found in the
# LICENSE file in the root directory of this source tree.
import datetime

import numpy as np
import torch

all_spocs_train = torch.load('new_all_resnetvecs_file_train_after_pca')
all_spocs_test = torch.load('new_all_resnetvecs_file_test_after_pca')

print('all_spocs_train_after_pca', all_spocs_train.shape)
print('all_spocs_test_after_pca', all_spocs_test.shape)

d = 1024                         # dimension
nb = 1093759                      # database size
nq = 115977                       # nb of queries
np.random.seed(1234)             # make reproducible
xb = all_spocs_train.cpu().numpy().astype('float32')
xq = all_spocs_test.cpu().numpy().astype('float32')

import sys

sys.path.append('/home/faiss/python/')
sys.path.append('/home/faiss/')
import faiss                  # make faiss available


index = faiss.IndexFlatL2(d)   # build the index
print(index.is_trained)
index.add(xb)                  # add vectors to the index
print(index.ntotal)

k = 100                          # we want to see 100 nearest neighbors
#D, I = index.search(xb[:5], k) # sanity check
#print('I and D')
#print(I)
#print(D)

D, I = index.search(xq, k)     # actual search
#print('neighbors of the 5 first queries')
#print(I[:5])                   # neighbors of the 5 first queries
#print('neighbors of the 5 last queries')
#print(I[-5:])                  # neighbors of the 5 last queries

print('I.shape ', I.shape)
np.save('100_nearest_neighbors_resnet', I)

I = np.load('/media/natasha/Data/Landmark Kaggle/100_nearest_neighbors_resnet.npy')
print('neighbors', I)


def get_X_d(list_1, list_2, d):
    return float(np.intersect1d(list_1[:d], list_2[:d]).shape[0])


def rank_biased_overlap_similarity(list_1, list_2, p, h):
    result = 0.0
    for d in range(1, h + 1):
        # print('np.power(p, d - 1) ', np.power(p, d - 1))
        Xd = get_X_d(list_1, list_2, d)
        #if Xd > 0.0:
        #    print('Xd ', Xd)
        result = result + np.power(p, d - 1) * Xd / float(d)
    return (p - 1.0) * result


def consistent_reranking(index, xb, xq, k, I):
    p = 0.9
    h = 10
    new_I = np.zeros_like(I)
    for i, query in enumerate(xq):
        if i % 1000 == 0:
            print('i = ', i, ' consistent_reranking ', datetime.datetime.now())
        indices_of_neighbors = I[i]
        neighbors = xb[indices_of_neighbors]
        D_neighbors, I_neighbors = index.search(neighbors, k)
        # need to comapre each line in I_neighbors with actual query list indices_of_neighbors
        rank = []
        for list_for_neighbor in I_neighbors:
            rank.append(rank_biased_overlap_similarity(indices_of_neighbors, list_for_neighbor, p, h))
        rank = np.array(rank)
        indices_for_reranking = np.argsort(rank)
        new_I[i] = I[i, indices_for_reranking]
    return new_I


faiss.omp_set_num_threads (6)

new_I = consistent_reranking(index, xb, xq, k, I)
print('new_I.shape ', new_I.shape)
np.save('reranked_100_nearest_neighbors', new_I)


def replace_queries_with_centriods(xb, xq, new_I, top=10):
    centroids = np.zeros_like(xq)
    for i, query in enumerate(xq):
        indices_of_top_neighbors = new_I[i, :top]
        neighbors = xb[indices_of_top_neighbors]
        centroids[i] = np.mean(neighbors, axis=0)
    return centroids


centroids = replace_queries_with_centriods(xb, xq, new_I, top=10)
D_centroids, I_centroids = index.search(centroids, k)
print('I_centroids.shape ', I_centroids.shape)
np.save('centroids_of_reranked_100_nearest_neighbors', I_centroids)

