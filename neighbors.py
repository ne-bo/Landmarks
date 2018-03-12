# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD+Patents license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch

all_spocs_train = torch.load('new_all_spocs_file_train_after_pca')
all_spocs_test = torch.load('new_all_spocs_file_test_after_pca')

print('all_spocs_train_after_pca', all_spocs_train.shape)
print('all_spocs_test_after_pca', all_spocs_test.shape)

d = 256                           # dimension
nb = 1093759                      # database size
nq = 115977                       # nb of queries
np.random.seed(1234)             # make reproducible
xb = all_spocs_train.cpu().numpy()
xq = all_spocs_test.cpu().numpy()

import sys

sys.path.append('/home/faiss/python/')
sys.path.append('/home/faiss/')
import faiss                  # make faiss available


index = faiss.IndexFlatL2(d)   # build the index
print(index.is_trained)
index.add(xb)                  # add vectors to the index
print(index.ntotal)

k = 100                          # we want to see 100 nearest neighbors
D, I = index.search(xb[:5], k) # sanity check
print('I and D')
print(I)
print(D)
D, I = index.search(xq, k)     # actual search
print('neighbors of the 5 first queries')
print(I[:5])                   # neighbors of the 5 first queries
print('neighbors of the 5 last queries')
print(I[-5:])                  # neighbors of the 5 last queries

print('I.shape ', I.shape)
np.save('100_nearest_neighbors', I)