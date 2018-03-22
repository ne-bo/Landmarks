import sys
sys.path.append('..')
import script
import numpy as np

import torch
sys.path.append('../../faiss/python/')
sys.path.append('../../faiss/')
import faiss 


class NeighboursFinder(object):
    def __init__(self, train_spocs, using_gpu=False, res=None):
        self.db_size, self.dim = train_spocs.shape # database size, dimension
        n_queries = train_spocs.shape[0]  # nb of queries

        if using_gpu and res is not None:
            self.index = faiss.GpuIndexFlatL2(res, self.dim)  # build the index
        else:
            self.index = faiss.IndexFlatL2(self.dim)     # build the index

        assert self.index.is_trained, "index is not trained"
        self.index.add(train_spocs)
        assert self.index.ntotal == self.db_size, "index.ntotal != db_size"
        print("Initialized. Ready for work!")

    def get_neighbours(self, query_spocs, k=100):
        D, I = self.index.search(query_spocs, k)
        assert I.shape == (query_spocs.shape[0], k), "Shape of answer should be %s, but it %s" % ((query_spocs.shape[0], k), I.shape)

        return I

def get_neighbors(neighbors, existing_train_keys, existing_test_keys, test_key):
    index = np.where(existing_test_keys.__eq__(str(test_key)))[0][0]
    neighbors_indices = np.array(neighbors[index])
    return existing_train_keys[neighbors_indices]


def get_dummy_neighbors():
    result = []
    for i in range(100):
        result.append('b6055a3d08503c42')
    return result

class SubmissionWritter(object):
    def __init__(self):
        with open('../keys_for_test', "r") as fin:
            line = fin.readline()

        self.existing_test_keys = np.array(list(map(str, line.split())))

        with open('../keys_for_train', "r") as fin:
            line = fin.readline()

        self.existing_train_keys = np.array(list(map(str, line.split())))

        for i, key in enumerate(self.existing_test_keys):
            self.existing_test_keys[i] = self.existing_test_keys[i].replace('.jpg', '')

        for i, key in enumerate(self.existing_train_keys):
            self.existing_train_keys[i] = self.existing_train_keys[i].replace('.jpg', '')

        print('existing_test_keys ', self.existing_test_keys)

        keys_urls = script.ParseData('test.csv')
        self.all_test_keys = []
        for i, pair in enumerate(keys_urls):
            self.all_test_keys.append(pair[0])

        print('all_test_keys ', self.all_test_keys)
        
    def write(self, neighbors, filename):
        # id,images
        # 000088da12d664db,0370c4c856f096e8 766677ab964f4311 e3ae4dcee8133159...
        # etc.
        count_dummy = 0
        with open(filename, "w") as fout:
            fout.write('id,images\n')
            for key in self.all_test_keys:
                fout.write(key + ',')
                if key in self.existing_test_keys:
                    neighbors_list = get_neighbors(neighbors, self.existing_train_keys, self.existing_test_keys, key)
                else:
                    neighbors_list = get_dummy_neighbors()
                    count_dummy = count_dummy + 1
                    print('count_dummy', count_dummy)
                fout.write(" ".join([str(el) for el in neighbors_list]))
                fout.write('\n')

        print('count_dummy', count_dummy)