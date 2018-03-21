import numpy as np

from utils import get_all_keys, get_existing_keys

existing_keys = get_existing_keys('keys_for_test')

existing_train_keys = get_existing_keys('keys_for_train')

all_test_keys = get_all_keys('/media/natasha/Data/Landmark Kaggle/test.csv')

print('all_test_keys ', len(all_test_keys))
print('existing_keys ', len(existing_keys))
input()
neighbors = np.load('100_nearest_neighbors_resnet.npy')
print('neighbors', neighbors)


def get_neighbors(neighbors, existing_train_keys, existing_test_keys, test_key):
    index = np.where(existing_test_keys.__eq__(str(test_key)))[0][0]
    neighbors_indices = np.array(neighbors[index])
    return existing_train_keys[neighbors_indices]


def get_dummy_neighbors(existing_train_keys):
    result = []
    for i in np.random.random_integers(low=0, high=existing_train_keys.shape[0] - 1, size=100):
        result.append(existing_train_keys[i])
    return result


# id,images
# 000088da12d664db,0370c4c856f096e8 766677ab964f4311 e3ae4dcee8133159...
# etc.
count_dummy = 0
with open('natasha_submission-resnet-random-dummy.csv', "w") as fout:
    fout.write('id,images\n')
    for key in all_test_keys:
        fout.write(key + ',')
        if key in existing_keys:
            neighbors_list = get_neighbors(neighbors, existing_train_keys, existing_keys, key)
        else:
            neighbors_list = get_dummy_neighbors(existing_train_keys)
            count_dummy = count_dummy + 1
            print('count_dummy', count_dummy)
        fout.write(" ".join([str(el) for el in neighbors_list]))
        fout.write('\n')

print('count_dummy', count_dummy)
