import numpy as np

import script

with open('keys_for_test', "r") as fin:
    line = fin.readline()

existing_test_keys = np.array(list(map(str, line.split())))

with open('keys_for_train', "r") as fin:
    line = fin.readline()

existing_train_keys = np.array(list(map(str, line.split())))

for i, key in enumerate(existing_test_keys):
    existing_test_keys[i] = existing_test_keys[i].replace('.jpg', '')

for i, key in enumerate(existing_train_keys):
    existing_train_keys[i] = existing_train_keys[i].replace('.jpg', '')

print('existing_test_keys ', existing_test_keys)

keys_urls = script.ParseData('test.csv')
all_test_keys = []
for i, pair in enumerate(keys_urls):
    all_test_keys.append(pair[0])

print('all_test_keys ', all_test_keys)

neighbors = np.load('100_nearest_neighbors.npy')
print('neighbors', neighbors)


def get_neighbors(neighbors, existing_train_keys, existing_test_keys, test_key):
    index = np.where(existing_test_keys.__eq__(str(test_key)))[0][0]
    neighbors_indices = np.array(neighbors[index])
    return existing_train_keys[neighbors_indices]


def get_dummy_neighbors():
    result = []
    for i in range(100):
        result.append('b6055a3d08503c42')
    return result


# id,images
# 000088da12d664db,0370c4c856f096e8 766677ab964f4311 e3ae4dcee8133159...
# etc.
count_dummy = 0
with open('natasha_submission.csv', "w") as fout:
    fout.write('id,images\n')
    for key in all_test_keys:
        fout.write(key + ',')
        if key in existing_test_keys:
            neighbors_list = get_neighbors(neighbors, existing_train_keys, existing_test_keys, key)
        else:
            neighbors_list = get_dummy_neighbors()
            count_dummy = count_dummy + 1
            print('count_dummy', count_dummy)
        fout.write(" ".join([str(el) for el in neighbors_list]))
        fout.write('\n')

print('count_dummy', count_dummy)