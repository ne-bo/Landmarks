#!/usr/bin/env python3

import numpy as np
from tqdm import tqdm
from copy import copy

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-start', action="store", dest="start_ind", type=int)
parser.add_argument('-stop', action="store", dest="stop_ind", type=int)

args = parser.parse_args()

nb_neighbors = np.load('nb_neighbors_total.npy')


def get_overlapping(arr1, arr2):
    ans = np.empty(arr1.shape[0])
    set1 = set()
    set2 = set()
    for i in range(arr1.shape[0]):
        set1.add(arr1[i])
        set2.add(arr2[i])
        ans[i] = len(set1&set2)
    return ans

def rbo_at_k(arr1, arr2, p=0.9):
    return (1-p) * np.sum(np.asarray([p**(i-1)/i for i in range(1, arr1.shape[0]+1)]) * get_overlapping(arr1, arr2))

def get_largest_entry(ans, list_of_ans):
    arr=[rbo_at_k(ans, x, p=0.9) for x in list_of_ans]
    return max(enumerate(arr), key=lambda x: x[1])[0]

def rerank(query_res, k=100):
    ans_for_neighbors = nb_neighbors[query_res, :]
    ans = []
    latest_item = query_res
    highest_ind = get_largest_entry(latest_item, ans_for_neighbors)
    latest_item = ans_for_neighbors[highest_ind,:]
    ans.append(query_res[highest_ind])
    ans_for_neighbors = np.delete(ans_for_neighbors, (highest_ind), axis=0)
    query_res = np.delete(query_res, (highest_ind), axis=0)
    
    for i in range(k-1):
        highest_ind = get_largest_entry(latest_item, ans_for_neighbors)
        latest_item = ans_for_neighbors[highest_ind,:]
        ans.append(query_res[highest_ind])
        ans_for_neighbors = np.delete(ans_for_neighbors, (highest_ind), axis=0)
        query_res = np.delete(query_res, (highest_ind), axis=0)
    return ans

if __name__ == "__main__":
    neighbors = np.load('100_nearest_neighbors.npy')
    real_end_ind = neighbors.shape[0]
    steps_to_save = 2000
    start_ind = args.start_ind
    stop_ind = args.stop_ind #not including last ind
    stop_ind = min(stop_ind, real_end_ind)
    last_save_ind = start_ind
    for i in tqdm(range(start_ind, stop_ind)):
        neighbors[i,:] = rerank(copy(neighbors[i,:]))
        if (i - last_save_ind+1) % steps_to_save == 0:
            np.save('ranks/reranked_%s' % i, neighbors[last_save_ind:i+1,:])
            tqdm.write("Saved from %s to %s." % (last_save_ind, i))
            last_save_ind = i+1
    if i>last_save_ind:
        np.save('ranks/reranked_%s' % i, neighbors[last_save_ind:i+1,:])
        tqdm.write("Saved from %s to %s." % (last_save_ind, i))