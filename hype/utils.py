#!/usr/bin/env python3
import numpy as np


def get_id_from_dict(element, dictionary):
    if element not in dictionary:
        dictionary[element] = len(dictionary)
    return dictionary[element]


def calculate_accuracy(embeddings, orders, manifold):
    dists = manifold.pnorm(embeddings)

    total = 0
    score = 0

    for child, parents in orders.items():
        for parent in parents:
            dist_p = dists[parent]
            dist_c = dists[child]

            total += 1
            score += 1 if dist_p < dist_c else 0

    return score, total


def calculate_accuracy_level(embeddings, orders, manifold, kw_level, id2phrase):
    pred_level = {1: [], 2: [], 3: [], 4: [], 5: []}
    dists = manifold.pnorm(embeddings)
    
    for child, parents in orders.items():
        for parent in parents:
            dist_p = dists[parent]
            dist_c = dists[child]

            if dist_p < dist_c:
                pred_level[kw_level[id2phrase[parent]]].append(1)
            else:
                pred_level[kw_level[id2phrase[parent]]].append(0)

    return (np.mean(pred_level[1]), sum(pred_level[1]), len(pred_level[1])), (np.mean(pred_level[2]+pred_level[1]), sum(pred_level[2]+pred_level[1]), len(pred_level[2]+pred_level[1]))


def load_wiki_pairs(data, global_dict):
    path = f"data/{data}/wiki_pairs_5.txt"

    ret = {}
    with open(path) as fin:
        for line in fin:
            parent, child = line.strip().split("\t")
            pid, cid = global_dict[parent], global_dict[child]
            if cid in ret:
                ret[cid].add(pid)
            else:
                ret[cid] = {pid}
    return ret
