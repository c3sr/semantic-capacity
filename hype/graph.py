#!/usr/bin/env python3
import random
from hype.utils import get_id_from_dict
from collections import defaultdict as ddict
import numpy as np
from numpy.random import choice
import torch as th
from torch import nn
from torch.utils.data import Dataset as DS
import networkx as nx


def load_npmi(path, global_dict, weight_opt=('npmi', 0.1)):
    assert weight_opt[0] == 'npmi'

    idx = []
    weight = []

    with open(path) as fin:
        for line in fin:
            parent, child, npmi = line.strip().split("\t")

            if parent == child:
                continue

            pid = global_dict[parent]
            cid = global_dict[child]
            npmi = float(npmi)

            w = {
                'npmi': npmi,
            }

            if w[weight_opt[0]] > weight_opt[1]:
                idx.append([pid, cid])
                weight.append(w[weight_opt[0]])

        return idx, weight


def build_global_dict(data):
    global_dict = {}
    id2phrase = {}

    with open(f"data/{data}/lemma_dict_5.txt") as fin:
        for line in fin:
            _, lemma = line.strip().split("\t")
            idx = get_id_from_dict(lemma, global_dict)
            id2phrase[idx] = lemma

    return global_dict, id2phrase


def prepare_data(data, weight_opt=('cnt', 0)):
    global_dict, id2phrase = build_global_dict(data)
    
    total_co_occur, total_weight = load_npmi(path=f"data/{data}/npmi_5.txt",
                                                 global_dict=global_dict,
                                                 weight_opt=weight_opt)

    random.shuffle(total_co_occur)
    total_co_occur = np.array(total_co_occur)

    objects = [id2phrase[i] for i in range(len(id2phrase))]
    weights = np.array(total_weight)  

    return total_co_occur, objects, weights, global_dict, id2phrase


class Embedding(nn.Module):
    def __init__(self, size, dim, manifold, sparse=True):
        super(Embedding, self).__init__()
        self.dim = dim
        self.nobjects = size
        self.manifold = manifold
        self.lt = nn.Embedding(size, dim, sparse=sparse)
        self.dist = manifold.distance
        self.pre_hook = None
        self.post_hook = None
        self.init_weights(manifold)

    def init_weights(self, manifold, scale=1e-4):
        manifold.init_weights(self.lt.weight, scale)

    def forward(self, inputs):
        e = self.lt(inputs)
        with th.no_grad():
            e = self.manifold.normalize(e)
        if self.pre_hook is not None:
            e = self.pre_hook(e)
        fval = self._forward(e)
        return fval

    def embedding(self):
        return list(self.lt.parameters())[0].data.cpu().numpy()

    def optim_params(self, manifold):
        return [{
            'params': self.lt.parameters(),
            'rgrad': manifold.rgrad,
            'expm': manifold.expm,
            'logm': manifold.logm,
            'ptransp': manifold.ptransp,
        }, ]


# This class is now deprecated in favor of BatchedDataset (graph_dataset.pyx)
class Dataset(DS):
    _neg_multiplier = 1
    _ntries = 10
    _sample_dampening = 0.75

    def __init__(self, idx, objects, weights, nnegs, unigram_size=1e8):
        assert idx.ndim == 2 and idx.shape[1] == 2
        assert weights.ndim == 1
        assert len(idx) == len(weights)
        assert nnegs >= 0
        assert unigram_size >= 0

        print('Indexing data')
        self.idx = idx
        self.nnegs = nnegs
        self.burnin = False
        self.objects = objects

        self._weights = ddict(lambda: ddict(int))
        self._counts = np.ones(len(objects), dtype=np.float)
        self.max_tries = self.nnegs * self._ntries
        for i in range(idx.shape[0]):
            t, h = self.idx[i]
            self._counts[h] += weights[i]
            self._weights[t][h] += weights[i]
        self._weights = dict(self._weights)
        nents = int(np.array(list(self._weights.keys())).max())
        assert len(objects) > nents, f'Number of objects do no match'

        if unigram_size > 0:
            c = self._counts ** self._sample_dampening
            self.unigram_table = choice(
                len(objects),
                size=int(unigram_size),
                p=(c / c.sum())
            )

    def __len__(self):
        return self.idx.shape[0]

    def weights(self, inputs, targets):
        return self.fweights(self, inputs, targets)

    def nnegatives(self):
        if self.burnin:
            return self._neg_multiplier * self.nnegs
        else:
            return self.nnegs

    @classmethod
    def collate(cls, batch):
        inputs, targets = zip(*batch)
        return th.cat(inputs, 0), th.cat(targets, 0)






