#!/usr/bin/env python3
# The implementation is partly based on https://github.com/facebookresearch/poincare-embeddings/
import argparse
import json
import logging
import os
from datetime import datetime

import numpy as np
import torch as th

from hype import train
from hype.checkpoint import LocalCheckpoint
from hype.graph import prepare_data
from hype.rsgd import RiemannianSGD
from hype.sn import initialize
from hype.lorentz import LorentzManifold


# Adapated from:
# https://thisdataguy.com/2017/07/03/no-options-with-argparse-and-python/
class Unsettable(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        super(Unsettable, self).__init__(option_strings, dest, nargs='?', **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        val = None if option_string.startswith('-no') else values
        setattr(namespace, self.dest, val)


def main():
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument("-mode", default="RUN")
    parser.add_argument('-data', type=str, default='cs')
    parser.add_argument('-device', type=int, default=0,
                        help='Cuda device')
    parser.add_argument('-dim', type=int, default=20,
                        help='Embedding dimension')
    parser.add_argument('-lr', type=float, default=100,
                        help='Learning rate')
    parser.add_argument('-epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('-batchsize', type=int, default=512,
                        help='Batchsize')
    parser.add_argument('-negs', type=int, default=50,
                        help='Number of negatives')
    parser.add_argument('-burnin', type=int, default=0,
                        help='Epochs of burn in')
    parser.add_argument('-dampening', type=float, default=0.75,
                        help='Sample dampening during burnin')
    parser.add_argument('-fresh', action='store_true', default=False,
                        help='Override checkpoint')
    parser.add_argument('-ndproc', type=int, default=8,
                        help='Number of data loading processes')
    parser.add_argument('-sym', action='store_true', default=False,
                        help='Symmetrize dataset')
    parser.add_argument('-maxnorm', '-no-maxnorm', default='500000',
                        action=Unsettable, type=int)
    parser.add_argument('-sparse', default=False, action='store_true',
                        help='Use sparse gradients for embedding table')
    parser.add_argument('-burnin_multiplier', default=0.1, type=float)
    parser.add_argument('-neg_multiplier', default=1.0, type=float)
    parser.add_argument('-quiet', action='store_true', default=False)
    parser.add_argument('-lr_type', choices=['scale', 'constant'], default='constant')
    parser.add_argument('-seed', type=int, default=15,
                        help='Seed')
  
    
    opt = parser.parse_args()

    th.manual_seed(opt.seed)
    np.random.seed(opt.seed)

    opt.weight_opt = ('npmi', 0.1)

    opt.checkdir = f"[{opt.data}]-{opt.mode}"

    time_tag = datetime.now().strftime('%m-%d-%H-%M-%S')
    opt.checkdir = f"{time_tag}-{opt.checkdir}"

    if not os.path.exists(f"checkpoint/{opt.checkdir}"):
        os.makedirs(f"checkpoint/{opt.checkdir}")

    # setup debugging and logigng
    log_path = f"checkpoint/{opt.checkdir}/log.log"
    log_level = logging.INFO
    log = logging.getLogger()
    log.setLevel(log_level)

    fh = logging.FileHandler(log_path)
    fh.setLevel(log_level)

    ch = logging.StreamHandler()
    ch.setLevel(log_level)

    formatter = logging.Formatter('%(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    log.addHandler(fh)
    log.addHandler(ch)

    # set default tensor type
    th.set_default_tensor_type('torch.DoubleTensor')
    # set device
    device = th.device(f"cuda:{opt.device}" if th.cuda.is_available() else 'cpu')
    log.info(f"use device: {device}")

    manifold = LorentzManifold(debug=False, max_norm=opt.maxnorm)
    opt.dim = manifold.dim(opt.dim)

    # data
    log.info('Loading data ...')
    idx, objects, weights, global_dict, id2phrase = prepare_data(opt.data, weight_opt=opt.weight_opt)

    model, data, model_name, conf = initialize(
        manifold, opt, idx, objects, weights, sparse=opt.sparse
    )
    
    # set burnin parameters
    data.neg_multiplier = opt.neg_multiplier
    train._lr_multiplier = opt.burnin_multiplier

    # Build config string for log
    log.info(f'json_conf: {json.dumps(vars(opt))}')

    if opt.lr_type == 'scale':
        opt.lr = opt.lr * opt.batchsize

    # setup optimizer
    optimizer = RiemannianSGD(model.optim_params(manifold), lr=opt.lr)

    # setup checkpoint
    checkpoint = LocalCheckpoint(
        f"checkpoint/{opt.checkdir}/checkpoint",
        include_in_all={'conf': vars(opt), 'objects': objects},
        start_fresh=opt.fresh
    )

    # get state from checkpoint
    state = checkpoint.initialize({'epoch': 0, 'model': model.state_dict()})
    model.load_state_dict(state['model'])
    opt.epoch_start = state['epoch']

    # control.checkpoint = True
    model = model.to(device)
    if hasattr(model, 'w_avg'):
        model.w_avg = model.w_avg.to(device)

    train.train(device, model, data, optimizer, opt, log,
                check_dir=opt.checkdir, ctrl=None,
                progress=not opt.quiet, global_dict=global_dict, id2phrase=id2phrase)


if __name__ == '__main__':
    main()
