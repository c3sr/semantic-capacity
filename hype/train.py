#!/usr/bin/env python3
import copy
import torch as th
import torch.nn as nn
import numpy as np
import timeit
import gc
from torch.utils import data as torch_data
from hype.utils import *
from hype.lorentz import LorentzManifold


_lr_multiplier = 0.1


def train(
        device,
        model,
        data,
        optimizer,
        opt,
        log,
        check_dir,
        rank=1,
        queue=None,
        ctrl=None,
        checkpointer=None,
        progress=False,
        global_dict=None,
        id2phrase=None,
):
    if isinstance(data, torch_data.Dataset):
        loader = torch_data.DataLoader(data, batch_size=opt.batchsize,
                                       shuffle=True, num_workers=opt.ndproc)
    else:
        loader = data

    epoch_loss = th.Tensor(len(loader))
    counts = th.zeros(model.nobjects, 1).to(device)

    best = None
    best_acc = 0
    best_model = copy.deepcopy(model.state_dict())

    manifold = LorentzManifold()

    kw_level = {}
    with open(f"data/{opt.data}/wiki_term_level.txt") as f:
        for line in f.readlines():
            x, h = line.strip().split('\t')
            h = int(h)
            kw_level[x] = h
    
    for epoch in range(opt.epoch_start+1, opt.epochs+1):
        
        epoch_loss.fill_(0)
        data.burnin = False
        lr = opt.lr
        t_start = timeit.default_timer()
        if epoch < opt.burnin:
            data.burnin = True
            lr = opt.lr * _lr_multiplier
            if rank == 1:
                log.info(f'Burn in negs={data.nnegatives()}, lr={lr}')

        loader_iter = loader

        for i_batch, (inputs, weights, _) in enumerate(loader_iter):

            lt = model.w_avg if hasattr(model, 'w_avg') else model.lt.weight.data

            elapsed = timeit.default_timer() - t_start

            inputs = inputs.to(device)
            weights = weights.to(device)

            optimizer.zero_grad()
            preds = model(inputs)

            loss = model.loss(preds, weights)
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), 1e-3)

            optimizer.step(lr=lr, counts=counts)
            epoch_loss[i_batch] = loss.cpu().item()
            
        if rank == 1:
            if hasattr(data, 'avg_queue_size'):
                qsize = data.avg_queue_size()
                misses = data.queue_misses()
                log.info(f'Average qsize for epoch was {qsize}, num_misses={misses}')

            if queue is not None:
                queue.put((epoch, elapsed, th.mean(epoch_loss).item(), model))
            elif ctrl is not None and epoch % opt.eval_each == (opt.eval_each - 1):
                with th.no_grad():
                    ctrl(model, epoch, elapsed, th.mean(epoch_loss).item())

            if checkpointer and hasattr(ctrl, 'checkpoint') and ctrl.checkpoint:
                checkpointer(model, epoch, epoch_loss)

        lt = model.w_avg if hasattr(model, 'w_avg') else model.lt.weight.data
        orders = load_wiki_pairs(opt.data, global_dict=global_dict)
        score, total = calculate_accuracy(lt, orders, manifold)
        acc_level = calculate_accuracy_level(lt, orders, manifold, kw_level, id2phrase)
        
        acc_msg = "({:.4f})[{}/{}]".format(score / total, score, total)
        level_msg = " ".join(["({:.4f})[{}/{}]".format(avg, r, t) for avg, r, t in acc_level])

        msg = f"Epoch: [{epoch}/{opt.epochs}] Acc: {acc_msg} Level-Acc: {level_msg}"

        if score / total > best_acc:
            best_acc = score / total
            best = msg
            best_model = copy.deepcopy(model.state_dict())

        log.info(msg)

        if epoch == opt.epochs:
            log.info(f"Best: {best}")
            model.load_state_dict(best_model)
            th.save(best_model, f"checkpoint/{check_dir}/model.torch")
            
        gc.collect()

