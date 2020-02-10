#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 25 Jan 2020 20:29:09

@author: jack
"""
"""
Find the best parameters (k1, k2, lambda) of reranking for a certain dataset
"""

import numpy as np
import sys
import torch
import heapq

sys.path.append('../')
from utils.metrics import eval_func
from utils.reranking import re_ranking


qf = torch.load('../log/qfeats.pth')
qf = torch.nn.functional.normalize(qf, dim=1, p=2)
print("The q feature is normalized")

gf = torch.load('../log/gfeats.pth')
gf = torch.nn.functional.normalize(gf, dim=1, p=2)
print("The g feature is normalized")

pids = np.load('../log/pids.npy')
q_pids = pids[:qf.shape[0]]
g_pids = pids[qf.shape[0]:]
camids = np.load('../log/camids.npy')
q_camids = camids[:qf.shape[0]]
g_camids = camids[qf.shape[0]:]
print('qfeats:{}, gfeats:{}, q_pids:{}, g_pids:{}'.format(qf.shape, gf.shape, len(q_pids), len(g_pids)))

m, n = qf.shape[0], gf.shape[0]
distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
          torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
distmat.addmm_(1, -2, qf, gf.t())
distmat = distmat.cpu().numpy()
cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)
print('mAP:{}, rank1:{}'.format(mAP, cmc[0]))

k1_list = [i for i in range(2, 10)] + [i for i in range(10, 20, 2)] + [i for i in range(20, 40, 10)] + \
          [i for i in range(40, 61, 20)]
k2_list = [i for i in range(2, 10)] + [i for i in range(10, 20, 2)] + [i for i in range(20, 40, 10)]


lambda_list = [0.1, 0.2, 0.3, 0.4]

best_queue = []
MAX_QUEUE_LEN = 10
for k1 in k1_list:
    for k2 in k2_list:
        if k2 > 0.7 * k1:
            break
        for lam in lambda_list:
            print('=======================================================================')
            print(best_queue)
            distmat_reranking = re_ranking(qf, gf, k1=k1, k2=k2, lambda_value=lam)
            cmc, mAP = eval_func(distmat_reranking, q_pids, g_pids, q_camids, g_camids)
            print('Processing k1:{}, k2:{}, k3:{}'.format(k1, k2, lam))
            print('mAP:{}, rank1:{}'.format(mAP, cmc[0]))
            score = mAP + cmc[0]
            if len(best_queue) <= MAX_QUEUE_LEN:
                heapq.heappush(best_queue, (score, (k1, k2, lam, mAP, cmc[0])))
            else:
                if score > best_queue[0][0]:
                    heapq.heappop(best_queue)
                    heapq.heappush(best_queue, (score, (k1, k2, lam, mAP, cmc[0])))
                else:
                    pass
np.save('../log/best_reranking_params.npy', best_queue)
