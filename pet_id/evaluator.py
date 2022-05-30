# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""

import json
import logging
import os
import itertools
from collections import OrderedDict, defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from fastreid.evaluation import ReidEvaluator
from fastreid.evaluation.query_expansion import aqe
from fastreid.utils import comm
from fastreid.utils.compute_dist import build_dist

logger = logging.getLogger("fastreid.pet_id_submission")


def partition_arg_topK(matrix, K, axis=0):
    """
    perform topK based on np.argpartition
    :param matrix: to be sorted
    :param K: select and sort the top K items
    :param axis: 0 or 1. dimension to be sorted.
    :return:
    """
    a_part = np.argpartition(matrix, K, axis=axis)
    if axis == 0:
        row_index = np.arange(matrix.shape[1 - axis])
        a_sec_argsort_K = np.argsort(matrix[a_part[0:K, :], row_index], axis=axis)
        return a_part[0:K, :][a_sec_argsort_K, row_index]
    else:
        column_index = np.arange(matrix.shape[1 - axis])[:, None]
        a_sec_argsort_K = np.argsort(matrix[column_index, a_part[:, 0:K]], axis=axis)
        return a_part[:, 0:K][column_index, a_sec_argsort_K]


def get_test_pairs(df: pd.DataFrame):
    pairs = []
    for idx, col in df.iterrows():
        imageA, imageB = col
        pairs.append((imageA, imageB))
    return pairs


def write_txt(l: list, save_path):
    if os.path.isfile(save_path):
        os.remove(save_path)
    for item in l:
        with open(save_path, 'a+') as f:
            f.write(item + '\n')


class PetIDEvaluator(ReidEvaluator):

    def process(self, inputs, outputs):
        prediction = {
            'feats': outputs.to(self._cpu_device, torch.float32),
            'pids': inputs['targets'],
            'camids': inputs['camids']
        }
        self._predictions.append(prediction)

    # def evaluate(self):
    #     if comm.get_world_size() > 1:
    #         comm.synchronize()
    #         predictions = comm.gather(self._predictions, dst=0)
    #         predictions = list(itertools.chain(*predictions))
    #         if not comm.is_main_process():
    #             return {}
    #     else:
    #         predictions = self._predictions

    #     features = []
    #     pids = []
    #     camids = []
    #     for prediction in predictions:
    #         features.append(prediction['feats'])
    #         pids.extend(prediction['pids'])
    #         camids.extend(prediction['camids'])

    #     features = torch.cat(features, dim=0)
    #     query_features = features[:self._num_query]
    #     gallery_features = features[self._num_query:]

    #     query_filename = pids[:self._num_query]
    #     gallery_filename = pids[self._num_query:]
    #     np.save(os.path.join(self.cfg.OUTPUT_DIR, 'query_f.npy'), query_features)
    #     np.save(os.path.join(self.cfg.OUTPUT_DIR, 'gallery_f.npy'), gallery_features)
    #     if self.cfg.TEST.AQE.ENABLED:
    #         logger.info("Test with AQE setting")
    #         qe_time = self.cfg.TEST.AQE.QE_TIME
    #         qe_k = self.cfg.TEST.AQE.QE_K
    #         alpha = self.cfg.TEST.AQE.ALPHA
    #         query_features, gallery_features = aqe(query_features, gallery_features, qe_time, qe_k, alpha)

    #     dist = build_dist(query_features, gallery_features, self.cfg.TEST.METRIC)

    #     if self.cfg.TEST.RERANK.ENABLED:
    #         logger.info("Test with rerank setting")
    #         k1 = self.cfg.TEST.RERANK.K1
    #         k2 = self.cfg.TEST.RERANK.K2
    #         lambda_value = self.cfg.TEST.RERANK.LAMBDA

    #         if self.cfg.TEST.METRIC == "cosine":
    #             query_features = F.normalize(query_features, dim=1)
    #             gallery_features = F.normalize(gallery_features, dim=1)

    #         rerank_dist = build_dist(query_features, gallery_features, metric="jaccard", k1=k1, k2=k2)
    #         dist = rerank_dist * (1 - lambda_value) + dist * lambda_value

    #     if self.cfg.TEST.SAVE_DIST.ENABLED:
    #         #save_dist = np.copy(dist).astype(np.float16)
    #         np.save(os.path.join(self.cfg.OUTPUT_DIR, 'dist.npy'), dist)
    #         write_txt(query_filename, os.path.join(self.cfg.OUTPUT_DIR, 'query_filename.txt'))
    #         write_txt(gallery_filename, os.path.join(self.cfg.OUTPUT_DIR, 'gallery_filename.txt'))

    #     submit = pd.read_csv('/mnt/data/data/cvpr2022_reid/pet_biometric_challenge_2022/test/test_data.csv')
        
    #     test_pair = get_test_pairs(submit)

    #     prediction = []
    #     for imageA, imageB in test_pair:
    #         #print(imageA, imageB)
    #         row = query_filename.index(imageA)
    #         column = gallery_filename.index(imageB)
    #         score = (1 - dist[row][column])
    #         prediction.append(score)

    #     submit['prediction'] = prediction
    #     submit.to_csv(os.path.join(os.path.join(self.cfg.OUTPUT_DIR, 'submit.csv')), index=False)

    #     return OrderedDict(submit='finished')


    def evaluate(self):
        if comm.get_world_size() > 1:
            comm.synchronize()
            predictions = comm.gather(self._predictions, dst=0)
            predictions = list(itertools.chain(*predictions))
            if not comm.is_main_process():
                return {}
        else:
            predictions = self._predictions

        pids = []
        for prediction in predictions:
            pids.extend(prediction['pids'])

        query_filename = pids[:self._num_query]
        gallery_filename = pids[self._num_query:]


        s101_dist_224_q = torch.from_numpy(np.load('./logs/s101_224/query_f.npy'))
        s101_dist_224_g = torch.from_numpy(np.load('./logs/s101_224/gallery_f.npy'))

        s101_dist_256_q = torch.from_numpy(np.load('./logs/s101_256/query_f.npy'))
        s101_dist_256_g = torch.from_numpy(np.load('./logs/s101_256/gallery_f.npy'))

        s101_dist_288_q = torch.from_numpy(np.load('./logs/s101_288/query_f.npy'))
        s101_dist_288_g = torch.from_numpy(np.load('./logs/s101_288/gallery_f.npy'))

        s200_dist_224_q = torch.from_numpy(np.load('./logs/s200_224/query_f.npy'))
        s200_dist_224_g = torch.from_numpy(np.load('./logs/s200_224/gallery_f.npy'))

       
        query_features = torch.cat((s101_dist_224_q, s101_dist_256_q, s101_dist_288_q, s200_dist_224_q), axis=1)
        gallery_features = torch.cat((s101_dist_224_q, s101_dist_256_q, s101_dist_288_q, s200_dist_224_q), axis=1)
 
        print(query_features.shape, gallery_features.shape)
        dist = build_dist(query_features, gallery_features, self.cfg.TEST.METRIC)

        submit = pd.read_csv('./data/test/test_data.csv')
        test_pair = get_test_pairs(submit)

        prediction = []
        for imageA, imageB in test_pair:
            row = query_filename.index(imageA)
            column = gallery_filename.index(imageB)
            score = (1 - dist[row][column]) * 100
            prediction.append(score)

        submit['prediction'] = prediction
        submit.to_csv(os.path.join(os.path.join(self.cfg.OUTPUT_DIR, 'submit.csv')), index=False)

        return OrderedDict(submit='finished')
