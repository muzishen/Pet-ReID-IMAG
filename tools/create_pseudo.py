import shutil
import argparse
import os.path as osp
import random
import numpy as np
import sys
import collections
import time
from datetime import timedelta
import os 
import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from infomap_cluster import get_dist_nbr, cluster_by_infomap


def generate_cluster_features(labels, features):
    centers = collections.defaultdict(list)
    for i, label in enumerate(labels):
        if label == -1:
            continue
        centers[labels[i]].append(features[i])

    centers = [
        torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())
    ]

    centers = torch.stack(centers, dim=0)
    return centers

def read_txt(file):
    with open(file, 'r') as f:
        data = f.readlines()
    data = list(map(lambda x: x.strip('\n'), data))
    return data

features = np.load('/mnt/data/code/pet_id/logs/s101_submit/query_f.npy')
query_files = read_txt('logs/s101_submit/query_filename.txt')
# features, _ = extract_features(model, cluster_loader, print_freq=50)
# features = torch.cat([features[f].unsqueeze(0) for f, _, _ in sorted(dataset.train)], 0)
features = torch.from_numpy(features)

features_array = F.normalize(features, dim=1).cpu().numpy()
feat_dists, feat_nbrs = get_dist_nbr(features=features_array, k=10, knn_method='faiss-gpu')
del features_array

s = time.time()
pseudo_labels = cluster_by_infomap(feat_nbrs, feat_dists, min_sim=0.5, cluster_num=3)
pseudo_labels = pseudo_labels.astype(np.intp)  # 1个样本对应1个标签,没聚类成功的就是-1

print('cluster cost time: {}'.format(time.time() - s))
num_cluster = len(set(pseudo_labels)) - (1 if -1 in pseudo_labels else 0)

# generate new dataset and calculate cluster centers



cluster_features = generate_cluster_features(pseudo_labels, features)

del features

# Create hybrid memory

new_features = F.normalize(cluster_features, dim=1).cuda()


pseudo_labeled_dataset = []
sf = 10000
save_path = '/mnt/data/data/cvpr2022_reid/pet_biometric_challenge_2022/train/pseudo_score50/'
if not os.path.exists(save_path):
    os.mkdir(save_path)
src_path = '/mnt/data/data/cvpr2022_reid/pet_biometric_challenge_2022/validation/images1/'
for i, (fname, label) in enumerate(zip(query_files, pseudo_labels)):
    if label != -1:
        new_label = str(label.item()+ sf)
        label_dir = save_path + new_label
        if not os.path.exists(label_dir):
            os.mkdir(label_dir)
        src_name = src_path + query_files[i]
        dst_name = save_path + new_label + '/' + new_label + '_' + query_files[i]
        #print(src_name, dst_name)
        shutil.copy(src_name, dst_name)
        pseudo_labeled_dataset.append(query_files[i])

print('==> Statistics for {} clusters {} sampler'.format(num_cluster, len(pseudo_labeled_dataset)))
