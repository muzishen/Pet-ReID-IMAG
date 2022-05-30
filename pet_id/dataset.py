# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""

import glob
import os
from collections import defaultdict
import os.path as osp
from fastreid.data.datasets import DATASET_REGISTRY
from fastreid.data.datasets.bases import ImageDataset

__all__ = ["PetID", "PetIDTest", ]


@DATASET_REGISTRY.register()
class PetID(ImageDataset):
    dataset_name = "per_id"
    dataset_dir = "/mnt/data/data/cvpr2022_reid/pet_biometric_challenge_2022/train"

    def __init__(self, root="datasets", **kwargs):
        self.root = root

        self.data_path = os.path.join(self.root, self.dataset_dir, "images")
        self.train_dir = './data/dir_train_fusai'
        self.query_label = os.path.join(self.root, self.dataset_dir, "val_query.txt")
        self.gallery_label = os.path.join(self.root, self.dataset_dir, "val_gallery.txt")

        train = self.process_train(self.train_dir)
        query= self.process_train(self.train_dir)
        gallery = self.process_train(self.train_dir)
        super().__init__(train, query, gallery, **kwargs)

    def process_train(self, train_dir):
        # img_paths = glob.glob(osp.join(train_dir, '*.jpg'))
        # pid_container = set()
        # for img_path in img_paths:
        #     pid = int(img_path.split('/')[-1].split('_')[0])
        #     # pid = list(pid)[0]
        #     if pid == -1: continue  # junk images are just ignored
        #     pid_container.add(pid)
        # pid2label = {pid: label for label, pid in enumerate(pid_container)}
        # dataset = []
        # for img_path in img_paths:
        #     pid = int(img_path.split('/')[-1].split('_')[0])
        #     pid = pid2label[pid]
        #     camid = 0
        #     dataset.append((img_path, pid, camid))
        # return dataset
        sub_dirs = os.listdir(train_dir)
        img_paths = []
        for cur_dir in sub_dirs:
            cur = os.path.join(train_dir, cur_dir)
            files = glob.glob(os.path.join(cur, '*.jpg'))
            if len(files) < 2:
                continue
            for f in files:
                img_path = f
                pid = self.dataset_name + "_" + str(int(cur_dir))
                camid = self.dataset_name + '_0'
                img_paths.append([img_path, pid, camid])
                #print(img_path, pid, camid)
        return img_paths

    def process_test(self, query_path, gallery_path):
        with open(query_path, 'r') as f:
            query_list = [i.strip('\n') for i in f.readlines()]

        with open(gallery_path, 'r') as f:
            gallery_list = [i.strip('\n') for i in f.readlines()]

        query_paths = []
        for data in query_list:
            img_name, pid = data.split(':')
            img_path = os.path.join(self.data_path, img_name)
            camid = 0
            query_paths.append([img_path, int(pid), camid])

        gallery_paths = []
        for data in gallery_list:
            img_name, pid = data.split(':')
            img_path = os.path.join(self.data_path, img_name)
            camid = 1
            gallery_paths.append([img_path, int(pid), camid])

        return query_paths, gallery_paths


@DATASET_REGISTRY.register()
class PetIDTest(ImageDataset):
    dataset_name = "per_id_test"
    dataset_dir = '/mnt/data/data/cvpr2022_reid/pet_biometric_challenge_2022/test'

    def __init__(self, root='datasets', **kwargs):
        self.root = root

        self.data_path = os.path.join(self.root, self.dataset_dir, "images")
        self.query_path = './data/test/test/'
        self.gallery_path = './data/test/test/'
        
        query = self.process_test(self.query_path)
        gallery = self.process_test(self.gallery_path)

        super().__init__([], query, gallery)

    def process_test(self, test_path):
        img_paths = glob.glob(osp.join(test_path, '*.jpg'))
        
        data = []
        for img_path in img_paths:
            # img_path = os.path.join(self.data_path, img_path)
            img_name = img_path.split("/")[-1]
            #print(img_path, img_name)
            data.append([img_path, img_name, "pet_0"])
        return data


@DATASET_REGISTRY.register()
class PetIDTestPseudo(ImageDataset):
    dataset_name = "per_id_test_pseudo"
    dataset_dir = 'pet_id'

    def __init__(self, root='datasets', **kwargs):
        self.root = root

        self.data_path = os.path.join(self.root, self.dataset_dir)
        # self.query_dir = os.path.join(self.root, self.dataset_dir, "validation", "images")
        self.query_dir = os.path.join(self.root, self.dataset_dir, "train", "images")
        self.gallery_dir = os.path.join(self.root, self.dataset_dir, "train", "images")

        query = self.process_test(self.query_dir)
        gallery = self.process_test(self.gallery_dir)

        super().__init__([], query, gallery)

    def process_test(self, img_dir):
        img_paths = glob.glob(os.path.join(img_dir, '*.jpg'))

        data = []
        for img_path in img_paths:
            img_name = os.path.basename(img_path)
            data.append([img_path, img_name, "pet_0"])
        return data
