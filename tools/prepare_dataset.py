import os
import argparse
import random
import shutil
import cv2
import pandas as pd

from tqdm import tqdm
from collections import defaultdict


def prepare_train(args):
    annotations = pd.read_csv(os.path.join(args.dataset_dir, 'train', 'train_data.csv'))
    save_dir = os.path.join(args.dataset_dir, 'train')

    # remove existed files
    for file in ['train.txt', 'val_gallery.txt', 'val_query.txt']:
        if os.path.isfile(os.path.join(save_dir, file)):
            os.remove(os.path.join(save_dir, file))

    # csv to dict
    id2files = defaultdict(list)
    for idx, col in annotations.iterrows():
        dog_id, filename = col
        id2files[dog_id].append(filename)

    # filter dog ids with only one sample
    dog_ids = list(id2files.keys())
    dog_ids = list(filter(lambda x: len(id2files[x]) > 1, dog_ids))

    # random shuffle
    if args.seed is not None:
        random.seed(args.seed)
    random.shuffle(dog_ids)

    # split k fold
    k_fold_dog_ids = []
    for i in range(args.k_fold):
        k_fold_dog_ids.append(dog_ids[i::args.k_fold])

    train_ids = []
    for i in range(args.k_fold):
        if i != args.val_fold or args.no_val:
            train_ids.extend(k_fold_dog_ids[i])
    val_ids = k_fold_dog_ids[args.val_fold]

    # write to txt file -> train
    img_dir = os.path.join(args.dataset_dir, 'train', 'images')
    for idx in tqdm(train_ids, desc='generating training file'):
        for file in id2files[idx]:
            if args.flip_aug:
                filename = file[:-4] + '_flip' + file[-4:]
                if not os.path.isfile(os.path.join(img_dir, filename)):
                    img = cv2.imread(os.path.join(img_dir, file))
                    img = img[:, ::-1, :]
                    cv2.imwrite(os.path.join(img_dir, filename), img)
                with open(os.path.join(save_dir, 'train.txt'), 'a+') as f:
                    f.write(filename + ':' + str(idx + 6000) + '\n')

            with open(os.path.join(save_dir, 'train.txt'), 'a+') as f:
                f.write(file + ':' + str(idx) + '\n')

    # write to txt file -> valid
    for idx in tqdm(val_ids, desc='generating valid file'):
        for ind, file in enumerate(id2files[idx]):
            if args.flip_aug:
                filename = file[:-4] + '_flip' + file[-4:]
                if not os.path.isfile(os.path.join(img_dir, filename)):
                    img = cv2.imread(os.path.join(img_dir, file))
                    img = img[:, ::-1, :]
                    cv2.imwrite(os.path.join(img_dir, filename), img)

                if ind >= len(id2files[idx]) // 2:
                    with open(os.path.join(save_dir, 'val_gallery.txt'), 'a+') as f:
                        f.write(filename + ':' + str(idx + 6000) + '\n')
                else:
                    with open(os.path.join(save_dir, 'val_query.txt'), 'a+') as f:
                        f.write(filename + ':' + str(idx + 6000) + '\n')

            if ind >= len(id2files[idx]) // 2:
                with open(os.path.join(save_dir, 'val_gallery.txt'), 'a+') as f:
                    f.write(file + ':' + str(idx) + '\n')
            else:
                with open(os.path.join(save_dir, 'val_query.txt'), 'a+') as f:
                    f.write(file + ':' + str(idx) + '\n')

    if args.saved_by_ID:
        img_dir = os.path.join(args.dataset_dir, 'train', 'images')
        save_img_dir = args.save_dir

        for idx in tqdm(train_ids, desc='save the images separately by ID'):
            cur_save_dir = os.path.join(save_img_dir, f'{idx}'.rjust(5, '0'))
            os.makedirs(cur_save_dir, exist_ok=True)

            for file in id2files[idx]:
                shutil.copy(os.path.join(img_dir, file), cur_save_dir)


def prepare_test(args):
    annotations = pd.read_csv(os.path.join(args.dataset_dir, 'validation', 'valid_data.csv'))
    save_dir = os.path.join(args.dataset_dir, 'validation')

    # remove existed files
    for file in ['gallery.txt', 'query.txt']:
        if os.path.isfile(os.path.join(save_dir, file)):
            os.remove(os.path.join(save_dir, file))

    # duplicate removal
    imagesA = set()
    imagesB = set()
    for idx, col in annotations.iterrows():
        imageA, imageB = col
        imagesA.add(imageA)
        imagesB.add(imageB)

    for imageA in tqdm(imagesA, desc='generating test query files'):
        with open(os.path.join(save_dir, 'query.txt'), 'a+') as f:
            f.write(imageA + '\n')

    for imageB in tqdm(imagesB, desc='generating test files'):
        with open(os.path.join(save_dir, 'gallery.txt'), 'a+') as f:
            f.write(imageB + '\n')


def parse_args():
    parser = argparse.ArgumentParser(description="prepare dataset")
    parser.add_argument('dataset_dir', help='The directory of training image')
    parser.add_argument('--flip-aug',
                        action='store_true',
                        help='Define the flipped image as a different ID')
    parser.add_argument('--saved-by-ID',
                        action='store_true',
                        help='Only save training images')
    parser.add_argument('--save-dir', type=str, default=None)
    parser.add_argument('--k-fold', type=int, default=4)
    parser.add_argument('--val-fold', type=int, default=0)
    parser.add_argument('--no-val', action='store_true', help='train with all data')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    if args.saved_by_ID:
        assert args.save_dir is not None, 'images will be saved in save_dir'

    prepare_train(args)
    prepare_test(args)
