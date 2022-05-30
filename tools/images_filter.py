import os
import shutil

import numpy as np
import pandas as pd

from tqdm import tqdm


def read_txt(file):
    with open(file, 'r') as f:
        data = f.readlines()
    data = list(map(lambda x: x.strip('\n'), data))
    return data


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


def file2id(df: pd.DataFrame):
    file2id_dict = {}
    for idx, col in df.iterrows():
        dog_id, file = col
        file2id_dict[file] = dog_id
    return file2id_dict


def main():
    # train_csv = pd.read_csv('datasets/pet_id/train/train_data.csv')
    query_files = read_txt('logs/s101_submit/query_filename.txt')
    gallery_files = query_files
    dist = np.load('logs/s101_submit/dist.npy')
    # byid_root = 'datasets/pet_id/byid'
    # remove_vis = 'datasets/pet_id/remove'

    # if os.path.isdir(remove_vis):
    #     shutil.rmtree(remove_vis)

    similarity = (1 - dist) * 100.
    print(similarity)
    # file2id_dict = file2id(train_csv)
    topk_indices = partition_arg_topK(dist, 8, axis=1)
    print(topk_indices)
    sim_thr = 55
    save_path = '/mnt/data/data/cvpr2022_reid/pet_biometric_challenge_2022/train/pseudo_score65/'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    src_path = '/mnt/data/data/cvpr2022_reid/pet_biometric_challenge_2022/validation/images1/'
    duplicate = []
    sf = 10000
    for i in tqdm(range(len(query_files)), colour='pink'):
        cur_label_list = []
        
        query_file = query_files[i]
        
        cur_label_list.append(query_file)
        cur_sim = similarity[i]
        top_k = topk_indices[i]

        if query_file in duplicate:
            continue

        for k in top_k:
            if query_file == gallery_files[k]:
                continue
            if cur_sim[k] > sim_thr:
                duplicate.append(gallery_files[k])
                cur_label_list.append(gallery_files[k])
        
        if len(cur_label_list)>1:
            sf = sf + 1
            os.mkdir(save_path + str(sf))
            for j in range(len(cur_label_list)):
                src_name = src_path + cur_label_list[j]
                
                dst_name = save_path + str(sf)+ '/' + str(sf)+ '_' + cur_label_list[j]
                shutil.copy(src_name, dst_name)

                #shutil.copy(src_name, rm_save_dir)
    #duplicate = list(set(duplicate))

    # for file in tqdm(duplicate, desc='remove duplicate images'):
    #     cur_id = file2id_dict[file]
    #     file = os.path.join(byid_root, f'{cur_id}'.rjust(5, '0'), file)
    #     rm_save_dir = os.path.join(remove_vis, f'{cur_id}'.rjust(5, '0'))
    #     os.makedirs(rm_save_dir, exist_ok=True)
    #     shutil.copy(file, rm_save_dir)
    #     os.remove(file)


if __name__ == '__main__':
    main()
