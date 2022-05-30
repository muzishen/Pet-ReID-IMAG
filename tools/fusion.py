import os
import shutil

import numpy as np
import pandas as pd

from tqdm import tqdm


def PaiXu(list_A):
    list_sort = []
    for i in list_A:
        sf = sum(j <= i for j in list_A)
        list_sort.append(sf)
    return list_sort
def main():
    submit = pd.read_csv('/mnt/data/data/cvpr2022_reid/pet_biometric_challenge_2022/validation/valid_data.csv')
####S101
    s101_data = pd.read_csv('logs/fusion_submit/s101_submit.csv')
    s101_pred = s101_data['prediction'].values.tolist()
    s101_sort =PaiXu(s101_pred)


#####convnext
    convb_data = pd.read_csv('logs/fusion_submit/convb_submit.csv')
    convb_pred = convb_data['prediction'].values.tolist()
    convb_sort = PaiXu(convb_pred)
    
    fusion_sort = np.sum([s101_sort,convb_sort],axis=0).tolist()

    fusion_sort = [x/2 for x in fusion_sort]

    submit['prediction'] = fusion_sort
    submit.to_csv('logs/fusion_submit/submit.csv', index=False)    
if __name__ == '__main__':
    main()
