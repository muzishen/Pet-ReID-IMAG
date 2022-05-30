import os
import pandas as pd
import numpy as np
import cv2

from tqdm import tqdm
from collections import defaultdict

csv_path = '/Users/zjh/Documents/Dataset/pet_biometric_challenge_2022/train/train_data.csv'
img_dir = '/Users/zjh/Documents/Dataset/pet_biometric_challenge_2022/train/images'
csv = pd.read_csv(csv_path)

id2files = defaultdict(list)
for idx, col in csv.iterrows():
    dog_id, filename = col
    id2files[dog_id].append(filename)

count = 0
for ind, files in tqdm(id2files.items()):
    if len(files) != 2:
        continue

    img1 = cv2.imread(os.path.join(img_dir, files[0]))
    img2 = cv2.imread(os.path.join(img_dir, files[1]))

    mean_1 = np.mean(img1, axis=(0, 1))
    mean_2 = np.mean(img2, axis=(0, 1))

    diff = np.linalg.norm(mean_2 - mean_1)
    if diff < 2:
        count += 1
        print(ind)

print(count)
