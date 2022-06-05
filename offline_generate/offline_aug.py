# -*- coding: utf-8 -*-
"""
Created on Sun Jul 18 22:02:45 2021

@author: HUAWEI
"""

import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np
import cv2
from pascal_voc_writer import Writer
import xml.etree.ElementTree as ET
import glob
from util import sequence
from util import annotation as an
import shutil
import sys

import os 

OUTPUT_DIR = './fusai'
AUGMENT_SIZE = 1

image_path = r'./clear_rename_images_v2'
def main():
    for file in glob.glob('%s/*.jpg' % image_path):

        augment(file)



def augment(filename):

    seq = sequence.get()

    for i in range(AUGMENT_SIZE):

        sp = filename.split('/')[-1].split('.')

        new_name = sp[0] + '-'+str(i) + '.jpg'
        outfile = os.path.join(OUTPUT_DIR, new_name)

        seq_det = seq.to_deterministic()

        image = cv2.imread(filename)
    
        image_aug = seq_det.augment_images([image])[0]
  
        cv2.imwrite(outfile, image_aug)

        
if __name__ == "__main__":       
    main()
