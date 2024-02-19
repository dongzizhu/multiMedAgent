############################################################################################
###
### This code is adapted from https://github.com/Holipori/EKAID/tree/main
###
############################################################################################

import pydicom
import os
import cv2
import numpy as np
from pydicom.pixel_data_handlers.util import apply_voi_lut
from PIL import Image
import pickle
import argparse
from tqdm import tqdm

import PIL # optional


def mimic_jpg2png(data_path, out_path):
    data_path = os.path.join(data_path, '2.0.0/files')
    p_folder = os.listdir(data_path)
    size = 1024
    n = 0
    dict = []
    mimic_shapeid = {}

    for p_fold in p_folder:
        if p_fold == 'index.html':
            continue
        p_path = os.path.join(data_path, p_fold)
        pp_folder = os.listdir(p_path)
        for pp_fold in pp_folder:
            if pp_fold == 'index.html':
                continue
            pp_path = os.path.join(p_path,pp_fold)
            if not os.path.isdir(pp_path):
                continue
            s_folder = os.listdir(pp_path)
            for s_fold in s_folder:
                if s_fold == 'index.html':
                    continue
                s_path = os.path.join(pp_path, s_fold)
                if not os.path.isdir(s_path):
                    continue
                files = os.listdir(s_path)
                for file in files:
                    if file == 'index.html':
                        continue
                    new_filename = os.path.join(out_path, file.replace('.jpg', '.png'))
                    record = {}
                    file_path = os.path.join(s_path,file)
                    im = Image.open(file_path)
                    record['image'] = file.replace('.jpg','')
                    record['height'] = im.size[0]
                    record['width'] = im.size[1]
                    dict.append(record)
                    mimic_shapeid[record['image']] = n
                    if os.path.exists(new_filename):
                        pass
                    else:
                        im = im.resize((size, size), Image.LANCZOS)
                        im.save(new_filename)
                    n += 1
                    if n % 50 == 0:
                        print('{} image converted'.format(n))
    if not os.path.exists('data'):
        os.mkdir('data')

    dicom2id = {}
    for i in tqdm(range(len(dict))):
        dicom2id[dict[i]['image']] = i
    with open('data/dicom2id.pkl', "wb") as tf:
        pickle.dump(dicom2id, tf)
        print('dicom2id saved')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--mimic_path", type=str, default=None, required=True, help="path to mimic-cxr-jpg dataset")
    parser.add_argument("-o", "--out_path", type=str, default=None, required=True, help="path to output png dataset")
    args = parser.parse_args()
    mimic_jpg2png(data_path = args.mimic_path, out_path = args.out_path)

if __name__ == '__main__':
    main()