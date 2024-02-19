import chunk
import os
import time
import torch
import torch.nn as nn
import _pickle as cPickle
import pandas as pd
from PIL import Image
import torchvision.transforms as T
import json
import numpy as np

import sys
sys.path.append("..")
import dataset_VQA
import utils

from tqdm import tqdm

transform = T.PILToTensor()
size = 84
size_ae = 128

if __name__ == '__main__':
    data_root = 'path_to_mimic-diff-vqa'
    image_root = 'path_to_png_images'
    path_study2dicom = os.path.join('data', 'study2dicom.pkl')
    dict_study2dicom = cPickle.load(open(path_study2dicom, 'rb'))

    path_qa = os.path.join(data_root, 'mimic_pair_questions.csv')


    for QT in ['abnormality', 'presence', 'level', 'location', 'type', 'view', 'all_and_diff']:
        print('QT: ', QT)
        df_qa = pd.read_csv(path_qa)
        if QT == 'all_and_diff':
            df_qa_train = df_qa[df_qa.split == 'train'] 
            df_qa_val = df_qa[df_qa.split == 'val']
            df_qa_test = df_qa[df_qa.split == 'test']
        elif QT == 'abnomality':
            df_qa_train = df_qa[(df_qa.question_type == QT) & (df_qa.question != 'what abnormalities are seen in this image?') & (df_qa.split == 'train')]
            df_qa_val = df_qa[(df_qa.question_type == QT) & (df_qa.question != 'what abnormalities are seen in this image?') & (df_qa.split == 'val')]
            df_qa_test = df_qa[(df_qa.question_type == QT) & (df_qa.question != 'what abnormalities are seen in this image?') & (df_qa.split == 'test')]    
        else:
            df_qa_train = df_qa[(df_qa.question_type == QT) & (df_qa.split == 'train')]
            df_qa_val = df_qa[(df_qa.question_type == QT) & (df_qa.split == 'val')]
            df_qa_test = df_qa[(df_qa.question_type == QT) & (df_qa.split == 'test')]    

        df_qa = pd.concat([df_qa_train, df_qa_val])
        
        print('Start generating dictionary...')
        all_q = df_qa.question.values.tolist()
        dictionaryDQA = dataset_VQA.Dictionary()
        for q in all_q:
            dictionaryDQA.tokenize(q, add_word=True)
        dictionaryDQA.dump_to_file('../src/data/data_{}/dictionary.pkl'.format(QT))
        print('Successfully generated dictionary')

        all_a = df_qa.answer.values.tolist()
        dict_ans2label = {}
        dict_label2ans = []
        idx_a = 0
        for ans in all_a:
            if ans not in dict_ans2label:
                dict_ans2label[ans] = idx_a
                dict_label2ans.append(ans)
                idx_a += 1

        with open('../src/data/data_{}/trainval_ans2label.pkl'.format(QT), 'wb') as f:
            cPickle.dump(dict_ans2label, f)
        with open('../src/data/data_{}/trainval_label2ans.pkl'.format(QT), 'wb') as f:
            cPickle.dump(dict_label2ans, f)
        print('Successfully generated ans and label')

        df_qa = pd.concat([df_qa, df_qa_test])
        all_sid = df_qa.study_id.values.tolist()
        all_rid = df_qa.ref_id.values.tolist()
        all_sid.extend(all_rid)
        study_list = list(set(all_sid))

        all_img_tensor = []
        all_ae_img_tensor = []
        dict_imgid2idx = {}
        for i, sid in enumerate(tqdm(study_list)): # 163640
            path_image = os.path.join(image_root, dict_study2dicom[sid] + '.png')
            img = Image.open(path_image)
            img_ae = img.resize((size_ae, size_ae), Image.LANCZOS)
            img = img.resize((size, size), Image.LANCZOS)
            img_tensor = transform(img)
            img_ae_tensor = transform(img_ae)
            all_img_tensor.append(torch.unsqueeze(img_tensor, 0).float())
            all_ae_img_tensor.append(torch.unsqueeze(img_ae_tensor, 0).float())

            dict_imgid2idx[sid] = i # it's actually study id to idx here
            # when we load it back, the key of dictionary will automatically transfered to str.
        
        with open('../src/data/data_{}/pytorch_images128_ae.pkl'.format(QT), 'wb') as f:
            cPickle.dump(all_ae_img_tensor, f)
        with open('../src/data/data_{}/pytorch_images{}.pkl'.format(QT, size), 'wb') as f:
            cPickle.dump(all_img_tensor, f)

        with open('../src/data/data_{}/imgid2idx.json'.format(QT), 'w') as f:
            json.dump(dict_imgid2idx, f)
        print('Successfully generated image data')

        all_q = df_qa.question.values.tolist()
        all_a = df_qa.answer.values.tolist()
        all_sid = df_qa.study_id.values.tolist()
        all_qt = df_qa.question_type.values.tolist()
        all_split = df_qa.split.values.tolist()
        qid = 0
        all_sample_train = []
        all_target_train = []
        all_sample_val = []
        all_target_val = []
        all_sample_test = []
        all_target_test = []
        count = 0
        for q, a, sid, qt, sp in zip(all_q, all_a, all_sid, all_qt, all_split):
            if a == 'yes' or a == 'no':
                at = 'yes/no'
            else:
                at = 'other'
            current_sample = {'qid': qid, 'image_name': str(sid), 'answer': a, 'answer_type': at, 'question_type': qt, 'question': q}
            if a in dict_ans2label:
                current_target = {'qid': qid, 'image_name': str(sid), 'labels': [dict_ans2label[a]], 'scores': [1.0]}
            else:
                current_target = {'qid': qid, 'image_name': str(sid), 'labels': [], 'scores': []}
                count += 1
            if sp == 'val':
                all_sample_val.append(current_sample)
                all_target_val.append(current_target)
            elif sp == 'train':
                all_sample_train.append(current_sample)
                all_target_train.append(current_target)
            else:
                all_sample_test.append(current_sample)
                all_target_test.append(current_target)
            qid += 1
        
        print('{} answers not seen before.'.format(count))


        with open('../src/data/data_{}/trainset.json'.format(QT), 'w') as f:
            json.dump(all_sample_train, f)
        with open('../src/data/data_{}/train_target.pkl'.format(QT), 'wb') as f:
            cPickle.dump(all_target_train, f)
        with open('../src/data/data_{}/valset.json'.format(QT), 'w') as f:
            json.dump(all_sample_val, f)
        with open('../src/data/data_{}/val_target.pkl'.format(QT), 'wb') as f:
            cPickle.dump(all_target_val, f)
        with open('../src/data/data_{}/testset.json'.format(QT), 'w') as f:
            json.dump(all_sample_test, f)
        with open('../src/data/data_{}/test_target.pkl'.format(QT), 'wb') as f:
            cPickle.dump(all_target_test, f)
        print('Successfully generated samples and targets')


        # save glove init emeb
        weights, word = utils.create_glove_embedding_init(dictionaryDQA.idx2word, '../src/data/glove/glove.6B.300d.txt')
        np.save('../src/data/data_{}/glove6b_init_300d.npy'.format(QT), weights)
        print('glove6b_init_300d (%d x %d) is generated.' % (weights.shape[0], weights.shape[1]))
