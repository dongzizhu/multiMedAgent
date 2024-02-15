############################################################################################
###
### This code is adapted from https://github.com/aioz-ai/MICCAI21_MMQ
###
############################################################################################
import argparse
from ast import arg
from email import message
from pickletools import read_uint1
import torch
from torch.utils.data import DataLoader
import dataset_VQA
import base_model
import utils
import pandas as pd
import os
import json
import numpy as np
import time
import pickle

import openai
openai.api_key = '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'

# load abnormality detection
DICT_ab = json.load(open('saved_models/sorted_abnormality_detection_256.json'))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ensemble', type=bool, default=False,
                        help='ensemble flag. If True, generate a logit file which is used in the ensemble part')
    # MODIFIABLE MEVF HYPER-PARAMETERS--------------------------------------------------------------------------------
    # Model loading/saving
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--input', type=str, default='saved_models/SAN_MEVF',
                        help='input file directory for loading a model')
    parser.add_argument('--output', type=str, default='results',
                        help='output file directory for saving VQA answer prediction file')
    # Utilities
    parser.add_argument('--epoch', type=str, default=19,
                        help='the best epoch')

    # Gradient accumulation
    parser.add_argument('--batch_size', type=int, default=1,
                        help='batch size')

    # Choices of Attention models
    parser.add_argument('--model', type=str, default='SAN', choices=['BAN', 'SAN'],
                        help='the model we use')

    # Choices of RNN models
    parser.add_argument('--rnn', type=str, default='LSTM', choices=['LSTM', 'GRU'],
                        help='the RNN we use')

    # BAN - Bilinear Attention Networks
    parser.add_argument('--gamma', type=int, default=2,
                        help='glimpse in Bilinear Attention Networks')
    parser.add_argument('--use_counter', action='store_true', default=False,
                        help='use counter module')

    # SAN - Stacked Attention Networks
    parser.add_argument('--num_stacks', default=2, type=int,
                        help='num of stacks in Stack Attention Networks')

    # Utilities - gpu
    parser.add_argument('--gpu', type=int, default=0,
                        help='specify index of GPU using for training, to use CPU: -1')

    # Question embedding
    parser.add_argument('--op', type=str, default='c',
                        help='concatenated 600-D word embedding')

    # Joint representation C dimension
    parser.add_argument('--num_hid', type=int, default=1024,
                        help='dim of joint semantic features')

    # Activation function + dropout for classification module
    parser.add_argument('--activation', type=str, default='relu', choices=['relu'],
                        help='the activation to use for final classifier')
    parser.add_argument('--dropout', default=0.5, type=float, metavar='dropout',
                        help='dropout of rate of final classifier')

    # Train with RAD
    parser.add_argument('--use_VQA', action='store_true', default=False,
                        help='Using TDIUC dataset to train')
    parser.add_argument('--VQA_dir', type=str,
                        help='RAD dir')

    # Optimization hyper-parameters
    parser.add_argument('--eps_cnn', default=1e-5, type=float, metavar='eps_cnn',
                        help='eps - batch norm for cnn')
    parser.add_argument('--momentum_cnn', default=0.05, type=float, metavar='momentum_cnn',
                        help='momentum - batch norm for cnn')

    # input visual feature dimension
    parser.add_argument('--feat_dim', default=32, type=int,
                        help='visual feature dim')
    parser.add_argument('--img_size', default=256, type=int,
                        help='image size')
    # Auto-encoder component hyper-parameters
    parser.add_argument('--autoencoder', action='store_true', default=False,
                        help='End to end model?')
    parser.add_argument('--ae_model_path', type=str, default='pretrained_ae.pth',
                        help='the maml_model_path we use')

    # MAML component hyper-parameters
    parser.add_argument('--maml', action='store_true', default=False,
                        help='End to end model?')
    parser.add_argument('--maml_model_path', type=str, default='pretrained_maml_pytorch_other_optimization_5shot_newmethod.pth',
                        help='the maml_model_path we use')
    parser.add_argument('--maml_nums', type=str, default='0,1,2,3,4,5',
                        help='the numbers of maml models')
    
    # data loading option
    parser.add_argument('--load', action='store_true', default=False,
                        help='Do we load the data matrix?')

    # Return args
    args = parser.parse_args()
    return args
# Load questions
def get_question(q, dataloader):
    q = q.squeeze(0)
    str = []
    dictionary = dataloader.dataset.dictionary
    for i in range(q.size(0)):
        str.append(dictionary.idx2word[q[i]] if q[i] < len(dictionary.idx2word) else '_')
    return ' '.join(str)

# Load answers
def get_answer(p, label2ans):
    _m, idx = p.max(1)
    # print(idx)
    # print(len(label2ans))
    return [label2ans[x.item()] for x in idx]


def my_get_answer(model, dataset, q, v, device, qt):
    img_num = v[0].shape[0]

    with torch.no_grad():
        v[0] = v[0][:-1].to(device)
        v[1] = v[1][:-1].to(device)
        q = q[:-1].to(device)

        # inference and get logit
        features, _ = model(v, q)
        preds = model.classifier(features)
        final_preds = preds

        return get_answer(final_preds, dataset.label2ans)




def tokenize(question, dictionary, max_length=12):
    """Tokenizes the questions.

    This will add q_token in each entry of the dataset.
    -1 represent nil, and should be treated as padding_idx in embedding
    """
    tokens = dictionary.tokenize(question, False)
    tokens = tokens[:max_length]
    if len(tokens) < max_length:
        # Note here we pad in front of the sentence
        padding = [dictionary.padding_idx] * (max_length - len(tokens))
        tokens = tokens + padding
    utils.assert_eq(len(tokens), max_length)
    return tokens



def promptGPT(imageID, refID, DQ, previous=None):
    if previous is None:
        prompt = "You are a radiologist trying to answer questions that pertain to the clinical progress and changes in the main image as compared to the reference image. \
            I will give you example answers in the format of question-answer pairs. \
            'what has changed compared to the reference image? the main image has an additional finding of pneumothorax than the reference image. the main image is missing the findings of fracture, lung opacity, and pleural effusion than the reference image.',\
            'what has changed compared to the reference image? the main image is missing the findings of lung opacity, consolidation, pleural effusion, and pleural thickening than the reference image.', \
            'what has changed in the right lung area? the level of pleural effusion has changed from small to moderate.'\
            'what has changed compared to the reference image? nothing has changed.'\
            You can ask questions about both images to gather information. These are six types of questions you can ask. \
            You can ask about the abnormalities in different images with the key being 'abnormality', like 'what abnormalities are seen in this image?', 'what abnormalities are seen in the upper lungs?'. \
            You can ask about the presence of a certain abnormality in an image with the key being 'presence', like 'is there evidence of atelectasis in this image?', 'is there edema?'. \
            You can ask about the level of a certain abnormality in an image with the key being 'level', like 'what level is the cardiomegaly?', 'what level is the pneumothorax?'.\
            You can ask about the location of a certain abnormality in a image with the key being 'location', like 'where in the image is the pleural effusion located?', 'is the atelectasis located on the left side or right side?'.\
            You can ask about the type of a certain abnormality in a image with the key being 'type', like 'what type is the opacity?', 'what type is the atelectasis?'.\
            Give me your questions one at a time about any of the images in the format of a Python dictionary with keys recording the imageID, question_type and question_content. Only return the Python dictionary.\
            In order answer this question for a main image with ID 000A with reference image ID 000B: {} ? You can start asking questions now.\
            Do not ask repeated questions, and ask as less questions as possible. You should stop asking questions once you have enough information. \
            Only reply with the difference when you answer the question. No explanation needed.".format(DQ)
        message = [{"role": "user", "content": prompt}]
        try:
            chat_completion = openai.ChatCompletion.create(model="gpt-4-1106-preview", messages=message, temperature=0.2)
        except:
            # sometime the API will fail for no reason
            chat_completion = openai.ChatCompletion.create(model="gpt-4-1106-preview", messages=message, temperature=0.2)
        return chat_completion['choices'][0]['message'], message
    
    else:
        comb_messages, previous_assistant, previous_mentor = previous
        if isinstance(previous_mentor, list):
            ans_list, p_list = previous_mentor
            temp_message = ["the probability of the anwser being '{}' is {}".format(a, round(p.item(), 3)) for a, p in zip(ans_list, torch.softmax(p_list, 0))]
            temp_message = ', '.join(temp_message)
            p_m = {"role": "user", "content": temp_message}
        else: 
            p_m = {"role": "user", "content": previous_mentor}
        print(p_m['content'])
        p_a = {"role": previous_assistant['role'], "content": previous_assistant['content']}
        comb_messages.append(p_a)
        comb_messages.append(p_m)
        try:
            chat_completion = openai.ChatCompletion.create(model="gpt-4-1106-preview", messages=comb_messages, temperature=0.2)
        except:
            # sometime the API will fail for no reason
            time.sleep(120)
            chat_completion = openai.ChatCompletion.create(model="gpt-4-1106-preview", messages=comb_messages, temperature=0.2)
        return chat_completion['choices'][0]['message'], comb_messages


def deal_with_ab(t_ab):
    if len(t_ab) == 0 or t_ab[0] == 'no finding':
        return ['no finding']
    else:
        if 'no finding' in t_ab:
            idx = t_ab.index('no finding')
            _ = t_ab.pop(idx)
        return t_ab[:3]


# Testing process
def process(args, dataset, model, question, imageID, type_dataset, quesntion_type):
    model.train(False)

    # abnormality detection
    if question.lower() == 'what abnormalities are seen in this image?':
        print('Simple detection for {}'.format(imageID))
        current_ab = deal_with_ab(DICT_ab[imageID])

        return ', '.join(current_ab)

    question = torch.from_numpy(np.array(tokenize(question, type_dataset.dictionary)))
    
    batch_v0 = torch.vstack([dataset.maml_images_data[dataset.entries[i]['image']].reshape(args.img_size * args.img_size) for i in range(args.batch_size-1)])
    batch_v1 = torch.vstack([dataset.ae_images_data[dataset.entries[i]['image']].reshape(128 * 128) for i in range(args.batch_size-1)])

    batch_q = np.vstack([type_dataset.entries[i]['q_token'] for i in range(args.batch_size-1)])
    batch_q = torch.from_numpy(batch_q)

    image = [torch.cat([dataset.maml_images_data[dataset.img_id2idx[imageID]], batch_v0.reshape(batch_v0.shape[0], 1, args.img_size, args.img_size)]), 
             torch.cat([dataset.ae_images_data[dataset.img_id2idx[imageID]], batch_v1.reshape(batch_v1.shape[0], 128, 128).unsqueeze(1)])]
    question = torch.cat([question.reshape((-1, question.shape[0])), batch_q])
    answers_list = my_get_answer(model, type_dataset, question, image, args.device, quesntion_type)
    return answers_list[0]



# Test phase
if __name__ == '__main__':
    start_time = time.time()
    args = parse_args()
    args.maml_nums = args.maml_nums.split(',')
    torch.backends.cudnn.benchmark = True
    args.device = torch.device("cuda:" + str(args.gpu) if args.gpu >= 0 else "cpu")
    batch_size = args.batch_size

    # load model for all
    dictionary = dataset_VQA.Dictionary.load_from_file(os.path.join(args.VQA_dir, 'dictionary.pkl'))
    eval_dset_all = dataset_VQA.VQAFeatureDataset(args.split, args, dictionary)

    # load mentor models
    mentors = ['abnormality', 'presence', 'level', 'location', 'type']
    mentor_models = {}
    for mentor in mentors:
        print('loading model for {}'.format(mentor))
        data_path = 'data/data_' + mentor
        model_input = 'saved_models/MMQ_BAN_MEVF_' + mentor
        args.VQA_dir = data_path
        args.load = False
        dictionary = dataset_VQA.Dictionary.load_from_file(os.path.join(data_path, 'dictionary.pkl'))
        eval_dset = dataset_VQA.VQAFeatureDataset(args.split, args, dictionary)
        constructor = 'build_%s' % args.model
        model = getattr(base_model, constructor)(eval_dset, args)

        model_path = model_input + '/model_epoch%s.pth' % args.epoch
        print('loading %s' % model_path)
        model_data = torch.load(model_path)

        # Comment because do not use multi gpu
        # model = nn.DataParallel(model)
        model = model.to(args.device)
        model.load_state_dict(model_data.get('model_state', model_data))
        mentor_models[mentor] = [model, eval_dset]
        print(len(eval_dset.ans2label))
    
    # retrieve difference question
    N = 5
    limit = 20
    data_root = '/users/PCON0023/dzz2023/VQA/EKAID/model/data'
    path_qa = os.path.join(data_root, 'mimic_pair_questions.csv')
    df_qa = pd.read_csv(path_qa)

    df_qa = df_qa[(df_qa.question_type == 'difference') & (df_qa.split == 'test')].tail(N)
    

    res = {'gt': [], 'prediction': []}
    conversations = []
    for sid, rid, difference_question, gt_answer in zip(df_qa.study_id, df_qa.ref_id, df_qa.question, df_qa.answer):
        current_conv = []
        # skip images not seen in other type of questions for now
        if str(sid) not in eval_dset_all.img_id2idx:
            print('not included', sid)
            continue
        if str(rid) not in eval_dset_all.img_id2idx:
            print('not included', rid)
            continue

        # Prompt GPT to get an question
        print('main image: {}, reference image: {}'.format(sid, rid))
        print('question: ', difference_question)
        r_GPT, last_message = promptGPT(sid, rid, difference_question)

        count = 1
        while '{' in r_GPT['content'].replace("\'", "\""):
            dict_str = r_GPT['content'].replace("\'", "\"")
            idx_s = dict_str.find('{')
            idx_e = dict_str.find('}')

            count += 1
            try:
                current_r_GPT = json.loads(dict_str[idx_s:idx_e+1].replace('000A', str(sid)).replace('000B', str(rid)))
            except:
                print(dict_str[idx_s:idx_e+1].replace('000A', str(sid)).replace('000B', str(rid)))
                break
            print(current_r_GPT.values())
            if 'question_content' not in current_r_GPT:
                print(current_r_GPT)
                break
            if current_r_GPT['question_type'] == 'comparison' or current_r_GPT['question_type'] == 'difference':
                print('Breaking because of wrong question type...')
                break
            current_conv.append(current_r_GPT)
            if count < limit:
                qt = current_r_GPT['question_type']
                ans = process(args, eval_dset_all, mentor_models[qt][0], current_r_GPT['question_content'], str(current_r_GPT['imageID']), mentor_models[qt][1], qt)
                current_conv.append(ans)
                r_GPT, last_message = promptGPT(sid, rid, difference_question, [last_message, r_GPT, ans])
            if count > limit:
                ans = process(args, eval_dset_all, mentor_models[qt][0], current_r_GPT['question_content'], str(current_r_GPT['imageID']), mentor_models[qt][1], qt)
                current_conv.append(ans)
                ans = ans + ' Now you have enough information, please answer this question: ' + difference_question
                r_GPT, last_message = promptGPT(sid, rid, difference_question, [last_message, r_GPT, ans])
                break
        if count > limit:
            print('prediction (breaking): ', r_GPT['content'])
        else:
            print('prediction: ', r_GPT['content'].replace("\'", "\""))
        res['gt'].append(gt_answer)
        res['prediction'].append(r_GPT['content'].replace("\'", "\""))
        print('ground truth: ', gt_answer)
        conversations.append(current_conv)

        if len(conversations) % 100 == 0:
            pickle.dump(conversations, open('conversations_test_{}.pkl'.format(N), 'wb'))
            df_res = pd.DataFrame.from_dict(res)
            df_res.to_csv('res_test_{}.csv'.format(N,N+1500), index=False)
    
    pickle.dump(conversations, open('conversations_test_{}.pkl'.format(N), 'wb'))
    df_res = pd.DataFrame.from_dict(res)
    df_res.to_csv('res_test_{}.csv'.format(N), index=False)

    end_time = time.time()
    print('It took: {}s'.format(end_time-start_time))
