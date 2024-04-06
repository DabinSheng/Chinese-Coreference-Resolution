import copy
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import warnings
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
from dataset import CR_Dataset
import sklearn
import re
import os
import json
import random
import sklearn.metrics
import seaborn as sns
import nltk
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix,precision_recall_curve
from copy import deepcopy
from imblearn.over_sampling import RandomOverSampler
from torchsummary import summary
warnings.filterwarnings("ignore")
def cr_dataloader(train_data,weights_of_distance,weights_of_position,weight_of_feature,weights_of_pronoun,samp_single,feature_df,batch_size):
    train_samp_wide=[]
    for t in tqdm(train_data,desc="Data loading"):
        # 有NULL文件，特判
        try:
            # 遍历各个pairs
            for mkey in range(t["antecedentNum"]):
                # 键值是str，作为索引需要把数字转为str
                str_key = str(mkey)
                # 先行词和代词的句子id
                n1 = t[str_key]["id"]
                n2 = t["pronoun"]["id"]
                # 有的任务句子号不存在，需要特判
                try:
                    between_words = copy.deepcopy(feature_df.loc[n1:n2])  # 存储这组任务涉及的所有句子，可能多于两句
                    # f存储这段话的特征
                    f = np.empty(5205)
                    for index, row in between_words.iterrows():
                        f1 = row[0:5205].to_numpy()  # 累加句子的特征
                        # print(f1)
                        f = f1 + f


                    # print(f)
                    # 下面这段代码检验序号和词语是不是对得上，经检验洗词效果良好
                    test_key = (samp_single.loc[n1]["sec"])
                    test_pro = (samp_single.loc[n2]["sec"])
                    # print(t[str_key]["id"])
                    # print(test_key[t[str_key]["indexFront"]:t[str_key]["indexBehind"] + 1])
                    # print("____")
                    # print(t["pronoun"]["id"])
                    # print(test_pro[t["pronoun"]["indexFront"]:t["pronoun"]["indexBehind"] + 1])
                    # print("")

                    # 代词分类添加：分成他，他们，她，她们，它，它们并将成分混入第二维向量
                    pronoun_words=str(test_pro[t["pronoun"]["indexFront"]:t["pronoun"]["indexBehind"] + 1])
                    # print(pronoun_words)
                    if pronoun_words=="他":
                        vec_p=np.full((5205,), 5)
                    elif pronoun_words == "他们":
                        vec_p = np.full((5205,), 4)
                    elif pronoun_words=="她":
                        vec_p=np.full((5205,), 3)
                    elif pronoun_words == "她们":
                        vec_p = np.full((5205,), 2)
                    elif pronoun_words=="它":
                        vec_p=np.full((5205,), 1)
                    elif pronoun_words == "它们":
                        vec_p = np.full((5205,), -1)
                    else:
                        vec_p = np.full((5205,), -2)
                    f = f+weights_of_pronoun*vec_p
                    f *= weight_of_feature
                    n_feature_f = copy.deepcopy(f)
                    # 先处理不在同一个句子的情况,经检验可以间隔两个以上的句子
                    if t[str_key]["id"] != t["pronoun"]["id"]:
                        # 计算两句子长度
                        between_line = samp_single.loc[n1:n2]
                        # 长度
                        total_words = 0
                        for index, row in between_line.iterrows():
                            # print(row["sec"])
                            total_words += len(row["sec"])



                        # 先行词和代词所在句子的长度
                        s1 = len(samp_single.loc[n1, 'sec'])
                        s2 = len(samp_single.loc[n2, 'sec'])
                        # 先行词的平均位置
                        p1 = int((t[str_key]["indexFront"] + t[str_key]["indexBehind"]) / 2)
                        # 代词的平均位置
                        p2 = int((t["pronoun"]["indexFront"] + t["pronoun"]["indexBehind"]) / 2)
                        # 判断先行词和代词谁在前边
                        index_pos_antecedent = samp_single.index.get_loc(t[str_key]["id"])
                        index_pos_pronoun = samp_single.index.get_loc(t["pronoun"]["id"])
                        # 根据先后的不同分别计算先行词和代词的距离和他们在这一整段话（可以有多个句子）里的索引
                        index_ant_pos = 0
                        index_pro_pos = 0
                        # 记录先行词和代词谁在前
                        fb_flag = 0
                        # 代词在前边
                        if index_pos_antecedent > index_pos_pronoun:
                            distance = total_words - p2 - (s1 - p1)
                            index_pro_pos = p2
                            index_ant_pos = total_words - p1
                            fb_flag = 1
                        else:
                            distance = total_words - p1 - (s2 - p2)
                            index_pro_pos = total_words - p2
                            index_ant_pos = p1
                        # distance为负说明有部分标点没统计进去，这种情况先行词和代词很近，直接把distance设为0
                        if distance < 0:
                            # print(t["taskID"])
                            # print(distance)
                            distance = 0
                        # 混入距离因素
                        f += weights_of_distance * distance
                        # 求位置因素
                        factor_position = math.sin(index_pro_pos) + math.cos(index_ant_pos)
                        # 混入位置因素
                        f += weights_of_position * factor_position
                        # 处理正例
                        train_samp_wide.append({'feature': f, 'label': 1})
                        # 处理负例
                        # 先行词在前代词在后
                        if fb_flag == 0:
                            for n_position in range(index_pro_pos):
                                n_feature = copy.deepcopy(n_feature_f)
                                n_distance = index_pro_pos - n_position
                                n_feature += weights_of_distance * n_distance
                                n_factor_position = math.sin(index_pro_pos) + math.cos(n_position)
                                n_feature += weights_of_position * n_factor_position
                                train_samp_wide.append({'feature': n_feature, 'label': 0})
                        # 代词在前先行词在后
                        else:
                            for n_position in range(index_pro_pos, total_words):
                                n_feature = copy.deepcopy(n_feature_f)
                                n_distance = n_position - index_pro_pos
                                n_feature += weights_of_distance * n_distance
                                n_factor_position = math.sin(index_pro_pos) + math.cos(n_position)
                                n_feature += weights_of_position * n_factor_position
                                train_samp_wide.append({'feature': n_feature, 'label': 0})


                    # 在同一个句子
                    else:

                        between_line = deepcopy(samp_single.loc[n1:n2])  # 存储这组任务涉及的所有句子，可能多于两句
                        total_words = 0
                        for index, row in between_line.iterrows():
                            # print(row["sec"])
                            total_words += len(row["sec"])

                        # 先行词的平均位置
                        p1 = int((t[str_key]["indexFront"] + t[str_key]["indexBehind"]) / 2)
                        # 代词的平均位置
                        p2 = int((t["pronoun"]["indexFront"] + t["pronoun"]["indexBehind"]) / 2)
                        # 判断先行词和代词谁在前边
                        index_pos_antecedent = samp_single.index.get_loc(t[str_key]["id"])
                        index_pos_pronoun = samp_single.index.get_loc(t["pronoun"]["id"])
                        # 根据先后的不同分别计算先行词和代词的距离和他们在这一段话里的索引
                        index_ant_pos = 0
                        index_pro_pos = 0
                        fb_flag = 0
                        # 代词在前边
                        if p2 > p1:
                            distance = abs(p1 - p2)
                            index_pro_pos = p2
                            index_ant_pos = p1
                            fb_flag = 1
                        else:
                            distance = abs(p1 - p2)
                            index_pro_pos = p2
                            index_ant_pos = p1
                        # 说明有部分标点没统计进去造成的负值，这种情况先行词和代词很近，直接把distance设为0
                        if distance < 0:
                            # print(t["taskID"])
                            # print(distance)
                            distance = 0
                        # 混入距离因素
                        f += weights_of_distance * distance
                        # 求位置因素
                        factor_position = math.sin(index_pro_pos) + math.cos(index_ant_pos)
                        # 混入位置因素
                        f += weights_of_position * factor_position
                        # 处理正例
                        train_samp_wide.append({'feature': f, 'label': 1})
                        # 处理负例
                        # 先行词在前代词在后
                        if fb_flag == 0:
                            for n_position in range(index_pro_pos):
                                n_feature = copy.deepcopy(n_feature_f)
                                n_distance = index_pro_pos - n_position
                                n_feature += weights_of_distance * n_distance
                                n_factor_position = math.sin(index_pro_pos) + math.cos(n_position)
                                n_feature += weights_of_position * n_factor_position
                                train_samp_wide.append({'feature': n_feature, 'label': 0})
                        # 代词在前先行词在后
                        else:
                            for n_position in range(index_pro_pos, total_words):
                                n_feature = copy.deepcopy(n_feature_f)
                                n_distance = n_position - index_pro_pos
                                n_feature += weights_of_distance * n_distance
                                n_factor_position = math.sin(index_pro_pos) + math.cos(n_position)
                                n_feature += weights_of_position * n_factor_position
                                train_samp_wide.append({'feature': n_feature, 'label': 0})
                except KeyError:
                    print(t["taskID"]+"任务的句子编号"+t["pronoun"]["id"]+"不存在")
                    pass

        except TypeError:
            print("有任务对应的json文件是空的（NULL）")
            pass
    # 生成dataset
    train_dataset = CR_Dataset(train_samp_wide)
    # 负例远大于正例，进行过采样
    # 计算总样本数量
    num_samples = len(train_dataset)
    # 统计0和1的数量
    num_class_0=0
    for sample in train_dataset:
        # 注意，dataset类型的元素是一个元组，不像dataframe的元素一样可以用str作为索引，访问它第二列的label属性只能用数字索引
        if sample[1]==0:
            num_class_0+=1
    num_class_1 = 0
    for sample in train_dataset:
        if sample[1] == 1:
            num_class_1 += 1

    # 计算每个类别的权重，使得权重和样本数量的比例大致相等
    weight_class_0 = 1 / (2 * num_class_0)
    weight_class_1 = 1 / (2 * num_class_1)

    # 创建样本权重列表，根据每个样本的类别给出相应的权重
    weights = [weight_class_0 if sample[1] == 0 else weight_class_1 for sample in train_dataset]
    # 生成采样器
    sampler = WeightedRandomSampler(weights, num_samples=num_samples, replacement=True)
    # 生成dataloader,注意用了采样器参数sampler就不能用shuffle的随机取样参数了
    train_loader = DataLoader(train_dataset, batch_size=batch_size,sampler=sampler)
    return train_loader


def cr_dataloaderbase():
    # Data Loading
    f = open('words.txt', encoding='gbk')
    sen_words = f.readlines()
    f.close()
    temp_words = deepcopy(sen_words)
    # 去空行后的文本存在sen_words
    for i in temp_words:
        if i == '\n':
            sen_words.remove(i)
    # 提取索引
    indexcol = []
    for i in range(len(sen_words)):
        if i == 16661:  # Since there is a 't' in the sentence
            indexcol.append(sen_words[i][1:20])
        else:
            indexcol.append(sen_words[i][0:19])
    # 参考了以下范例代码
    # # 去掉特殊字符
    # sen_wordsnew1 = []
    # for i in sen_words:
    #     str1 = re.sub('[a-zA-Z0-9’!"#$%&\'()*+,-.:;<=>?@，。?〓★、…【】{}《》？“”‘’！１０９２（）．—３５：７４８６；[\\]^_`{|}~\s]+', "", i)
    #     sen_wordsnew1.append(str1)
    # # 替换连续的//为/
    # sen_wordsnew = []
    # for i in sen_wordsnew1:
    #     str1 = re.sub('//', '/', i)
    #     sen_wordsnew.append(str1[1:-1])
    # # 分词后存在samp，分词符号为, 这道题是按照源数据库给的两词序号，不能用jieba，得按北大的库分
    # samp = []
    # for sentence in sen_wordsnew:
    #     sentence = sentence.replace('/',',')
    #     seg_line=sentence.split(',')
    #     seg_line=[w for w in seg_line if w]  # 去空字符
    #     samp.append(seg_line)

    # 任务的词语索引编号时考虑了标点符号，要做一遍不去标点的操作(single后缀表示不去除标点)

    # 取开头序号
    sen_words_without_id = []
    for i in range(len(sen_words)):
        str2 = sen_words[i].replace(indexcol[i], "")
        sen_words_without_id.append(str2)
    # 去特殊字符但保留标点
    sen_wordsnew1_single = []
    for i in sen_words_without_id:
        str3 = re.sub('[a-zA-Z0-9’!"#$&\'()*+-.<=>@〓★【】{}‘’[\\]^_`{|}~\s]+', "", i)
        sen_wordsnew1_single.append(str3)
    # 替换//
    sen_wordsnew_single = []
    for i in sen_wordsnew1_single:
        str4 = re.sub('//', '/', i)
        sen_wordsnew_single.append(str4)
    # 分词
    samp_single = []
    for sentence in sen_wordsnew_single:
        sentence = sentence.replace('/', '#')
        seg_line = sentence.split('#')
        seg_line = [w for w in seg_line if w]  # 去空字符
        samp_single.append(seg_line)
    # 按词性提取
    seg_sen = []
    seg_sen = [sentence.split() for sentence in sen_words]
    # 名词提取
    key = []
    for i in seg_sen:
        for j in i:
            m = re.findall('[\u4e00-\u9fff].*?(?=/n)', j)
            m2 = re.findall('[\u4e00-\u9fff].*?(?=/nrf)', j)
            m3 = re.findall('[\u4e00-\u9fff].*?(?=/nt)', j)
            m4 = re.findall('[\u4e00-\u9fff].*?(?=/nz)', j)
            m5 = re.findall('[\u4e00-\u9fff].*?(?=/ns)', j)
            m6 = re.findall('[\u4e00-\u9fff].*?(?=/nrg)', j)
            if 0 < len(m) <= 4:
                key.append(m)
            elif 0 < len(m2) <= 4:
                key.append(m2)
            elif 0 < len(m3) <= 4:
                key.append(m3)
            elif 0 < len(m4) <= 4:
                key.append(m4)
            elif 0 < len(m5) <= 4:
                key.append(m5)
            elif 0 < len(m6) <= 4:
                key.append(m6)
            else:
                continue

    key_words = []
    for i in key:
        for j in i:
            key_words.append(j)
    # 代词提取
    key_r = []
    for i in seg_sen:
        for j in i:
            m = re.findall('[\u4e00-\u9fff].*?(?=/r)', j)
            m2 = re.findall('[\u4e00-\u9fff].*?(?=/rr)', j)
            m3 = re.findall('[\u4e00-\u9fff].*?(?=/ry)', j)
            m4 = re.findall('[\u4e00-\u9fff].*?(?=/ryw)', j)
            m5 = re.findall('[\u4e00-\u9fff].*?(?=/rz)', j)
            m6 = re.findall('[\u4e00-\u9fff].*?(?=/rzw)', j)
            if 0 < len(m) <= 4:
                key_r.append(m)
            elif 0 < len(m2) <= 4:
                key_r.append(m2)
            elif 0 < len(m3) <= 4:
                key_r.append(m3)
            elif 0 < len(m4) <= 4:
                key_r.append(m4)
            elif 0 < len(m5) <= 4:
                key_r.append(m5)
            elif 0 < len(m6) <= 4:
                key_r.append(m6)
            else:
                continue

    key_words_r = []
    for i in key_r:
        for j in i:
            key_words_r.append(j)
    # 词频统计
    keywords = []
    counts = nltk.FreqDist(key_words)
    counts2 = nltk.FreqDist(key_words_r)
    item = list(counts.items())
    item2 = list(counts2.items())
    item.sort(key=lambda x: x[1], reverse=True)
    item2.sort(key=lambda x: x[1], reverse=True)
    for sen in item[:4900]:
        keywords.append(sen[0])
    for sen in item2[:310]:
        keywords.append(sen[0])
    keywords_res = []
    for i in keywords:
        if i not in keywords_res:
            keywords_res.append(i)
    # print(keywords_res[-1])
    # # 去掉最后一个元素 不知道源码为什么这么做
    # keywords_res = keywords_res[:-1]
    # 独热编码
    # 基于词频建立词袋
    bow_feature = CountVectorizer(vocabulary=keywords_res)  # 词袋
    # dict3是把samp_single内层的list结构换成了str，这样才能调用词袋方法
    dic3 = []
    for s in samp_single:
        dic3.append(','.join(s))
    # 独热编码生成每句话的特征
    wordsfeature = bow_feature.fit_transform(dic3).toarray()
    # 特征矩阵转dadaframe并标记索引
    feature_df = pd.DataFrame(wordsfeature)
    feature_df['index'] = indexcol
    feature_df = feature_df.set_index('index')

    # # 生成dataframe格式的samp
    # tmp_w=[]
    # c=0;
    # for s in samp:
    #     tmp_w.append({'no':indexcol[c], 'sec':s})
    #     c+=1
    # samp=tmp_w
    # samp=pd.DataFrame(samp)
    # samp['index'] = indexcol
    # samp = samp.set_index('index')

    # 生成dataframe格式的samp_single
    tmp1_ws = []
    c = 0;
    for s in samp_single:
        tmp1_ws.append({'no': indexcol[c], 'sec': s})
        c += 1
    samp_single = tmp1_ws
    samp_single = pd.DataFrame(samp_single)
    samp_single['index'] = indexcol
    samp_single = samp_single.set_index('index')


    # 读取数据集
    directory = 'CR_Data/train'
    train_data = []
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r',encoding='gb18030') as f:
                data = json.load(f)
                train_data.append(data)
    directory = 'CR_Data/validation'
    validation_data = []
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r',encoding='gb18030') as f:
                data = json.load(f)
                validation_data.append(data)
    directory = 'CR_Data/test'
    test_data = []
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r',encoding='gb18030') as f:
                data = json.load(f)
                test_data.append(data)

    return train_data,validation_data,test_data,samp_single,feature_df