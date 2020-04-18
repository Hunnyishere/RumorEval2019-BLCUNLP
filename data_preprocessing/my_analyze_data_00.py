# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 18:14:07 2019

@author: dell
"""

import numpy as np
from collections import Counter
import os
from my_create_embedding_4 import *


class DataAnalyzer(object):
    def __init__(self):
        # load train & dev data
        self.all_data = {}
        self.all_data['support'] = np.load('data_analysis/support_content.npy')
        self.all_data['deny'] = np.load('data_analysis/deny_content.npy')
        self.all_data['query'] = np.load('data_analysis/query_content.npy')
        self.all_data['comment'] = np.load('data_analysis/comment_content.npy')
        self.all_data['true'] = np.load('data_analysis/true_content.npy')
        self.all_data['false'] = np.load('data_analysis/false_content.npy')
        self.all_data['unverified'] = np.load('data_analysis/unverified_content.npy')
        self.all_labels = ['support','deny','query','comment','true','false','unverified']
        self.all_label_feature_count = {}
        for label in self.all_labels:
            for i,tw in enumerate(self.all_data[label]):
                for j,word in enumerate(tw['text']):
                    self.all_data[label][i]['text'][j] = word.lower()


    def most_frequent_word_count(self,my_label):
        # 统计前300个高频词
        vocab = Counter()
        for tw in self.all_data[my_label]:
            vocab.update(tw['text'])
        for x in vocab.most_common(300):
            print(x[0],x[1])


    def label_word_feature_count(self,my_label):
        label_feature_count = {}
        my_data = GetEmbedding()
        for tw in self.all_data[my_label]:
            my_data.get_sent_features(tw['text'])
            if not label_feature_count.keys():
                for k in my_data.feature_str_dict.keys():
                    label_feature_count[k] = 0
            if not self.all_label_feature_count.keys():
                for k in my_data.feature_str_dict.keys():
                    self.all_label_feature_count[k] = {}
            for k,v in my_data.feature_str_dict.items():
                if v!='None':
                    label_feature_count[k] += 1
        for k,v in label_feature_count.items():
            label_feature_count[k] = v/len(self.all_data[my_label])
            self.all_label_feature_count[k][my_label] = label_feature_count[k]


    def other_word_feature(self):
        self.label_other_word_count = {}
        word_dicts = {'rt':['rt'],'cc':['cc'],'mt':['mt'],'mp':['mp'],'positive_sad':['sad','saddest','cry','crying','cries','cried','disturbing'],
                      'positive_words':['suspect','suspects','suspecting','suspected','end',
                                           'convince','convinces','convinced','obvious','really','exactly',
                                           'believe','believed','believes','credibility','true','truth',
                                           'sure','yes','exactly','agree','good','confirm','confirms','confirmed','fact','facts'],
                      'negative_words':['false','no','not','but','misunderstanding','different',"n't",'fake','smear',
                                        'worst','photoshop','stop','serious','doubt','rather','weird','stupid','nor',
                                        'shame','wrong','lie','lying','lies','lied','crazy','correction','misinformation',
                                        'conspiracy', 'denying'],
                      'swear_words':['abuse','abused','abuses','shut','leave','quit','delete','stfu','sexist','assaulted','assault','assaults'],
                      'query_words':['do','tell','if','is','are','there','right','no','wonder','or',"n't",
                                     'vs.','need','more','better','details','still']
                      }
        for word_feature in word_dicts.keys():
            self.label_other_word_count[word_feature]={}
            for label in self.all_labels:
                self.label_other_word_count[word_feature][label] = 0
        for word_feature in word_dicts.keys():
            for label in self.all_labels:
                for tw in self.all_data[label]:
                    for word in tw['text']:
                        if word in word_dicts[word_feature]:
                            self.label_other_word_count[word_feature][label] += 1
        for word_feature in word_dicts.keys():
            for label,count in self.label_other_word_count[word_feature].items():
                self.label_other_word_count[word_feature][label] = count/len(self.all_data[label])



if __name__ == "__main__":
    my_analyzer = DataAnalyzer()
    # 啊啊啊啊所有数据并没有转成小写！天哪！
    '''
    # 已有的词级别特征
    for label in my_analyzer.all_labels:
        my_analyzer.label_word_feature_count(label)
    for k in my_analyzer.all_label_feature_count.keys():
        print(k,my_analyzer.all_label_feature_count[k])
    '''
    # 尝试的新词级别特征
    my_analyzer.other_word_feature()
    for k,v in my_analyzer.label_other_word_count.items():
        print(k,v)

    np.save('data_analysis/all_labels_word_feature_count',my_analyzer.all_label_feature_count)