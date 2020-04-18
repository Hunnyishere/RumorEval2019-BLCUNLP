# -*- coding: utf-8 -*-

import numpy as np
import os
from my_create_embedding_4 import *
import random
import re


class SplitText(object):
    def __init__(self):
        self.max_p_length = 0  # 1582->1495
        self.max_h_length = 0  # 1163->1107
        self.all_p_length = 0
        self.all_h_length = 0
        self.p_num = 0  # 仅train切分：7197 = 6700+795-300
        # train: premise=hypothesis=labels=5712 (切前5217), dev:  premise=hypothesis=labels=1485

        self.A_stop_p_length = 180  # 仅切分h，未切分p：180:94.8%, 200:96.1%
        self.A_stop_h_length = 32  # 32:95.6%, 35:96.3%, 40:96.8%
        self.B_stop_p_length = 70  # 90%: 按照branch(3560+966):120，一条source仅一次(327+38):70
        self.B_stop_h_length = 28  # 19:54.2%,25:89.3%,28:94.5%,32:99.2%(太高了）
        self.overlen_p_num = 0
        self.overlen_h_num = 0
        self.overlen_h_new = 0

        # 处理成 src + src特征 + 可变长同branch其他tw + reply N-1 + reply N-1特征 + reply N + reply N特征 的格式
        self.all_length = 512  # 之前是450
        self.delimiter_num = 8
        self.feature_dict = ['favorite','retweet','depth',
                             'user_verified','user_entities_description',
                             'user_entities_url','user_followers',
                             'user_listed','user_statuses','user_description',
                             'user_friends','user_default_profile',
                             'selftext','kind',
                             'archived','edited','is_submitter']
        self.int_feature_dict = ['favorite','retweet','depth','user_followers',
                    'user_listed','user_statuses','user_friends']
        self.str_feature_dict = ['user_verified','user_entities_description',
                    'user_entities_url','user_description',
                    'user_default_profile','selftext','kind','archived',
                    'edited','is_submitter']


    def taskA_to_GPT(self):
        print("\nstarting task A:")
        id_set = {}
        id_set['train'] = []
        id_set['dev'] = []
        id_set['test'] = []
        # 纯文本
        sources = {}
        sources['train'] = []
        sources['dev'] = []
        sources['test'] = []
        branch_other_tws = {}
        branch_other_tws['train'] = []
        branch_other_tws['dev'] = []
        branch_other_tws['test'] = []
        last_tws = {}
        last_tws['train'] = []
        last_tws['dev'] = []
        last_tws['test'] = []
        this_tws = {}
        this_tws['train'] = []
        this_tws['dev'] = []
        this_tws['test'] = []
        labels = {}
        labels['train'] = []
        labels['dev'] = []
        labels['test'] = []
        # 带特征：source, last tw, this tw
        featured_sources = {}
        featured_sources['train'] = []
        featured_sources['dev'] = []
        featured_sources['test'] = []
        featured_last_tws = {}
        featured_last_tws['train'] = []
        featured_last_tws['dev'] = []
        featured_last_tws['test'] = []
        featured_this_tws = {}
        featured_this_tws['train'] = []
        featured_this_tws['dev'] = []
        featured_this_tws['test'] = []
        whichset = ['train', 'dev', 'test']
        for sset in whichset:
            branches_text_in_one_set = np.load('data/taskA_newformdata/' + sset + '/cleaned_tokenized_branch_A_B_label_stanford.npy')
            print(len(branches_text_in_one_set))
            for j, branch in enumerate(branches_text_in_one_set):
                source = branch[0]['text']
                featured_source = branch[0]
                source_len = len(source)
                for i, tweet in enumerate(branch):
                    id_str = tweet['id']
                    this_tw = tweet['text']
                    featured_this_tw = tweet
                    #print(featured_this_tw['favorite'])
                    if i<=1:  # source tweet + 第一条
                        last_tw = []
                        featured_last_tw = {}
                        other_tw = []
                    elif i==2:
                        last_tw = branch[1]['text']
                        featured_last_tw = branch[1]
                        other_tw = []
                    else:
                        last_tw = branch[i - 1]['text']
                        featured_last_tw = branch[i - 1]
                        other_tw = []
                        for branch_tw in branch[1:i - 1]:
                            other_tw.extend(branch_tw['text'])
                    last_tw_len = len(last_tw)
                    this_tw_len = len(this_tw)
                    other_tw_len = len(other_tw)
                    if id_str not in id_set[sset]:
                        if source_len + this_tw_len >= (self.all_length - self.delimiter_num):
                            id_set[sset].append(id_str)
                            sources[sset].append(source)
                            featured_sources[sset].append(featured_source)
                            last_tws[sset].append([])
                            featured_last_tws[sset].append({})
                            this_tws[sset].append(this_tw[:(self.all_length-self.delimiter_num-source_len)])
                            featured_this_tws[sset].append(featured_this_tw)
                            labels[sset].append(tweet['A_label'])
                            branch_other_tws[sset].append([])
                        elif source_len + last_tw_len + this_tw_len >= (self.all_length - self.delimiter_num):
                            id_set[sset].append(id_str)
                            sources[sset].append(source)
                            featured_sources[sset].append(featured_source)
                            last_tws[sset].append(last_tw[:(self.all_length-self.delimiter_num-source_len-this_tw_len)])
                            featured_last_tws[sset].append(featured_last_tw)
                            this_tws[sset].append(this_tw)
                            featured_this_tws[sset].append(featured_this_tw)
                            labels[sset].append(tweet['A_label'])
                            branch_other_tws[sset].append([])
                        elif sset =="train" and other_tw_len!=0:  # 这里才开始有other tweet的位置
                            free_len = self.all_length - self.delimiter_num - source_len - last_tw_len - this_tw_len
                            num_split = int(other_tw_len/free_len)  # 这里可能是0
                            if other_tw_len%free_len > 0:  #有other_tweet的长度才新拆分一条数据
                                num_split+=1
                            for i in range(num_split):
                                id_set[sset].append(id_str)
                                sources[sset].append(source)
                                featured_sources[sset].append(featured_source)
                                last_tws[sset].append(last_tw)
                                featured_last_tws[sset].append(featured_last_tw)
                                this_tws[sset].append(this_tw)
                                featured_this_tws[sset].append(featured_this_tw)
                                labels[sset].append(tweet['A_label'])
                                branch_other_tws[sset].append(other_tw[i*free_len:min((i+1)*free_len,other_tw_len)])
                        else:  # dev+test，不切分，other_tw本来就没有的，在这里加入数据
                            id_set[sset].append(id_str)
                            sources[sset].append(source)
                            featured_sources[sset].append(featured_source)
                            last_tws[sset].append(last_tw)
                            featured_last_tws[sset].append(featured_last_tw)
                            this_tws[sset].append(this_tw)
                            featured_this_tws[sset].append(featured_this_tw)
                            labels[sset].append(tweet['A_label'])
                            branch_other_tws[sset].append(other_tw[:(self.all_length-self.delimiter_num-source_len-this_tw_len-last_tw_len)])
            for i in range(len(id_set[sset])):
                all_len = len(sources[sset][i])+len(branch_other_tws[sset][i])+len(last_tws[sset][i])+len(this_tws[sset][i])
                if all_len>self.all_length - self.delimiter_num:  # 感觉用字符串拼接以后的分词算法不太一样？词数会变多...所以不能到512
                    print(all_len)
                    print(len(sources[sset][i]),len(branch_other_tws[sset][i]),len(last_tws[sset][i]),len(this_tws[sset][i]))
            print(sset,len(id_set[sset]))  # train:5217->5278, dev:1485, test:1827
            '''
            np.save(os.path.join('data/taskA_newformdata/', sset, 'A_id_set'), id_set[sset])
            np.save(os.path.join('data/taskA_newformdata/', sset, 'A_source'),sources[sset])
            np.save(os.path.join('data/taskA_newformdata/', sset, 'A_other_tw'), branch_other_tws[sset])
            np.save(os.path.join('data/taskA_newformdata/', sset, 'A_last_tw'), last_tws[sset])
            np.save(os.path.join('data/taskA_newformdata/', sset, 'A_this_tw'), this_tws[sset])
            np.save(os.path.join('data/taskA_newformdata/', sset, 'A_label'), labels[sset])
            '''
            # 取得所有来自elena的句子级别特征
            np.save(os.path.join('data/taskA_newformdata/', sset, 'A_featured_source'), featured_sources[sset])
            np.save(os.path.join('data/taskA_newformdata/', sset, 'A_featured_last_tw'), featured_last_tws[sset])
            np.save(os.path.join('data/taskA_newformdata/', sset, 'A_featured_this_tw'), featured_this_tws[sset])




    def combine_new_tweets(self):
        # labels: 0:support,1:comment,2:deny,3:query
        # filter new data
        all_paths = ['data/taskA_examine_support_3/','data/taskA_examine_deny_2/','data/taskA_examine_query_2/']
        all_pred_filters = ['documents/support_examine_3.tsv','documents/deny_examine_2.tsv','documents/query_examine_2.tsv']
        i2labels = [0,2,3]
        filtered_new_tweets = []  # 三类扩充数据没必要分开了，直接叠加到一起即可
        for k in range(len(all_paths)):
            all_new_premise = np.load(os.path.join(all_paths[k],'dev/A_premise.npy'))
            all_new_hypothesis = np.load(os.path.join(all_paths[k],'dev/A_hypothesis.npy'))
            all_new_labels = np.load(os.path.join(all_paths[k],'dev/A_labels.npy'))
            all_new_ids = np.load(os.path.join(all_paths[k], 'dev/A_id_set.npy'))
            f = open(all_pred_filters[k], 'r')
            lines = f.readlines()[1:]
            f.close()
            for i in range(len(lines)):
                line = lines[i].strip()
                result = re.match(r'([a-z\d]+)\s(\d)\s\[(.*?)\]', line)
                if int(result.group(2)) == i2labels[k]:  # 替换成各类label对应的数字
                    tw = {}
                    id_str = int(result.group(1))  # 这个在新数据里其实就是下标
                    tw['source'] = all_new_premise[id_str]
                    tw['other_tw'] = []
                    tw['last_tw'] = []
                    tw['this_tw'] = all_new_hypothesis[id_str]
                    tw['label'] = all_new_labels[id_str]
                    tw['id'] = all_new_ids[id_str]
                    filtered_new_tweets.append(tw)

        # merge new & old train data  在新形式数据上加入扩充数据
        sources = np.load('data/taskA_newformdata/train/A_source.npy').tolist()
        other_tws = np.load('data/taskA_newformdata/train/A_other_tw.npy').tolist()
        last_tws = np.load('data/taskA_newformdata/train/A_last_tw.npy').tolist()
        this_tws = np.load('data/taskA_newformdata/train/A_this_tw.npy').tolist()
        labels = np.load('data/taskA_newformdata/train/A_label.npy').tolist()
        ids = np.load('data/taskA_newformdata/train/A_id_set.npy').tolist()
        for tw in filtered_new_tweets:
            sources.append(tw['source'])
            other_tws.append(tw['other_tw'])
            last_tws.append(tw['last_tw'])
            this_tws.append(tw['this_tw'])
            labels.append(tw['label'])
            ids.append(tw['id'])

        # re-organize merged data
        reorganized_tweets = []
        for i in range(len(sources)):
            tweet = {}
            tweet['source'] = sources[i]
            tweet['other_tw'] = other_tws[i]
            tweet['last_tw'] = last_tws[i]
            tweet['this_tw'] = this_tws[i]
            tweet['label'] = labels[i]
            tweet['id'] = ids[i]
            reorganized_tweets.append(tweet)
        random.shuffle(reorganized_tweets)

        # save reorganized merged data
        all_sources = []
        all_other_tws = []
        all_last_tws = []
        all_this_tws = []
        all_labels = []
        all_ids = []
        for i in range(len(reorganized_tweets)):
            all_sources.append(reorganized_tweets[i]['source'])
            all_other_tws.append(reorganized_tweets[i]['other_tw'])
            all_last_tws.append(reorganized_tweets[i]['last_tw'])
            all_this_tws.append(reorganized_tweets[i]['this_tw'])
            all_labels.append(reorganized_tweets[i]['label'])
            all_ids.append(reorganized_tweets[i]['id'])
        np.save('data/taskA_enlarge_newform_data/train/A_source', all_sources)
        np.save('data/taskA_enlarge_newform_data/train/A_other_tw', all_other_tws)
        np.save('data/taskA_enlarge_newform_data/train/A_last_tw', all_last_tws)
        np.save('data/taskA_enlarge_newform_data/train/A_this_tw', all_this_tws)
        np.save('data/taskA_enlarge_newform_data/train/A_label', all_labels)
        np.save('data/taskA_enlarge_newform_data/train/A_id_set', all_ids)


    def get_taskA_word_features(self,my_embedding):
        whichset = ['train','dev','test']
        for sset in whichset:
            sources = np.load(os.path.join('data/taskA_enlarge_newform_data',sset,'A_source.npy'))
            last_tws = np.load(os.path.join('data/taskA_enlarge_newform_data',sset,'A_last_tw.npy'))
            this_tws = np.load(os.path.join('data/taskA_enlarge_newform_data',sset,'A_this_tw.npy'))
            source_word_features = []
            last_tw_word_features = []
            this_tw_word_features = []
            for i in range(len(sources)):
                source_word_features.append(my_embedding.create_taskA_word_features(sources[i]))
                last_tw_word_features.append(my_embedding.create_taskA_word_features(last_tws[i]))
                this_tw_word_features.append(my_embedding.create_taskA_word_features(this_tws[i]))
            np.save(os.path.join('data/taskA_enlarge_newform_data',sset,'A_source_word_feature'), source_word_features)
            np.save(os.path.join('data/taskA_enlarge_newform_data',sset,'A_last_tw_word_feature'), last_tw_word_features)
            np.save(os.path.join('data/taskA_enlarge_newform_data',sset,'A_this_tw_word_feature'), this_tw_word_features)


    def get_taskA_sentence_features(self):
        whichset = ['train','dev','test']
        all_thistw_int_sent_features = {}
        for feature in self.int_feature_dict:
            all_thistw_int_sent_features[feature] = []
        for sset in whichset:  # 仅为原数据的特征，还要和新数据进行合并
            sources = np.load(os.path.join('data/taskA_newformdata',sset,'A_featured_source.npy'))
            last_tws = np.load(os.path.join('data/taskA_newformdata',sset,'A_featured_last_tw.npy'))
            this_tws = np.load(os.path.join('data/taskA_newformdata',sset,'A_featured_this_tw.npy'))
            source_sent_features = []
            last_tw_sent_features = []
            this_tw_sent_features = []
            for i in range(len(sources)):
                temp_source_sent_features = []
                temp_last_tw_sent_features = []
                temp_this_tw_sent_features = []
                temp_source_sent_features.extend(self.get_int_sent_features(sources[i]))
                temp_this_tw_sent_features.extend(self.get_int_sent_features(this_tws[i]))
                for feature in self.str_feature_dict:
                    temp_source_sent_features.append(sources[i][feature])
                    temp_this_tw_sent_features.append(this_tws[i][feature])
                if last_tws[i] != {}:
                    temp_last_tw_sent_features.extend(self.get_int_sent_features(last_tws[i]))
                    for feature in self.str_feature_dict:
                        temp_last_tw_sent_features.append(last_tws[i][feature])
                source_sent_features.append(temp_source_sent_features)
                last_tw_sent_features.append(temp_last_tw_sent_features)
                this_tw_sent_features.append(temp_this_tw_sent_features)

            np.save(os.path.join('data/taskA_newformdata',sset,'A_source_sent_feature'), source_sent_features)
            np.save(os.path.join('data/taskA_newformdata',sset,'A_last_tw_sent_feature'), last_tw_sent_features)
            np.save(os.path.join('data/taskA_newformdata',sset,'A_this_tw_sent_feature'), this_tw_sent_features)
        #np.save('all_int_sent_features',all_thistw_int_sent_features)


    def get_int_sent_features(self,tw):
        int_sent_features = []
        # favorite
        if tw['favorite'] < 0:
            int_sent_features.append(-1)
        elif tw['favorite'] == 0:
            int_sent_features.append(0)
        elif tw['favorite'] >0 and tw['favorite']<=1:
            int_sent_features.append(1)
        elif tw['favorite'] >1 and tw['favorite'] <=2:
            int_sent_features.append(2)
        elif tw['favorite'] >2 and tw['favorite'] <=5:
            int_sent_features.append(5)
        elif tw['favorite'] >5 and tw['favorite'] <=22:
            int_sent_features.append(22)
        else:
            int_sent_features.append(100)
        # retweet
        if tw['retweet'] < 0:
            int_sent_features.append(-1)
        elif tw['retweet'] == 0:
            int_sent_features.append(0)
        elif tw['retweet'] >0 and tw['retweet'] <=1:
            int_sent_features.append(1)
        elif tw['retweet'] >1 and tw['retweet'] <=100:
            int_sent_features.append(100)
        else:
            int_sent_features.append(200)
        # depth
        int_sent_features.append(tw['depth'])
        # user_followers
        if tw['user_followers'] < 0:
            int_sent_features.append(-1)
        elif tw['user_followers'] == 0:
            int_sent_features.append(0)
        elif tw['user_followers'] >0 and tw['user_followers']<=500:
            int_sent_features.append(500)
        elif tw['user_followers'] >500 and tw['user_followers'] <=1000:
            int_sent_features.append(1000)
        elif tw['user_followers'] >1000 and tw['user_followers'] <=2000:
            int_sent_features.append(2000)
        elif tw['user_followers'] >2000 and tw['user_followers'] <=5000:
            int_sent_features.append(5000)
        elif tw['user_followers'] >5000 and tw['user_followers'] <=10000:
            int_sent_features.append(10000)
        elif tw['user_followers'] >10000 and tw['user_followers'] <=100000:
            int_sent_features.append(100000)
        else:
            int_sent_features.append(1000000)
        # user_listed
        if tw['user_listed'] < 0:
            int_sent_features.append(-1)
        elif tw['user_listed'] == 0:
            int_sent_features.append(0)
        elif tw['user_listed'] >0 and tw['user_listed']<=1:
            int_sent_features.append(1)
        elif tw['user_listed'] >1 and tw['user_listed'] <=6:
            int_sent_features.append(6)
        elif tw['user_listed'] >6 and tw['user_listed'] <=20:
            int_sent_features.append(20)
        elif tw['user_listed'] >20 and tw['user_listed'] <=100:
            int_sent_features.append(100)
        elif tw['user_listed'] >100 and tw['user_listed'] <=1000:
            int_sent_features.append(1000)
        else:
            int_sent_features.append(10000)
        # user_statuses
        if tw['user_statuses'] < 0:
            int_sent_features.append(-1)
        elif tw['user_statuses'] >=0 and tw['user_statuses']<=1000:
            int_sent_features.append(1000)
        elif tw['user_statuses'] >1000 and tw['user_statuses']<=2500:
            int_sent_features.append(2500)
        elif tw['user_statuses'] >2500 and tw['user_statuses'] <=5000:
            int_sent_features.append(5000)
        elif tw['user_statuses'] >5000 and tw['user_statuses'] <=10000:
            int_sent_features.append(10000)
        elif tw['user_statuses'] >10000 and tw['user_statuses'] <=20000:
            int_sent_features.append(20000)
        elif tw['user_statuses'] >20000 and tw['user_statuses'] <=35000:
            int_sent_features.append(35000)
        elif tw['user_statuses'] >35000 and tw['user_statuses'] <=75000:
            int_sent_features.append(75000)
        else:
            int_sent_features.append(150000)
        # user_friends
        if tw['user_friends'] < 0:
            int_sent_features.append(-1)
        elif tw['user_friends']>=0 and tw['user_friends']<=200:
            int_sent_features.append(200)
        elif tw['user_friends'] >200 and tw['user_friends']<=380:
            int_sent_features.append(380)
        elif tw['user_friends'] >380 and tw['user_friends'] <=650:
            int_sent_features.append(650)
        elif tw['user_friends'] >650 and tw['user_friends'] <=1100:
            int_sent_features.append(1100)
        elif tw['user_friends'] >1100 and tw['user_friends'] <=2000:
            int_sent_features.append(2000)
        else:
            int_sent_features.append(10000)
        return int_sent_features


    def merge_sent_with_word_features_in_newdata(self):
        whichset = ['train', 'dev', 'test']
        for sset in whichset:
            all_source_word_features = np.load(os.path.join('data/taskA_enlarge_newform_data', sset, 'A_source_word_feature.npy'))
            all_last_tw_word_features = np.load(os.path.join('data/taskA_enlarge_newform_data', sset, 'A_last_tw_word_feature.npy'))
            all_this_tw_word_features = np.load(os.path.join('data/taskA_enlarge_newform_data', sset, 'A_this_tw_word_feature.npy'))
            all_ids = np.load('data/taskA_enlarge_newform_data/' + sset + '/A_id_set.npy')
            source_sent_features = np.load(os.path.join('data/taskA_newformdata', sset, 'A_source_sent_feature.npy'))
            last_tw_sent_features = np.load(os.path.join('data/taskA_newformdata', sset, 'A_last_tw_sent_feature.npy'))
            this_tw_sent_features = np.load(os.path.join('data/taskA_newformdata', sset, 'A_this_tw_sent_feature.npy'))
            ori_ids = np.load(os.path.join('data/taskA_newformdata', sset, 'A_id_set.npy'))
            # 需要满足加上扩充数据后的打乱顺序
            for i in range(len(all_ids)):
                flag = 0
                for j,ori_id in enumerate(ori_ids):
                    if all_ids[i]==ori_id:
                        flag = 1
                        all_source_word_features[i].extend(source_sent_features[j])
                        all_last_tw_word_features[i].extend(last_tw_sent_features[j])
                        all_this_tw_word_features[i].extend(this_tw_sent_features[j])
                        break
                if not flag:
                    all_source_word_features[i].extend([-1,-1,-1,-1,-1,-1,-1,'None','None','None','None','None','None','None','None','None','None'])
            np.save(os.path.join('data/taskA_enlarge_newform_data', sset, 'A_source_word_sent_feature'),all_source_word_features)
            np.save(os.path.join('data/taskA_enlarge_newform_data', sset, 'A_last_tw_word_sent_feature'),all_last_tw_word_features)
            np.save(os.path.join('data/taskA_enlarge_newform_data', sset, 'A_this_tw_word_sent_feature'),all_this_tw_word_features)






    #def get_taskB_word_features(self):


    def split_cleaned_text_A(self):
        print("\nstarting task A:")
        id_set = {}
        id_set['train']=[]
        id_set['dev']=[]
        id_set['test']=[]
        premise = {}
        premise['train']=[]
        premise['dev']=[]
        premise['test']=[]
        hypothesis = {}
        hypothesis['train']=[]
        hypothesis['dev']=[]
        hypothesis['test']=[]
        labels={}
        labels['train']=[]
        labels['dev']=[]
        labels['test']=[]

        #whichset = ['train', 'dev', 'test']
        whichset = ['train', 'dev']  # 19rumor

        for sset in whichset:
            branches_text_in_one_set = np.load('saved_dataRumEval2019/' + sset + '/cleaned_tokenized_branch_A_B_label_stanford.npy')
            for j, branch in enumerate(branches_text_in_one_set):
                for i, tweet in enumerate(branch):
                    p = []
                    branch_tweets = branch[:i + 1]
                    for branch_tweet in branch_tweets:
                        for word in branch_tweet['text'][:self.A_stop_h_length]:
                            if len(p) < self.A_stop_p_length:
                                p.append(word)
                    h = tweet['text']

                    id_str = tweet['id']
                    label = tweet['A_label']
                    if id_str not in id_set[sset]:
                        temp_i = 1  # 1条未重复的tweet变成了几条
                        p_len = len(p)
                        h_len = len(h)
                        #h_word_features = self.get_taskB_word_features(h, h_len)[h_len:]
                        #print('word features:',h_word_features)
                        # 句子级别特征有None和-1但是不会太多
                        #print('sent features:',[tweet[feature] for feature in self.feature_dict])
                        if p_len > self.A_stop_p_length:
                            self.overlen_p_num += 1
                        premise[sset].append(p)
                        id_set[sset].append(id_str)
                        labels[sset].append(label)
                        if h_len > self.A_stop_h_length and sset=='train':  # 只有训练集切分，验证集先不切
                            self.overlen_h_num += 1
                            h1 = h[:self.A_stop_h_length]
                            h1_len = len(h1)
                            self.all_h_length += h1_len
                            for feature in self.feature_dict:
                                h1.append(tweet[feature])
                            #print(len(h1)-h1_len)  # 句子级别特征有多少个
                            h1 = self.get_taskB_word_features(h1,h1_len)
                            #print(len(h1)-h1_len)  # 词+句子级别特征有多少个
                            #print(h1)
                            hypothesis[sset].append(h1)

                            if h_len <= self.A_stop_h_length * 2:
                                temp_i += 1
                                h2 = h[-self.A_stop_h_length:]
                                h2_len = len(h2)
                                self.all_h_length += h2_len
                                for feature in self.feature_dict:
                                    h2.append(tweet[feature])
                                h2 = self.get_taskB_word_features(h2,h2_len)
                                #print(h2)
                                premise[sset].append(p)
                                hypothesis[sset].append(h2)
                                id_set[sset].append(id_str)
                                labels[sset].append(label)

                            elif h_len <= self.A_stop_h_length * 3:
                                temp_i += 2
                                h2 = h[self.A_stop_h_length:self.A_stop_h_length * 2]
                                h2_len = len(h2)
                                self.all_h_length += h2_len
                                for feature in self.feature_dict:
                                    h2.append(tweet[feature])
                                h2 = self.get_taskB_word_features(h2,h2_len)
                                #print(h2)
                                premise[sset].append(p)
                                hypothesis[sset].append(h2)
                                id_set[sset].append(id_str)
                                labels[sset].append(label)

                                h3 = h[-self.A_stop_h_length:]
                                h3_len = len(h3)
                                self.all_h_length += h3_len
                                for feature in self.feature_dict:
                                    h3.append(tweet[feature])
                                h3 = self.get_taskB_word_features(h3,h3_len)
                                #print(h3)
                                premise[sset].append(p)
                                hypothesis[sset].append(h3)
                                id_set[sset].append(id_str)
                                labels[sset].append(label)
                            else:
                                temp_i += 3
                                h2 = h[self.A_stop_h_length:self.A_stop_h_length * 2]
                                h2_len = len(h2)
                                self.all_h_length += h2_len
                                for feature in self.feature_dict:
                                    h2.append(tweet[feature])
                                h2 = self.get_taskB_word_features(h2,h2_len)
                                #print(h2)
                                premise[sset].append(p)
                                hypothesis[sset].append(h2)
                                id_set[sset].append(id_str)
                                labels[sset].append(label)

                                h3 = h[self.A_stop_h_length * 2:self.A_stop_h_length * 3]
                                h3_len = len(h3)
                                self.all_h_length += h3_len
                                for feature in self.feature_dict:
                                    h3.append(tweet[feature])
                                h3 = self.get_taskB_word_features(h3,h3_len)
                                #print(h3)
                                premise[sset].append(p)
                                hypothesis[sset].append(h3)
                                id_set[sset].append(id_str)
                                labels[sset].append(label)

                                h4 = h[-self.A_stop_h_length:]
                                h4_len = len(h4)
                                self.all_h_length += h4_len
                                for feature in self.feature_dict:
                                    h4.append(tweet[feature])
                                h4 = self.get_taskB_word_features(h4,h4_len)
                                #print(h4)
                                premise[sset].append(p)
                                hypothesis[sset].append(h4)
                                id_set[sset].append(id_str)
                                labels[sset].append(label)
                        else:
                            for feature in self.feature_dict:
                                h.append(tweet[feature])
                            h = self.get_taskB_word_features(h,h_len)
                            #print(h)
                            hypothesis[sset].append(h)
                            self.all_h_length += h_len

                        self.p_num += temp_i
                        self.overlen_h_new += temp_i-1
                        if self.max_p_length < p_len:
                            self.max_p_length = p_len
                        if self.max_h_length < h_len:
                            self.max_h_length = h_len
                        self.all_p_length += temp_i * p_len

            # save to files

            np.save(os.path.join('saved_dataRumEval2019', sset, 'A_featured_premise'),premise[sset])
            np.save(os.path.join('saved_dataRumEval2019', sset, 'A_featured_hypothesis'), hypothesis[sset])
            np.save(os.path.join('saved_dataRumEval2019', sset, 'A_featured_labels'), labels[sset])
            np.save(os.path.join('saved_dataRumEval2019', sset, 'A_featured_id_set'), id_set[sset])


            print(sset)
            print('length of premise:',len(premise[sset]))
            print('length of hypothesis:', len(hypothesis[sset]))
            print('length of labels:', len(labels[sset]))


    def check_A(self):
        print('p_num:', self.p_num)  # 切开之前：6700 -> 少两条？
        print('overlen num',self.overlen_h_num)
        print('overlen tweets increases',self.overlen_h_new,'tweets.')

        print('max p length:', self.max_p_length)  # 621  19:1582->537(max_h=32)（人为设定最大300，只取前300）
        print('max h length:', self.max_h_length)  # 39  19:1163->切分前1107（人为设定最大50）
        aver_p_length = self.all_p_length/self.p_num
        aver_h_length = self.all_h_length/self.p_num
        print('aver p length:', aver_p_length)  # 56.4  19:86.7->61.6（max_p=180,max_h=32）
        print('aver h length:', aver_h_length)  # 19.6  19:24.3->20.6（最大32）
        print('stop p length:', self.A_stop_p_length, 'percentage:', (6700 - self.overlen_p_num)/6700)  # 人为设定后，没有超过长度的
        print('stop h length:', self.A_stop_h_length, 'percentage:', (6700 - self.overlen_h_num)/6700)


    def split_cleaned_text_B(self):
        print("\nstarting task B:")
        id_set = {}
        id_set['train'] = []
        id_set['dev'] = []
        id_set['test'] = []
        premise = {}
        premise['train'] = []
        premise['dev'] = []
        premise['test'] = []
        hypothesis = {}
        hypothesis['train'] = []
        hypothesis['dev'] = []
        hypothesis['test'] = []
        B_labels = {}
        B_labels['train'] = []
        B_labels['dev'] = []
        B_labels['test'] = []
        # train: 3560个branch, premise=hypothsis=labels=id_set=
        # (1) train扩展，stop_p_len=120：(4105,)
        # (2) train不用多个branch，只过长切分，stop_p_len=70：(1595,)
        # (3) train只多个branch，不过长切分：
        # (4) train每个source只一次，stop_p_len=70：(327,)
        # dev: 966个branch, premise=hypothsis=labels=id_set = (38,) --dev每个source只一次
        # test:

        whichset = ['train', 'dev', 'test']

        for sset in whichset:
            unverified_num = 0
            unverified_ids = []
            branches_text_in_one_set = np.load('data/taskA_newformdata/' + sset + '/cleaned_tokenized_branch_A_B_label_stanford.npy')
            for j, branch in enumerate(branches_text_in_one_set):
                source = branch[0]
                id_str = source['id']
                B_label = source['B_label']
                # 如果按主办方说的要做二分类，就需要把unverified类的训练数据丢弃。
                if sset=="train" and B_label == 2:
                    if id_str not in unverified_ids:
                        unverified_ids.append(id_str)
                        unverified_num += 1
                    continue
                h = source['text'][:self.B_stop_h_length]
                h_len = len(h)
                # 以下2行是加特征
                #h = self.get_taskB_word_features(h, h_len)
                #h.extend(self.get_int_sent_features(source))
                if self.max_h_length < h_len:
                    self.max_h_length = h_len
                branch_words = []
                for branch_tweet in branch:
                    branch_words.extend(branch_tweet['text'])
                if id_str not in id_set[sset]:
                    if len(source['text']) > self.B_stop_h_length:
                        self.overlen_h_num += 1
                    if len(branch_words) > self.B_stop_p_length:  # 一条source只算一次
                        self.overlen_p_num += 1
                p = []
                # if len(branch_words) > stop_p_length:  # 按branch计算
                #    overlen_p_num += 1
                for word in branch_words:
                    if len(p) < self.B_stop_p_length:
                        p.append(word)
                    # 把下面都注释掉是single

                    elif sset == "dev" or sset=="test":
                        break
                    # 只有train才把长的数据分好几条
                    else:
                        premise[sset].append(p)
                        hypothesis[sset].append(h)
                        id_set[sset].append(id_str)
                        B_labels[sset].append(B_label)
                        self.p_num += 1
                        p_len = len(p)
                        self.all_p_length += p_len
                        self.all_h_length += h_len
                        if self.max_p_length < p_len:
                            self.max_p_length = p_len
                        p = []


                # 加上该branch的最后一个词没到stop_p_length时，把剩余的写入

                # single: train和dev每条都只写入一次（train过长切开，dev不切，只截断） 372条
                #if p != [] and id_str not in id_set[sset]:
                # full: dev和test集每条source只写入一次 4838条
                if p != [] and (sset == "train" or ((sset == "dev" or sset=="test") and id_str not in id_set[sset])):
                    premise[sset].append(p)
                    hypothesis[sset].append(h)
                    id_set[sset].append(id_str)
                    B_labels[sset].append(B_label)
                    self.p_num += 1
                    p_len = len(p)
                    self.all_p_length += p_len
                    self.all_h_length += h_len

            # save to files
            np.save(os.path.join('data/taskB_bi_classify_full', sset, 'B_featured_premise'), premise[sset])
            np.save(os.path.join('data/taskB_bi_classify_full', sset, 'B_featured_hypothesis'), hypothesis[sset])
            np.save(os.path.join('data/taskB_bi_classify_full', sset, 'B_featured_labels'), B_labels[sset])
            np.save(os.path.join('data/taskB_bi_classify_full', sset, 'B_featured_id_set'), id_set[sset])

            print(sset)
            print('length of premise:', len(premise[sset]))
            print('length of hypothesis:', len(hypothesis[sset]))
            print('length of labels:', len(B_labels[sset]))
            print('unverified num:',unverified_num)


    def check_B(self):
        print('p_num:', self.p_num)
        print('max p length:', self.max_p_length)  # 70
        print('max h length:', self.max_h_length)  # 28
        aver_p_length = self.all_p_length / self.p_num
        aver_h_length = self.all_h_length / self.p_num
        print('aver p length:', aver_p_length)  # train=861->88.5,train=327->42.48
        print('aver h length:', aver_h_length)  # 19.7
        # print('stop p length:',stop_p_length,'overlen p num:',overlen_p_num,'stop length include percentage:',1-overlen_p_num/(3560+966))  # 算的是各个branch的context
        print('stop p length:', self.B_stop_p_length, 'overlen p num:', self.overlen_p_num, 'stop length include percentage:',
              1 - self.overlen_p_num / (327 + 38))  # 算的是source的
        print('stop h length:', self.B_stop_h_length, 'overlen h num:', self.overlen_h_num, 'stop length include percentage:',
              1 - self.overlen_h_num / (327 + 38))  # 算的是source的


    def get_taskB_word_features(self,h,h_len):
        if h_len:  # h有空的
            my_embedding = GetEmbedding()
            word_features = my_embedding.create_taskB_word_features(h[:h_len])
            if word_features:  # 有的为空
                h.extend(word_features)
        return h


if __name__ == "__main__":
    my_split = SplitText()  # 3.py
    my_embedding = GetEmbedding()  # 4.py
    # 1月30-31
    # task A
    # 把数据搞成讨论出的新格式
    #my_split.taskA_to_GPT()
    # 加上扩充数据
    #my_split.combine_new_tweets()
    # 加上taskA词级别特征
    #my_split.get_taskA_word_features(my_embedding)
    # 加上taskA句子级别特征
    #my_split.get_taskA_sentence_features()
    # 在加上词级别特征的原+扩充数据集上再添加句子特征
    #my_split.merge_sent_with_word_features_in_newdata()

    # task B
    # 加上taskB词级别特征
    my_split.split_cleaned_text_B()
    #my_split.get_taskB_word_features(my_embedding)


    # 原
    #my_split.split_cleaned_text_A()
    #my_split.check_A()
    #my_split.split_cleaned_text_B()
    #my_split.check_B()