# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 18:05:51 2019

@author: dell
"""
from Levenshtein import *
import numpy as np
import json
import sys



'''
1. load original data
'''
ori_support_tweets = np.load("data/ori_data/support_tweets.npy")
ori_deny_tweets = np.load("data/ori_data/deny_tweets.npy")
ori_query_tweets = np.load("data/ori_data/query_tweets.npy")
ori_comment_tweets = np.load("data/ori_data/comment_tweets.npy")



'''
2.new dataset: support, deny
'''

# (1) semEval 2016 task 6
semeval16_train_file="data/stancedataset/train.csv"
semeval16_test_file="data/stancedataset/test.csv"

f=open(semeval16_train_file,'r')
lines=f.readlines()[1:]
f.close()

f=open(semeval16_test_file,'r')
lines.extend(f.readlines()[1:])
f.close()

tweets = []
stances = []
for line in lines:
    line = line.strip()
    items = line.split(",")
    tweets.append(items[0])
    stances.append(items[2])

new_support_tweets=[]
new_deny_tweets=[]
for i,tw in enumerate(tweets):
    if stances[i]=="FAVOR":
        new_support_tweets.append(tw)
    elif stances[i]=="AGAINST":
        new_deny_tweets.append(tw)

#print(len(new_support_tweets))  # 889
#print(len(new_deny_tweets))  # 1646


# (2) Emergent
emergent_train_file="data/emergent_data/url-versions-2015-06-14-clean-train.csv"
emergent_test_file="data/emergent_data/url-versions-2015-06-14-clean-test.csv"

f=open(emergent_train_file,'r')
lines=f.readlines()[1:]
f.close()

f=open(emergent_test_file,'r')
lines.extend(f.readlines()[1:])
f.close()

support_emergent_ps = []
support_emergent_hs = []
deny_emergent_ps = []
deny_emergent_hs = []
for line in lines:
    line = line.strip()
    items = line.split(",")
    if items[3]=="for":
        support_emergent_ps.append(items[1])
        support_emergent_hs.append(items[2])
    elif items[3]=="against":
        deny_emergent_ps.append(items[1])
        deny_emergent_hs.append(items[2])

#print(len(support_emergent_hs))  # 963
#print(len(deny_emergent_hs))  # 273



# (3) Twitter sentiment analysis --sentiment140
sentiment140_train_file="data/sentiment140/training.1600000.processed.noemoticon.csv"

f=open(sentiment140_train_file,'r',encoding = "ISO-8859-1")
lines=f.readlines()
f.close()

support_sentiment140_hs = []
deny_sentiment140_hs = []
for line in lines:
    line = line.strip()
    items = line.split(",")
    if items[0]=='"4"':  # 双引号也是这个字符串的一部分
        support_sentiment140_hs.append(items[5])
    elif items[0]=='"0"':
        deny_sentiment140_hs.append(items[5])


#print(len(support_sentiment140_hs))  # 800000
#print(len(deny_sentiment140_hs))  # 800000






'''
3.new dataset: query
'''
# (1) NewsQA -- domain:news
# (2) MS Marco -- type:unanswerable                

# (3) SQuAD 2.0 -- type:unanswerable 49443 (relevant to topic, existence of plausible answers)
# unanswerable: "answers": [], "is_impossible": true
'''
文件结构：
version,data
data[0]
paragraphs,title
paragraphs[0]
context-P(一段),qas
qas[0]
id,answers=[],question,is_impossible=True (unanswerable)
'''

squad_train_file="data/SQuAD_2.0/train-v2.0.json"
squad_dev_file="data/SQuAD_2.0/dev-v2.0.json"

squad_train = json.load(open(squad_train_file,'r'))
squad_dev = json.load(open(squad_dev_file,'r'))
squad_files = [squad_train,squad_dev]

squad_ps = []
squad_hs = []
for t in range(2):
    for i in range(len(squad_files[t]['data'])):
        for j in range(len(squad_files[t]['data'][i]['paragraphs'])):
            p = squad_files[t]['data'][i]['paragraphs'][j]['context']
            for k in range(len(squad_files[t]['data'][i]['paragraphs'][j]['qas'])):
                if squad_files[t]['data'][i]['paragraphs'][j]['qas'][k]['is_impossible']:
                    h = squad_files[t]['data'][i]['paragraphs'][j]['qas'][k]['question']
                    squad_ps.append(p)
                    squad_hs.append(h)
#print(len(squad_ps))  # 49443
#print(len(squad_hs))                



# (4) CoQA -- type:unanswerable, domain:news,reddit
'''
文件结构：
version,data
data[0]
filename,story-P(一段),name,id,source,questions-Q,answers-A,additional_answers-给每个Q额外收集了3个可能的A
questions[0]:turn_id,input_text (与answers中unknown相对应的turn_id,一对Q-A的turn_id相同)
answers[0]:turn_id,span_end=-1,span_start=-1,span_text=unknown,input_text=unknown (unanswerable)
additional_answers: '2','1','0'
additional_answers['2'][0]:turn_id,span_end,span_start,span_text,input_text (answers是unknown的，additional answers里给出了别的可能答案，但不是最合适)
'''

coqa_train_file="data/CoQA/coqa-train-v1.0.json"
coqa_dev_file="data/CoQA/coqa-dev-v1.0.json"

coqa_train = json.load(open(coqa_train_file,'r'))
coqa_dev = json.load(open(coqa_dev_file,'r'))
coqa_files = [coqa_train,coqa_dev]

coqa_ps = []
coqa_hs = []
for t in range(2):
    for i in range(len(coqa_files[t]['data'])):
        p = coqa_files[t]['data'][i]['story']
        for j in range(len(coqa_files[t]['data'][i]['answers'])):
            if coqa_files[t]['data'][i]['answers'][j]['input_text']=='unknown':
                turn_id = coqa_files[t]['data'][i]['answers'][j]['turn_id']
                for k in range(len(coqa_files[t]['data'][i]['questions'])):
                    if coqa_files[t]['data'][i]['questions'][k]['turn_id']==turn_id:
                        h = coqa_files[t]['data'][i]['questions'][k]['input_text']
                        coqa_ps.append(p)
                        coqa_hs.append(h)
#print(len(coqa_ps))  # 1437
#print(len(coqa_hs))









'''
3. calculate the distance and filter
'''

# a. semEval 2016 task 6
# 1. support
support_tw_dis = []
add_support_tweets = []
stop_dis = 0.7
for new_tw in new_support_tweets:
    tw={}
    min_dis = 1
    for old_tw in ori_support_tweets:
        dis = distance(new_tw, old_tw['text'])
        dis = dis/max(len(new_tw), len(old_tw['text']))
        if dis < min_dis:
            min_dis = dis
    tw['dis']=min_dis
    tw['text']=new_tw
    support_tw_dis.append(tw)
    if min_dis <= stop_dis:
        add_support_tweets.append(tw)

print(len(add_support_tweets))
#print(support_tw_dis)

'''
# 2. deny  dis<=0.7:354
deny_tw_dis = []
add_deny_tweets = []
stop_dis = 0.7
for new_tw in new_deny_tweets:
    tw={}
    min_dis = 1
    for old_tw in ori_deny_tweets:
        dis = distance(new_tw, old_tw['text'])
        dis = dis/max(len(new_tw), len(old_tw['text']))
        if dis < min_dis:
            min_dis = dis
    tw['dis']=min_dis
    tw['text']=new_tw
    deny_tw_dis.append(tw)
    if min_dis <= stop_dis:
        add_deny_tweets.append(tw)

print(len(add_deny_tweets))
#print(deny_tw_dis)
'''



# b. Emergent
# 1. support  dis<0.7:275
support_tw_dis = []
stop_dis = 0.7
for new_tw in support_emergent_hs:
    tw={}
    min_dis = 1
    for old_tw in ori_support_tweets:
        dis = distance(new_tw, old_tw['text'])
        dis = dis/max(len(new_tw), len(old_tw['text']))
        if dis < min_dis:
            min_dis = dis
    tw['dis']=min_dis
    tw['text']=new_tw
    support_tw_dis.append(tw)
    if min_dis <= stop_dis:
        add_support_tweets.append(tw)

print(len(add_support_tweets))
#print(support_tw_dis)

'''
# 2. deny  dis<=0.7:72
deny_tw_dis = []
stop_dis = 0.7
for new_tw in deny_emergent_hs:
    tw={}
    min_dis = 1
    for old_tw in ori_deny_tweets:
        dis = distance(new_tw, old_tw['text'])
        dis = dis/max(len(new_tw), len(old_tw['text']))
        if dis < min_dis:
            min_dis = dis
    tw['dis']=min_dis
    tw['text']=new_tw
    deny_tw_dis.append(tw)
    if min_dis <= stop_dis:
        add_deny_tweets.append(tw)

print(len(add_deny_tweets))
#print(deny_tw_dis)
'''


# c. Twitter sentiment analysis --sentiment140
# 1. support  dis<0.6:1914
'''
support_tw_dis = []
for i,new_tw in enumerate(support_sentiment140_hs):
    tw={}
    min_dis = 1
    if i%10000==0:
        print(i)  # 每一万输出打印一下，知道进度
        sys.stdout.flush()
    for old_tw in ori_support_tweets:
        dis = distance(new_tw, old_tw['text'])
        dis = dis/max(len(new_tw), len(old_tw['text']))
        if dis < min_dis:
            min_dis = dis
    tw['dis']=min_dis
    tw['text']=new_tw
    support_tw_dis.append(tw)
np.save('documents/sentiment140_support_dis', support_tw_dis)
'''
stop_dis = 0.58
support_tweets = np.load('documents/sentiment140_support_dis.npy')
for tw in support_tweets:
    if tw['dis'] <= stop_dis:
        add_support_tweets.append(tw)
print(len(add_support_tweets))
np.save('documents/new_support_data', add_support_tweets)



'''
# 2. deny  dis<=0.6:1202,dis<=0.61:1710,dis<=0.62:3006
deny_tw_dis = []
for i,new_tw in enumerate(deny_sentiment140_hs):
    tw={}
    min_dis = 1
    if i%10000==0:
        print(i)  # 每一万输出打印一下，知道进度
    for old_tw in ori_deny_tweets:
        dis = distance(new_tw, old_tw['text'])
        dis = dis/max(len(new_tw), len(old_tw['text']))
        if dis < min_dis:
            min_dis = dis
    tw['dis']=min_dis
    tw['text']=new_tw
    deny_tw_dis.append(tw)
np.save('documents/sentiment140_deny_dis',deny_tw_dis)

stop_dis = 0.65
deny_tweets = np.load('documents/sentiment140_deny_dis.npy')
for tw in deny_tweets:
    if tw['dis'] <= stop_dis:
        add_deny_tweets.append(tw)
print(len(add_deny_tweets))
np.save('documents/new_deny_data', add_deny_tweets)
'''






'''
# 3. query
add_query_tweets = []

# c. SQuAD 2.0
query_tw_dis = []
stop_dis = 0.6  # dis<0.6:1121
for i,new_tw in enumerate(squad_hs):
    #print(i)
    tw={}
    min_dis = 1
    for old_tw in ori_query_tweets:
        dis = distance(new_tw, old_tw['text'])
        dis = dis/max(len(new_tw), len(old_tw['text']))
        if dis < min_dis:
            min_dis = dis
    tw['dis']=min_dis
    tw['p']=squad_ps[i]
    tw['h']=new_tw
    query_tw_dis.append(tw)
    if min_dis <= stop_dis:
        add_query_tweets.append(tw)

#print(len(add_query_tweets))
#print(query_tw_dis)

# d. CoQA
query_tw_dis = []
stop_dis = 0.65  # dis<0.6:98, dis<0.65:574
for i,new_tw in enumerate(coqa_hs):
    #print(i)
    tw={}
    min_dis = 1
    for old_tw in ori_query_tweets:
        dis = distance(new_tw, old_tw['text'])
        dis = dis/max(len(new_tw), len(old_tw['text']))
        if dis < min_dis:
            min_dis = dis
    tw['dis']=min_dis
    tw['p'] = coqa_ps[i]
    tw['h'] = new_tw
    query_tw_dis.append(tw)
    if min_dis <= stop_dis:
        add_query_tweets.append(tw)

#print(len(add_query_tweets))
#print(query_tw_dis)
np.save('documents/new_query_data', add_query_tweets)
'''
