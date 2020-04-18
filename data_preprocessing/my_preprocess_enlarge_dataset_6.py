import numpy as np
import os
import re
import random
import pandas as pd
#from stanfordcorenlp import StanfordCoreNLP

# 正则表达式
p1 = r'^[\.!@#\$%\\\^&\*\)\(\+=\{\}\[\]\/\",\'\“\”\’<>~\·`\?:;|\-\s]{2,}$'
p2 = r'[^a-zA-Z0-9\.!@#\$%\\\^&\*\)\(\+=\{\}\[\]\/\",\'\“\”\’<>~\·`\?:;|\-\s]+'
p3 = r'(\w+?)/(\w+)'  # \w=[A-Za-z0-9_]
p4 = r'(.+)@(.+)'
p5 = r'([A-Z]*[a-z]+)\.([A-Z]*[a-z]+)'  # 非大写字母
p6 = r'(\w+)\?(\w+)'
p7 = r'[A-Z]+'
p8 = r'[😄🚫🚨🙋🙄🙃😭😩😨😦😥😤😠😕😔😄👺👌🐸🍸🍁❌♡☕️☕►…не😱😪👏👍👇❤️😴😂🙏💔💜😒]+'
p9 = r'(.*)urlurlurl(.+)'
p10 = r'(.+)urlurlurl(.*)'
p11 = r'[\.!@#\$%\\\^&\*\)\(\+=\{\}\[\]\/\",\'\“\”\’<>~\·`\?:;|\-\s]+(\d+)'
p12 = r'[\s\n\r\t]+'
p13 = r'@(.+)'


def eng_word_tokenize(nlp):
    new_dataset = np.load('new_support_data.npy', encoding='bytes')
    new_tokenized_tweets = []
    for tw in new_dataset:
        #tw['p'] = nlp.word_tokenize(tw['p'])
        #tw['h'] = nlp.word_tokenize(tw['h'])
        tw['text'] = nlp.word_tokenize(tw['text'])
        new_tokenized_tweets.append(tw)
    np.save('new_support_tokenized_data', new_tokenized_tweets)


def clean_tokenized_text():
    new_tokenized_data = np.load('documents/new_support_tokenized_data.npy')
    cleaned_tweets = []
    for i, tw in enumerate(new_tokenized_data):
        #tw['p']=post_process(tw['p'])
        #tw['h']=post_process(tw['h'])
        tw['text'] = post_process(tw['text'])
        cleaned_tweets.append(tw)
    return cleaned_tweets


def post_process(text):
    for k, word in enumerate(text):
        # stanford corenlp result post-processing
        # 去空
        if not text[k] or re.match(p12, text[k]):  # 这回匹配到4个，看词表里还会不会出现空
            text.pop(k)
            if k >= len(text):
                break
        # 去除表情符号和不常见标点
        # 只有pop没有insert就自动进入下一个词了，其他筛选条件需要在这之后
        if re.match(p2, text[k]):
            text.pop(k)
            if k >= len(text):
                break
        if re.match(p8, text[k]) or text[k] == '':  # 一些表情符号没去除干净的，人工去除（又筛除101个）
            text.pop(k)
            if k >= len(text):
                break

        # 把网址转换成 'urlurlurl'
        if 'http' in text[k] or 'www.' in text[k] or '.com' in text[
            k]:  # 只有第一个可以用word，后面只能修改了的当前位置的词
            text[k] = 'urlurlurl'

        # 按/切开
        a = re.match(p3, text[k])
        temp_i = k
        while a:
            flag = 1
            text.pop(temp_i)
            text.insert(temp_i, a.group(1))
            temp_i += 1
            text.insert(temp_i, '/')
            temp_i += 1
            latter = a.group(2)
            a = re.match(p3, latter)
        if temp_i > k:
            text.insert(temp_i, latter)

        # 按@切开
        b = re.match(p4, text[k])
        if b:
            text.pop(k)
            text.insert(k, b.group(1))
            text.insert(k + 1, '@' + b.group(2))
        # 按.切开
        c = re.match(p5, text[k])
        temp_i = k
        while c:
            text.pop(temp_i)
            text.insert(temp_i, c.group(1))
            temp_i += 1
            text.insert(temp_i, '.')
            temp_i += 1
            latter = c.group(2)
            c = re.match(p5, latter)
        if temp_i > k:
            text.insert(temp_i, latter)
        # 按?切开
        d = re.match(p6, text[k])
        if d:
            text.pop(k)
            text.insert(k, d.group(1))
            text.insert(k + 1, '?')
            text.insert(k + 2, d.group(2))
        # 把'urlurlurl'和其他的内容切分开
        e = re.match(p9, text[k])
        if e:
            text.pop(k)
            if e.group(1):
                text.insert(k, e.group(1))
                text.insert(k + 1, 'urlurlurl')
                text.insert(k + 2, e.group(2))
            else:
                text.insert(k, 'urlurlurl')
                text.insert(k + 1, e.group(2))
        f = re.match(p10, text[k])
        if f:
            text.pop(k)
            text.insert(k, f.group(1))
            text.insert(k + 1, 'urlurlurl')
            if f.group(2):
                text.insert(k + 2, f.group(2))
        # 把多个连起来的标点符号切分开
        if re.match(p1, text[k]):
            temp_word = text[k]
            text.pop(k)
            l = len(temp_word)
            for x in range(l):
                text.insert(k + x, temp_word[x])
        # 将所有带大写字母的词转为小写
        if re.findall(p7, text[k]):
            text[k] = text[k].lower()
        # 把@user转化成@
        if re.match(p1, text[k]):
            text.pop(k)
            text.insert(k, '@')
    return text


def examine_new_tweets(cleaned_tweets):
    # labels: 0:support,1:comment,2:deny,3:query
    premise = []
    hypothesis = []
    labels = []
    ids = []
    random.shuffle(cleaned_tweets)
    for i,tw in enumerate(cleaned_tweets):
        premise.append(tw['text'])
        hypothesis.append(tw['text'])
        labels.append(0)  # 替换成各类label对应的数字
        ids.append(i)
    # 放到dev里判断是否输出和预期相同的label
    np.save('data/taskA_examine_support_3/dev/A_premise', premise)
    np.save('data/taskA_examine_support_3/dev/A_hypothesis', hypothesis)
    np.save('data/taskA_examine_support_3/dev/A_labels', labels)
    np.save('data/taskA_examine_support_3/dev/A_id_set', ids)



def combine_new_tweets():
    # labels: 0:support,1:comment,2:deny,3:query
    # filter new data
    all_new_premise = np.load('data/taskA_examine_support_3/dev/A_premise.npy')
    all_new_hypothesis = np.load('data/taskA_examine_support_3/dev/A_hypothesis.npy')
    all_new_labels = np.load('data/taskA_examine_support_3/dev/A_labels.npy')
    filtered_new_tweets = []
    f = open('documents/support_examine_3.tsv', 'r')
    lines = f.readlines()[1:]
    f.close()
    for i in range(len(lines)):
        line = lines[i].strip()
        result = re.match(r'([a-z\d]+)\s(\d)\s\[(.*?)\]', line)
        if int(result.group(2))==0:  # 替换成各类label对应的数字
            tw = {}
            id_str = int(result.group(1))
            tw['p'] = all_new_premise[id_str]
            tw['h'] = all_new_hypothesis[id_str]
            tw['label'] = all_new_labels[id_str]
            filtered_new_tweets.append(tw)

    # merge new & old train data  --在加了一个类的基础上加另一个类的数据
    premise = np.load('data/taskA_add_deny_1/train/A_premise.npy').tolist()
    hypothesis = np.load('data/taskA_add_deny_1/train/A_hypothesis.npy').tolist()
    labels = np.load('data/taskA_add_deny_1/train/A_labels.npy').tolist()
    for tw in filtered_new_tweets:
        premise.append(tw['p'])
        hypothesis.append(tw['h'])
        labels.append(tw['label'])

    # re-organize merged data
    reorganized_tweets = []
    for i in range(len(premise)):
        tweet={}
        tweet['p']=premise[i]
        tweet['h']=hypothesis[i]
        tweet['label']=labels[i]
        reorganized_tweets.append(tweet)
    random.shuffle(reorganized_tweets)

    # save reorganized merged data
    all_premise = []
    all_hypothesis = []
    all_labels = []
    for i in range(len(reorganized_tweets)):
        all_premise.append(reorganized_tweets[i]['p'])
        all_hypothesis.append(reorganized_tweets[i]['h'])
        all_labels.append(reorganized_tweets[i]['label'])
    np.save('data/taskA_add_support_3/train/A_premise', all_premise)
    np.save('data/taskA_add_support_3/train/A_hypothesis', all_hypothesis)
    np.save('data/taskA_add_support_3/train/A_labels', all_labels)


def error_analysis():
    #dev_hs = np.load('data/taskA_add_support_3/dev/A_hypothesis.npy')
    #dev_labels = np.load('data/taskA_add_support_3/dev/A_labels.npy')
    dev_hs = np.load('data/taskB_data_0102_full/dev/B_hypothesis.npy')
    dev_labels = np.load('data/taskB_data_0102_full/dev/B_labels.npy')
    dev_branches = np.load('data/taskA_data_0103/dev/tokenized_branch_id_A_B_label_stanford.npy')
    f = open('error_analysis/taskB_medium_7630.tsv', 'r')
    lines = f.readlines()[1:]
    f.close()
    ids = []
    texts = []
    #source_texts = []
    predict_labels = []
    true_labels = []
    A_labels = {0:'support',1:'comment',2:'deny',3:'query'}
    B_labels = {0: 'true', 1: 'false', 2: 'unverified'}
    for i in range(len(lines)):
        line = lines[i].strip()
        result = re.match(r'([a-z\d]+)\s(\d)\s\[(.*?)\]', line)
        pred_label = int(result.group(2))
        true_label = int(dev_labels[i])
        if pred_label != true_label:
            id_str = result.group(1)
            ids.append(id_str)
            my_str = ' '
            '''
            # taskA
            src=''
            for branch in dev_branches:
                for tw in branch:
                    if tw['id']==id_str:
                        src=branch[0]['text']
                        source_texts.append(my_str.join(src))
                        break
                if src:
                    break
            '''
            texts.append(my_str.join(dev_hs[i]))
            predict_labels.append(B_labels[pred_label])
            true_labels.append(B_labels[true_label])

    #print(len(source_texts))
    #print(len(texts))

    # 写入csv
    # 字典中的key值即为csv中列名
    '''
    # taskA
    dataframe = pd.DataFrame(
        {'id': ids, 'text': texts, 'source_texts':source_texts,'true_label': true_labels,
         'pred_label': predict_labels})
    '''
    dataframe = pd.DataFrame(
        {'id': ids, 'text': texts, 'true_label': true_labels,
         'pred_label': predict_labels})

    # 将DataFrame存储为csv,index表示是否显示行名，default=True
    dataframe.to_csv("error_analysis/error_taskB_medium_7630.csv",index=True,sep=',')


if __name__ == "__main__":
    # 在31上分词
    #nlp = StanfordCoreNLP(r'/home/ruoyao/ruoyao/stanford-corenlp-full-2018-10-05')
    #eng_word_tokenize(nlp)
    # 处理后让GPT预测筛选
    #cleaned_tweets = clean_tokenized_text()
    #examine_new_tweets(cleaned_tweets)
    # 筛选后加入到原本的训练数据里
    #combine_new_tweets()
    # 错误分析
    #error_analysis()