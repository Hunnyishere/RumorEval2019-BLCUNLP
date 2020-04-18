import os
import csv
import numpy as np

from tqdm import tqdm

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

seed = 3535999445

def _rocstories(path):
    with open(path) as f:
        f = csv.reader(f)
        st = []
        ct1 = []
        ct2 = []
        y = []
        for i, line in enumerate(tqdm(list(f), ncols=80, leave=False)):
            if i > 0:
                s = ' '.join(line[1:5])
                c1 = line[5]
                c2 = line[6]
                st.append(s)
                ct1.append(c1)
                ct2.append(c2)
                y.append(int(line[-1])-1)
        return st, ct1, ct2, y

def rocstories(data_dir, n_train=1497, n_valid=374):
    #storys, comps1, comps2, ys = _rocstories(os.path.join(data_dir, 'ROC_tr_wy.csv'))
    #teX1, teX2, teX3, _ = _rocstories(os.path.join(data_dir, 'ROC_te_wy.csv'))
    storys, comps1, comps2, ys = _rocstories(os.path.join(data_dir, 'cloze_test_ALL_val.csv'))
    teX1, teX2, teX3, _ = _rocstories(os.path.join(data_dir, 'cloze_test_ALL_test.csv'))
    tr_storys, va_storys, tr_comps1, va_comps1, tr_comps2, va_comps2, tr_ys, va_ys = train_test_split(storys, comps1, comps2, ys, test_size=n_valid, random_state=seed)
    trX1, trX2, trX3 = [], [], []
    trY = []
    for s, c1, c2, y in zip(tr_storys, tr_comps1, tr_comps2, tr_ys):
        trX1.append(s)
        trX2.append(c1)
        trX3.append(c2)
        trY.append(y)

    vaX1, vaX2, vaX3 = [], [], []
    vaY = []
    for s, c1, c2, y in zip(va_storys, va_comps1, va_comps2, va_ys):
        vaX1.append(s)
        vaX2.append(c1)
        vaX3.append(c2)
        vaY.append(y)
    trY = np.asarray(trY, dtype=np.int32)
    vaY = np.asarray(vaY, dtype=np.int32)
    return (trX1, trX2, trX3, trY), (vaX1, vaX2, vaX3, vaY), (teX1, teX2, teX3)

def _ruoyao(path):
    source = np.load(os.path.join(path,'A_source.npy'))
    other_tw = np.load(os.path.join(path, 'A_other_tw.npy'))
    last_tw = np.load(os.path.join(path, 'A_last_tw.npy'))
    this_tw = np.load(os.path.join(path, 'A_this_tw.npy'))
    label = np.load(os.path.join(path, 'A_label.npy'))
    # 单词级别特征
    source_feature = np.load(os.path.join(path, 'A_source_word_feature.npy'))
    last_tw_feature = np.load(os.path.join(path, 'A_last_tw_word_feature.npy'))
    this_tw_feature = np.load(os.path.join(path, 'A_this_tw_word_feature.npy'))
    '''
    # 词+句子级别特征
    source_feature = np.load(os.path.join(path, 'A_source_word_sent_feature.npy'))
    last_tw_feature = np.load(os.path.join(path, 'A_last_tw_word_sent_feature.npy'))
    this_tw_feature = np.load(os.path.join(path, 'A_this_tw_word_sent_feature.npy'))
    '''
    source_features = []
    last_tw_features = []
    this_tw_features = []

    for i in range(len(source)):
        source[i] = ' '.join(source[i])
        other_tw[i] = ' '.join(other_tw[i])
        last_tw[i] = ' '.join(last_tw[i])
        this_tw[i] = ' '.join(this_tw[i])
        # 因为有int型，所以需要如下做：
        s = ''
        for j in source_feature[i]:
            s = s + ' ' + str(j)
        source_features.append(s)
        l = ''
        for j in last_tw_feature[i]:
            l = l + ' ' + str(j)
        last_tw_features.append(l)
        t = ''
        for j in this_tw_feature[i]:
            t = t + ' ' + str(j)
        this_tw_features.append(t)
    '''
    h = os.path.join(path,'B_featured_hypothesis.npy')
    p = os.path.join(path,'B_featured_premise.npy')
    l = os.path.join(path,'B_featured_labels.npy')
    hy = np.load(h)
    pre  = np.load(p)
    label = np.load(l)
    hys = []    
    for i in range(len(hy)):
        s = ''
        for j in hy[i]:
            #print(j)
            s = s + ' ' +str(j)
        hys.append(s)
    pres = []
    for i in range(len(pre)):
        s = ''
        for j in pre[i]:
            s = s + ' ' +str(j)
        pres.append(s)
    return pres,hys,label
    '''
    return source,source_features, other_tw, last_tw,last_tw_features, this_tw,this_tw_features, label

def ruoyao(data_dir):
    #p,h,l = _ruoyao(os.path.join(data_dir,'train'))
    #tr_comps1, va_comps1, tr_comps2, va_comps2, tr_ys, va_ys = train_test_split(p,h,l, test_size=0.15, random_state=seed)
    #trainset = (tr_comps1, tr_comps2, tr_ys)
    #devset = (va_comps1, va_comps2, va_ys)
    trainset = _ruoyao(os.path.join(data_dir,'train'))
    devset = _ruoyao(os.path.join(data_dir,'dev'))
    testset = _ruoyao(os.path.join(data_dir,'test'))
    return trainset,devset,testset
