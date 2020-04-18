# -*- coding: utf-8 -*-

# read a vocab and precompute a .npy embedding matrix.
# if a vocab entry is in the provided glove embeddings then use the glove data.
# if it's not, generate a random vector but scale it to the median length of the glove embeddings.
# reserve row 0 in the matrix for the PAD embedding (always set to {0})
# reserve row 1 in the matrix for the UNK embedding (given a random value)
import argparse
import numpy as np
import sys
from sklearn import random_projection
import gensim
import nltk

parser = argparse.ArgumentParser()
parser.add_argument("--vocab", required=True, help="reference vocab of non glove data; token \t idx")
parser.add_argument("--npy", required=True, help="npy output")
parser.add_argument("--random-projection-dimensionality", default=None, type=float,
                    help="if set we randomly project the glove data to a smaller dimensionality")
opts = parser.parse_args()


class GetEmbedding(object):
    def __init__(self):
        self.opts = opts


    def init_embedding(self):
        # slurp vocab entries. assume idxs are valid, ie 1 < i < |v|, no dups, no gaps, etc
        # (recall reserving 0 for UNK)
        self.vocab = {}  # token => idx
        for line in open(self.opts.vocab, "r"):
            token, idx = line.strip().split("\t")
            if idx == 0:
                assert token == '_PAD', "expecting to reserve 0 for _PAD"
            elif idx == 1:
                assert token == '_UNK', "expecting to reserve 1 for _UNK"
            elif idx == 2:
                assert token == '_GO', "expecting to reverse 2 for _GO"
            elif idx == 3:
                assert token == '_EOS', "expecting to reverse 3 for _EOS"
            else:
                self.vocab[token] = int(idx)  # self.vocab是个字典：key=token, value=id
        print ("vocab has", len(self.vocab), "entries (not _PAD or _UNK or _GO or _EOS)")

        # alloc output after we see first glove embedding (so we know it's dimensionality)
        self.embeddings = None
        self.glove_dimensionality = None

        # pass over glove data copying data into embedddings array
        # for the cases where the token is in the reference vocab.
        self.tokens_requiring_random = set(self.vocab.keys())
        self.glove_embedding_norms = []


    def get_qmark(self, sent):
        # 标点符号
        self.feature_int_dict['hasqmark'] = 0
        self.feature_str_dict['hasqmark'] = 'None'
        for token in sent:
            if token.find('?') >= 0:  # find()，找到就返回第一个字符的下标，找不到就返回-1
                self.feature_int_dict['hasqmark'] = 1
                self.feature_str_dict['hasqmark'] = 'question_mark'
                break

    def get_emark(self, sent):
        self.feature_int_dict['hasemark'] = 0
        self.feature_str_dict['hasemark'] = 'None'
        for token in sent:
            if token.find('!') >= 0:
                self.feature_int_dict['hasemark'] = 1
                self.feature_str_dict['hasemark'] = 'exclamation_mark'
                break

    def get_period(self, sent):
        self.feature_int_dict['hasperiod'] = 0
        self.feature_str_dict['hasperiod'] = 'None'
        for token in sent:
            if token.find('.') >= 0:
                self.feature_int_dict['hasperiod'] = 1
                self.feature_str_dict['hasperiod'] = 'period'
                break

    def get_hashtag(self, sent):
        self.feature_int_dict['hashashtag'] = 0
        self.feature_str_dict['hashashtag'] = 'None'
        for token in sent:
            if token.find('#') >= 0:
                self.feature_int_dict['hashashtag'] = 1
                self.feature_str_dict['hashashtag'] = 'hashtag'
                break

    def get_usermention(self, sent):
        self.feature_int_dict['hasusermention'] = 0
        self.feature_str_dict['hasusermention'] = 'None'
        for token in sent:
            if token.find('@') >= 0:
                self.feature_int_dict['hasusermention'] = 1
                self.feature_str_dict['hasusermention'] = 'user_mention'
                break

    def get_url(self, sent):
        self.feature_int_dict['hasurl'] = 0
        self.feature_str_dict['hasurl'] = 'None'
        for token in sent:
            if token.find('urlurlurl') >= 0 or token.find('http') >= 0:
                self.feature_int_dict['hasurl'] = 1
                self.feature_str_dict['hasurl'] = 'url'
                break

    '''
    def get_pic(self, sent):  
        self.feature_int_dict['haspic'] = 0
        self.feature_str_dict['haspic'] = 'None'
        for token in sent:
            if (token.find('picpicpic') >= 0) or (token.find('pic.twitter.com') >= 0) or (token.find('instagr.am') >= 0):
                self.feature_int_dict['haspic'] = 1
                self.feature_str_dict['haspic'] = 'picture'
                break
    '''

    '''
    # 词表已经全部转成小写了怎么计算大写比例啊？
    def word_capital(self, token):
        l = len(token)
        uppers = [l for l in token if l.isupper()]
        if l != 0:
            self.feature_int_dict['capitalratio'] = float(len(uppers)) / l
        else:
            self.feature_int_dict['capitalratio'] = 0
    '''

    def get_RT(self, sent):
        # 该条tweet是否是转发(retweet)
        self.feature_int_dict['hasrt'] = 0
        self.feature_str_dict['hasrt'] = 'None'
        rtwords = ['rt']
        for token in sent:
            if token in rtwords:
                self.feature_int_dict['hasrt'] = 1
                self.feature_str_dict['hasrt'] = 'rt'
                break

    def get_positivewords(self, sent):
        self.feature_int_dict['haspositive'] = 0
        self.feature_str_dict['haspositive'] = 'None'
        # 肯定词
        positivewords = ['true','exactly','yes','indeed','omg','know']
        for token in sent:
            if token in positivewords:
                self.feature_int_dict['haspositive'] = 1
                self.feature_str_dict['haspositive'] = 'positive'
                #print('haspositive', token)
                break
        '''
        for token in sent:
            for positiveword in positivewords:
                if token.find(positiveword) >= 0:
                    self.feature_int_dict['haspositive'] = 1
                    self.feature_str_dict['haspositive'] = 'positive'
                    print('haspositive',positiveword,token)
                    break
            break
        '''

    def get_positiveSad(self, sent):
        self.feature_int_dict['haspositiveSad'] = 0
        self.feature_str_dict['haspositiveSad'] = 'None'
        positive_sad = ['sad','saddest','cry','disturbing']
        for token in sent:
            if token in positive_sad:
                self.feature_int_dict['haspositiveSad'] = 1
                self.feature_str_dict['haspositiveSad'] = 'positive_sad'
                break

    def get_negationwords(self, sent):
        self.feature_int_dict['hasnegation'] = 0
        self.feature_str_dict['hasnegation'] = 'None'
        # 否定词 (最后2个不确定有用)
        negationwords = ['not', 'no', 'nobody', 'nothing', 'none', 'never','neither',
                         'nor', 'nowhere', 'hardly','scarcely', 'barely', 'dont','doesnt',
                         'isnt', 'arent','wasnt','werent','shouldnt', 'wouldnt', 'couldnt',
                         'hasnt','havent','didnt','aint','cant','wont','impossible','shut']
        for token in sent:
            if token in negationwords:
                self.feature_int_dict['hasnegation'] = 1
                self.feature_str_dict['hasnegation'] = 'negation'
                #print('hasnegation',token)
                break

    def get_negationwords2(self, sent):
        self.feature_int_dict['hasnegation2'] = 0
        self.feature_str_dict['hasnegation2'] = 'None'
        # 否定词 (最后2个不确定有用)
        negationwords = ['false','but','misunderstanding','different',
                         "n't",'fake','smear','worst','photoshop','stop','serious',
                         'doubt','rather','weird','stupid','shame','wrong',
                         'lie','lying','lies','lied','crazy','correction',
                         'misinformation','conspiracy', 'denying']
        for token in sent:
            if token in negationwords:
                self.feature_int_dict['hasnegation2'] = 1
                self.feature_str_dict['hasnegation2'] = 'negation2'
                break

    def get_swearwords(self, sent):
        swearwords = []
        with open('data/badwords.txt', 'r') as f:
            for line in f:
                swearwords.append(line.strip().lower())
        self.feature_int_dict['hasswearwords'] = 0
        self.feature_str_dict['hasswearwords'] = 'None'
        for token in sent:
            if token in swearwords:
                self.feature_int_dict['hasswearwords'] = 1
                self.feature_str_dict['hasswearwords'] = 'swear'
                #print('hasswearwords',token)
                break

    def get_swearwords2(self, sent):
        self.feature_int_dict['hasswearwords2'] = 0
        self.feature_str_dict['hasswearwords2'] = 'None'
        swearwords = ['abuse','abused','abuses','shut','leave','quit','delete',
                      'stfu','sexist','assaulted','assault','assaults']
        for token in sent:
            if token in swearwords:
                self.feature_int_dict['hasswearwords2'] = 1
                self.feature_str_dict['hasswearwords2'] = 'swear2'
                break

    def get_querywords(self, sent):
        whwords = ['what', 'when', 'where', 'which', 'who', 'whom', 'whose', 'why',
                   'how']
        self.feature_int_dict['haswhwords'] = 0
        self.feature_str_dict['haswhwords'] = 'None'
        for token in sent:
            if token in whwords:
                self.feature_int_dict['haswhwords'] = 1
                self.feature_str_dict['haswhwords'] = 'question'
                #print('haswhwords',token)
                break

    # 词性标注，GPT要是一个句子里有多少个不同的词性加多少维的话，加的维度不固定，只在ESIM里加吧。
    def get_pos(self, token):  # 各个可能的词性列表，出现了对应的词性=1，其余=0
        self.feature_str_dict['pos_tag'] = 'None'
        self.feature_int_dict['pos_tag'] = 0  # 初始化
        t=[]
        t.append(token)
        postag_tuples = nltk.pos_tag(t)
        #self.feature_str_dict['pos_tag'] = postag_tuples[0][1]
        postag_list = [x[1] for x in postag_tuples]
        possible_postags = {'WRB':1, 'WP$':2, 'WP':3, 'WDT':4, 'VBZ':5, 'VBP':6, 'VBN':7,
                            'VBG':8, 'VBD':9, 'VB':10, 'UH':11, 'TO':12, 'SYM':13, 'RP':14, 'RBS':15,
                            'RBR':16, 'RB':17, 'PRP$':18, 'PRP':19, 'POS':20, 'PDT':21, 'NNS':22,
                            'NNPS':23, 'NNP':24, 'NN':25, 'MD':26, 'LS':27, 'JJS':28, 'JJR':29,
                            'JJ':30, 'IN':31, 'FW':32, 'EX':33, 'DT':34, 'CD':35, 'CC':36, '$':37}
        for tok in postag_list:
            if tok in possible_postags:
                self.feature_int_dict['pos_tag'] = possible_postags[tok]

    def false_syn_ant(self, sent):
        false_synonyms = ['false', 'bogus', 'deceitful', 'dishonest',
                          'distorted', 'erroneous', 'fake', 'fanciful',
                          'faulty', 'fictitious', 'fraudulent',
                          'improper', 'inaccurate', 'incorrect',
                          'invalid', 'misleading', 'mistaken', 'phony',
                          'specious', 'spurious', 'unfounded', 'unreal',
                          'untrue', 'untruthful', 'apocryphal',
                          'beguiling', 'casuistic', 'concocted',
                          'cooked-up', 'counterfactual',
                          'deceiving', 'delusive', 'ersatz',
                          'fallacious', 'fishy', 'illusive', 'imaginary',
                          'inexact', 'lying', 'mendacious',
                          'misrepresentative', 'off the mark', 'sham',
                          'sophistical', 'trumped up', 'unsound', 'liar']
        false_antonyms = ['accurate', 'authentic', 'correct', 'fair',
                          'faithful', 'frank', 'genuine', 'honest', 'moral',
                          'open', 'proven', 'real', 'right', 'sincere',
                          'sound', 'true', 'trustworthy', 'truthful',
                          'valid', 'actual', 'factual', 'just', 'known',
                          'precise', 'reliable', 'straight', 'substantiated']
        self.feature_int_dict['false_synonyms'] = 0
        self.feature_str_dict['false_synonyms'] = 'None'
        for token in sent:
            if token in false_synonyms:
                self.feature_int_dict['false_synonyms'] = 1
                self.feature_str_dict['false_synonyms'] = 'false_synonyms'
                #print('false_synonyms',token)
                break
        self.feature_int_dict['false_antonyms'] = 0
        self.feature_str_dict['false_antonyms'] = 'None'
        for token in sent:
            if token in false_antonyms:
                self.feature_int_dict['false_antonyms'] = 1
                self.feature_str_dict['false_antonyms'] = 'false_antonyms'
                #print('false_antonyms',token)
                break

    def other_word_types_taskA(self, sent):  # 38
        SpeechAct = {}
        SpeechAct['SpeechAct_ASK1'] = ['ask', 'request', 'beg', 'bespeech',
                                       'implore',
                                       'appeal', 'plead', 'intercede', 'apply',
                                       'urge', 'persuade', 'dissuade', 'convince']
        SpeechAct['SpeechAct_FORBID'] = ['forbid', 'prohibit', 'veto', 'refuse',
                                         'decline', 'reject', 'rebuff', 'renounce',
                                         'cancel', 'resign', 'dismiss']
        SpeechAct['SpeechAct_PERMIT'] = ['permit', 'allow', 'consent', 'accept',
                                         'agree', 'approve', 'disapprove',
                                         'authorize', 'appoint']
        SpeechAct['SpeechAct_REPRIMAND'] = ['reprimand', 'rebuke', 'reprove',
                                            'admonish', 'reproach', 'nag',
                                            'scold', 'abuse', 'insult']
        SpeechAct['SpeechAct_MOCK'] = ['ridicule', 'joke']
        SpeechAct['SpeechAct_ACCUSE'] = ['accuse', 'charge', 'challenge', 'defy',
                                         'dare']
        SpeechAct['SpeechAct_WARN '] = ['warn', 'threaten', 'blackmail']
        SpeechAct['SpeechAct_PRAISE '] = ['praise', 'commend', 'compliment',
                                          'boast', 'credit']
        SpeechAct['SpeechAct_PROMISE '] = ['promise', 'pledge', 'vow', 'swear',
                                           'vouch for', 'guarante']
        SpeechAct['SpeechAct_COMPLAIN'] = ['complain', 'protest', 'object',
                                           'moan', 'bemoan', 'lament', 'bewail']
        SpeechAct['SpeechAct_EXCLAIM'] = ['exclaim', 'enthuse', 'exult', 'swear',
                                          'blaspheme']
        SpeechAct['SpeechAct_HINT'] = ['hint', 'imply', 'insinuate']
        SpeechAct['SpeechAct_CONCLUDE'] = ['conclude', 'deduce', 'infer', 'gather',
                                           'reckon', 'estimate', 'calculate',
                                           'count', 'prove', 'compare']
        SpeechAct['SpeechAct_INFORM'] = ['inform', 'notify', 'announce',
                                         'inform on', 'reveal']
        SpeechAct['SpeechAct_STRESS'] = ['stress', 'emphasize', 'insist', 'repeat',
                                         'point out', 'note', 'remind', 'add']
        SpeechAct['SpeechAct_ANSWER'] = ['answer', 'reply']
        for k in SpeechAct.keys():
            self.feature_int_dict[k] = 0
            self.feature_str_dict[k] = 'None'
            for token in sent:
                if token in SpeechAct[k]:
                    self.feature_int_dict[k] = 1
                    self.feature_str_dict[k] = k
                    #print(k,token)
                    break

    def other_word_types_taskB(self, sent):  # 15
        SpeechAct = {}
        SpeechAct['SpeechAct_ORDER'] = ['command', 'demand', 'tell', 'direct',
                                        'instruct', 'require', 'prescribe',
                                        'order']
        SpeechAct['SpeechAct_ASK1'] = ['ask', 'request', 'beg', 'bespeech',
                                       'implore',
                                       'appeal', 'plead', 'intercede', 'apply',
                                       'urge', 'persuade', 'dissuade', 'convince']
        SpeechAct['SpeechAct_ASK2'] = ['ask', 'inquire', 'enquire', 'interrogate',
                                       'question', 'query']
        SpeechAct['SpeechAct_PERMIT'] = ['permit', 'allow', 'consent', 'accept',
                                         'agree', 'approve', 'disapprove',
                                         'authorize', 'appoint']
        SpeechAct['SpeechAct_ARGUE'] = ['argue', 'disagree', 'refute', 'contradict',
                                        'counter', 'deny', 'recant', 'retort',
                                        'quarrel', 'oppose']
        SpeechAct['SpeechAct_ACCUSE'] = ['accuse', 'charge', 'challenge', 'defy',
                                         'dare']
        SpeechAct['SpeechAct_ADVISE '] = ['advise', 'councel', 'consult',
                                          'recommend', 'suggest', 'propose',
                                          'advocate']
        SpeechAct['SpeechAct_OFFER '] = ['offer', 'volunteer', 'grant', 'give']
        SpeechAct['SpeechAct_PRAISE '] = ['praise', 'commend', 'compliment',
                                          'boast', 'credit']
        SpeechAct['SpeechAct_FORGIVE '] = ['forgive', 'excuse', 'justify',
                                           'absolve', 'pardon', 'convict',
                                           'acquit', 'sentence']
        SpeechAct['SpeechAct_ASSERT'] = ['assert', 'affirm', 'claim', 'maintain',
                                         'contend', 'state', 'testify']
        SpeechAct['SpeechAct_CONFIRM'] = ['confirm', 'assure', 'reassure', 'definitely', 'admit']
        SpeechAct['SpeechAct_STRESS'] = ['stress', 'emphasize', 'insist', 'repeat',
                                         'point out', 'note', 'remind', 'add']
        SpeechAct['SpeechAct_DECLARE'] = ['declare', 'pronounce', 'proclaim',
                                          'decree', 'profess', 'vote', 'resolve',
                                          'decide']
        SpeechAct['SpeechAct_BAPTIZE'] = ['baptize', 'chirsten', 'name',
                                          'excommunicate']
        for k in SpeechAct.keys():
            self.feature_int_dict[k] = 0
            self.feature_str_dict[k] = 'None'
            for token in sent:
                if token in SpeechAct[k]:
                    self.feature_int_dict[k] = 1
                    self.feature_str_dict[k] = k
                    #print(k,token)
                    break

    def get_word_features(self,token):
        self.feature_int_dict = {}
        self.feature_str_dict = {}
        sent = []
        sent.append(token)
        self.get_punctuations(sent)
        self.get_accessories(sent)
        self.get_positivewords(sent)
        self.get_negationwords(sent)
        self.get_swearwords(sent)
        self.get_querywords(sent)
        self.get_pos(token)
        self.other_word_types(sent)
        self.false_syn_ant(sent)

    def create_taskA_word_features(self,sent):
        self.feature_int_dict = {}
        self.feature_str_dict = {}
        self.get_qmark(sent)
        self.get_hashtag(sent)
        self.get_url(sent)
        self.get_RT(sent)
        self.get_positiveSad(sent)
        self.get_negationwords(sent)
        self.get_negationwords2(sent)
        self.get_swearwords(sent)
        self.get_swearwords2(sent)
        self.get_querywords(sent)
        self.other_word_types_taskA(sent)
        word_features = []
        for feature_value in self.feature_str_dict.values():
            if feature_value!='None':
                word_features.append(feature_value)
        return word_features

    def create_taskB_word_features(self,sent):
        self.feature_int_dict = {}
        self.feature_str_dict = {}
        self.get_emark(sent)
        self.get_positivewords(sent)
        self.get_negationwords(sent)
        self.get_querywords(sent)
        self.false_syn_ant(sent)
        self.other_word_types_taskA(sent)
        word_features = []
        for feature_value in self.feature_str_dict.values():
            if feature_value != 'None':
                word_features.append(feature_value)
        return word_features


    def get_embedding(self):
        self.init_embedding()
        self.loadW2vModel()
        # 生成词向量矩阵
        for token in self.vocab.keys():
            if token in self.model:
                glove_embedding = self.model[token]
                if self.embeddings is None:
                    self.glove_dimensionality = len(glove_embedding)
                    self.embeddings = np.empty((len(self.vocab), self.glove_dimensionality),
                                          dtype=np.float32)  # +1 for pad & unk
                assert len(glove_embedding) == self.glove_dimensionality, "differing dimensionality in glove data?"
                # 把Google News embedding赋给词表里的token
                self.embeddings[self.vocab[token]] = glove_embedding
                self.tokens_requiring_random.remove(token)  # 全部token去掉能在glove中找到的
                self.glove_embedding_norms.append(np.linalg.norm(glove_embedding))

        # given these embeddings we can calculate the median norm of the glove data
        self.median_glove_embedding_norm = np.median(self.glove_embedding_norms)

        print >> sys.stderr, "after passing over glove there are", len(self.tokens_requiring_random), \
        "tokens requiring a random alloc"

        # assign PAD and UNK random embeddings (pre projection)
        self.embeddings[0] = self.random_embedding()  # PAD
        self.embeddings[1] = self.random_embedding()  # UNK

        # assign random projections for every other fields requiring it
        for token in self.tokens_requiring_random:
            self.embeddings[self.vocab[token]] = self.random_embedding()

        # randomly project (if configured to do so)
        if opts.random_projection_dimensionality is not None:
            # assign a temp random embedding for PAD before projection (and zero it after)
            p = random_projection.GaussianRandomProjection(n_components=opts.random_projection_dimensionality)
            self.embeddings = p.fit_transform(self.embeddings)

        # zero out PAD embedding  # PAD的embedding为全0
        self.embeddings[0] = [0] * self.embeddings.shape[1]

        '''
        # 生成特征矩阵
        self.feature_embedding = None
        for token in self.vocab.keys():
            tweet_rep = []
            self.get_word_features(token)
            if self.feature_embedding is None:
                self.feature_length = len(self.feature_int_dict)
                #print(self.feature_int_dict.keys())
                self.feature_embedding = np.empty((len(self.vocab), self.feature_length), dtype=np.float32)
            for feature_name in self.feature_int_dict:
                tweet_rep.append(self.feature_int_dict[feature_name])  # 都是一维标量
            tweet_rep = np.asarray(tweet_rep)
            assert len(tweet_rep) == self.feature_embedding.shape[1], "differing dimensionality in %d and %s ?"%(self.feature_length, self.feature_int_dict)
            # 300维词向量的意思是第0阶上的shape是300，即(300,xx,..)
            self.feature_embedding[self.vocab[token]] = tweet_rep
        print('shape of feature embedding is',self.feature_embedding.shape)  # (12913,89)

        # 第1阶上拼接词向量矩阵和特征矩阵
        self.embeddings = np.concatenate((self.embeddings,self.feature_embedding),axis=1)
        '''

        # write embeddings npy to disk 写入snli_embedding.npy
        np.save(opts.npy, self.embeddings)  # (12913,389)


    # return a random embedding with the same norm as the glove data median norm
    def random_embedding(self):
        random_embedding = np.random.randn(1, self.glove_dimensionality)
        random_embedding /= np.linalg.norm(random_embedding)
        random_embedding *= self.median_glove_embedding_norm
        return random_embedding

    def loadW2vModel(self):
        # LOAD PRETRAINED MODEL  --Google News dataset(300d)
        print("Loading the model")
        model_GN = gensim.models.KeyedVectors.load_word2vec_format(
            '../rumorEval_data/GoogleNews-vectors-negative300.bin', binary=True)
        print("Done!")
        self.model = model_GN



def main():
    my_embedding = GetEmbedding()
    my_embedding.get_embedding()


if __name__ == "__main__":
    main()