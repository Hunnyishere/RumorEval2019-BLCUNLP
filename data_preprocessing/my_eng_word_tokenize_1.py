import numpy as np
import os
from stanfordcorenlp import StanfordCoreNLP


def eng_word_tokenize():
    tokenized_branch_tweet_id = {}
    tokenized_branch_tweet_id['train']=[]
    tokenized_branch_tweet_id['dev']=[]
    tokenized_branch_tweet_id['test']=[]

    whichset = ['train', 'dev','test']  # 19rumor

    support_ids = []
    support_content = []
    deny_ids = []
    deny_content = []
    query_ids = []
    query_content = []
    comment_ids = []
    comment_content = []
    true_ids = []
    true_content = []
    false_ids = []
    false_content = []
    unverified_ids = []
    unverified_content = []

    for sset in whichset:
        print(sset)
        branches_text_in_one_set = np.load('saved_data_19/' + sset + '/branch_id_A_B_label.npy',encoding='bytes')
        for j, branch in enumerate(branches_text_in_one_set):
            #source = branch[0]
            branch_temp = []
            for i, tweet in enumerate(branch):
                tweet_temp = {}
                # stanford corenlp  --大概1分钟？
                #print('\nid', tweet[0])
                tweet_temp['id']=tweet[0]
                tweet_temp['text'] = nlp.word_tokenize(tweet[1])
                if sset=='test':
                    tweet_temp['A_label'] = -1
                    tweet_temp['B_label'] = -1
                else:
                    tweet_temp['A_label'] = int(tweet[2])
                    tweet_temp['B_label'] = int(tweet[3])
                #print('favorite',tweet[4])
                tweet_temp['favorite'] = int(tweet[4])
                #print('retweet',tweet[5])
                tweet_temp['retweet'] = int(tweet[5])
                #print('depth', tweet[6])
                tweet_temp['depth'] = int(tweet[6])
                #print('user_verified', tweet[7])
                tweet_temp['user_verified'] = tweet[7]
                #print('user_entities_description', tweet[8])
                tweet_temp['user_entities_description'] = tweet[8]
                #print('user_entities_url', tweet[9])
                tweet_temp['user_entities_url'] = tweet[9]
                #print('user_followers', tweet[10])
                tweet_temp['user_followers'] = int(tweet[10])
                #print('user_listed', tweet[11])
                tweet_temp['user_listed'] = int(tweet[11])
                #print('user_statuses', tweet[12])
                tweet_temp['user_statuses'] = int(tweet[12])
                #print('user_description', tweet[13])
                tweet_temp['user_description'] = tweet[13]
                #print('user_friends', tweet[14])
                tweet_temp['user_friends'] = int(tweet[14])
                #print('user_default_profile', tweet[15])
                tweet_temp['user_default_profile'] = tweet[15]
                #print('selftext', tweet[16])
                tweet_temp['selftext'] = tweet[16]
                #print('kind', tweet[17])
                tweet_temp['kind'] = tweet[17]
                #print('archived', tweet[18])
                tweet_temp['archived'] = tweet[18]
                #print('edited', tweet[19])
                tweet_temp['edited'] = tweet[19]
                #print('is_submitter', tweet[20])
                tweet_temp['is_submitter'] = tweet[20]

                '''
                # 根据label存数据，进行观察
                # labelA: 0:support; 1:comment; 2:deny; 3:query
                # labelB: 0:true; 1:false; 2:unverified
                if tweet_temp['A_label'] == 0 and tweet_temp['id'] not in support_ids:
                    support_ids.append(tweet_temp['id'])
                    support_content.append(tweet_temp)
                if tweet_temp['A_label'] == 1 and tweet_temp['id'] not in comment_ids:
                    comment_ids.append(tweet_temp['id'])
                    comment_content.append(tweet_temp)
                if tweet_temp['A_label'] == 2 and tweet_temp['id'] not in deny_ids:
                    deny_ids.append(tweet_temp['id'])
                    deny_content.append(tweet_temp)
                if tweet_temp['A_label'] == 3 and tweet_temp['id'] not in query_ids:
                    query_ids.append(tweet_temp['id'])
                    query_content.append(tweet_temp)
                if i==0:
                    if tweet_temp['B_label'] == 0 and tweet_temp['id'] not in true_ids:
                        true_ids.append(tweet_temp['id'])
                        true_content.append(tweet_temp)
                    if tweet_temp['B_label'] == 1 and tweet_temp['id'] not in false_ids:
                        false_ids.append(tweet_temp['id'])
                        false_content.append(tweet_temp)
                    if tweet_temp['B_label'] == 2 and tweet_temp['id'] not in unverified_ids:
                        unverified_ids.append(tweet_temp['id'])
                        unverified_content.append(tweet_temp)
                '''

                branch_temp.append(tweet_temp)

            tokenized_branch_tweet_id[sset].append(branch_temp)

        np.save(os.path.join('saved_data_19',sset,'tokenized_branch_id_A_B_label_stanford'), tokenized_branch_tweet_id[sset])

    '''
    np.save('train+dev_tokenized_data/support_content',support_content)
    np.save('train+dev_tokenized_data/deny_content', deny_content)
    np.save('train+dev_tokenized_data/query_content', query_content)
    np.save('train+dev_tokenized_data/comment_content', comment_content)
    np.save('train+dev_tokenized_data/true_content', true_content)
    np.save('train+dev_tokenized_data/false_content', false_content)
    np.save('train+dev_tokenized_data/unverified_content', unverified_content)
    '''




if __name__ == "__main__":
    nlp = StanfordCoreNLP(r'/home/ruoyao/ruoyao/stanford-corenlp-full-2018-10-05')
    eng_word_tokenize()