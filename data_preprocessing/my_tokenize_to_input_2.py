import numpy as np
import re
import os


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



def clean_tokenized_text():
    cleaned_tokenized_branch_tweet_id = {}
    cleaned_tokenized_branch_tweet_id['train']=[]
    cleaned_tokenized_branch_tweet_id['dev']=[]
    cleaned_tokenized_branch_tweet_id['test'] = []

    whichset = ['train', 'dev','test']  # 19rumor

    for sset in whichset:
        tokenized_branch_tweet_id = np.load('data/taskA_newformdata/' + sset + '/tokenized_branch_id_A_B_label_stanford.npy')
        for j, branch in enumerate(tokenized_branch_tweet_id):
            branch_temp = []
            for i, tweet in enumerate(branch):
                # for i,x in enumerate(a): --是动态的，a的长度动态变化，就遍历到变化后的最后一个。
                # 但i一直在当前基础上+1，所以至少在i+1的位置插入，接下来才能遍历到。
                # 在k位置插入，刚好就是当前遍历的内容(需不断更新word)，新插入的部分(k+1,k+2)会在接下来遍历到。
                for k,word in enumerate(tweet['text']):
                    # stanford corenlp result post-processing
                    # 去空
                    if not tweet['text'][k] or re.match(p12,tweet['text'][k]):  # 这回匹配到4个，看词表里还会不会出现空
                        tweet['text'].pop(k)
                        if k >= len(tweet['text']):
                            break
                    # 去除表情符号和不常见标点
                    # 只有pop没有insert就自动进入下一个词了，其他筛选条件需要在这之后
                    if re.match(p2, tweet['text'][k]):  # 已经筛掉535个了，知足吧= =
                        tweet['text'].pop(k)
                        if k >= len(tweet['text']):  # 只pop不insert，需要判断是否pop之后没有下一个词了，应当退出此次循环
                            break
                    if re.match(p8, tweet['text'][k]) or tweet['text'][k] == '':  # 一些表情符号没去除干净的，人工去除（又筛除101个）
                        tweet['text'].pop(k)
                        if k >= len(tweet['text']):
                            break

                    # 把网址转换成 'urlurlurl'
                    if 'http' in tweet['text'][k] or 'www.' in tweet['text'][k] or '.com' in tweet['text'][k]:  # 只有第一个可以用word，后面只能修改了的当前位置的词
                        tweet['text'][k] = 'urlurlurl'

                    # 按/切开
                    a = re.match(p3, tweet['text'][k])
                    temp_i = k
                    while a:
                        flag = 1
                        tweet['text'].pop(temp_i)
                        tweet['text'].insert(temp_i, a.group(1))
                        temp_i += 1
                        tweet['text'].insert(temp_i, '/')
                        temp_i += 1
                        latter = a.group(2)
                        a = re.match(p3, latter)
                    if temp_i > k:
                        tweet['text'].insert(temp_i, latter)

                    # 按@切开
                    b = re.match(p4,tweet['text'][k])
                    if b:
                        tweet['text'].pop(k)
                        tweet['text'].insert(k, b.group(1))
                        tweet['text'].insert(k+1, '@'+b.group(2))
                    # 按.切开
                    c = re.match(p5, tweet['text'][k])
                    temp_i = k
                    while c:
                        tweet['text'].pop(temp_i)
                        tweet['text'].insert(temp_i, c.group(1))
                        temp_i += 1
                        tweet['text'].insert(temp_i, '.')
                        temp_i += 1
                        latter = c.group(2)
                        c = re.match(p5, latter)
                    if temp_i > k:
                        tweet['text'].insert(temp_i, latter)
                    # 按?切开
                    d = re.match(p6, tweet['text'][k])
                    if d:
                        tweet['text'].pop(k)
                        tweet['text'].insert(k, d.group(1))
                        tweet['text'].insert(k + 1, '?')
                        tweet['text'].insert(k + 2, d.group(2))
                    # 把'urlurlurl'和其他的内容切分开
                    e = re.match(p9, tweet['text'][k])
                    if e:
                        tweet['text'].pop(k)
                        if e.group(1):
                            tweet['text'].insert(k, e.group(1))
                            tweet['text'].insert(k + 1, 'urlurlurl')
                            tweet['text'].insert(k + 2, e.group(2))
                        else:
                            tweet['text'].insert(k, 'urlurlurl')
                            tweet['text'].insert(k + 1, e.group(2))
                    f = re.match(p10, tweet['text'][k])
                    if f:
                        tweet['text'].pop(k)
                        tweet['text'].insert(k, f.group(1))
                        tweet['text'].insert(k + 1, 'urlurlurl')
                        if f.group(2):
                            tweet['text'].insert(k + 2, f.group(2))
                    # 把多个连起来的标点符号切分开
                    if re.match(p1, tweet['text'][k]):
                        temp_word = tweet['text'][k]
                        tweet['text'].pop(k)
                        l = len(temp_word)
                        for x in range(l):
                            tweet['text'].insert(k + x, temp_word[x])
                    # 将所有带大写字母的词转为小写
                    if re.findall(p7,tweet['text'][k]):
                        tweet['text'][k] = tweet['text'][k].lower()
                    # 把@user转化成@
                    if re.match(p1, tweet['text'][k]):
                        tweet['text'].pop(k)
                        tweet['text'].insert(k, '@')

                branch_temp.append(tweet)
            cleaned_tokenized_branch_tweet_id[sset].append(branch_temp)

        np.save(os.path.join('data/featured_data_0130/', sset, 'cleaned_tokenized_branch_A_B_label_stanford'), cleaned_tokenized_branch_tweet_id[sset])


if __name__ == "__main__":
    clean_tokenized_text()