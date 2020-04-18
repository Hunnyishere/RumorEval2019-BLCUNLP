import numpy as np

whichset = ['train', 'dev', 'test']
for sset in whichset:
    #print(sset, 'tweet content:')
    # 读取该集里各branch里的tweet数据
    # shape: train=(3030,25,xx);dev=(215,20,xx);test=(772,14,xx) 最后一维是tweet内容的长度，不确定
    branches_in_one_set = np.load('saved_data_ruoyao/'+sset+'/branch_tweet_arrays.npy')
    #print(branches_in_one_set)
    print(len(branches_in_one_set))
    #for j, branch in enumerate(branches_in_one_set):
        #for i, tweet in enumerate(branch):
            #print(i,':',tweet)

    # 读取该集里各branch的pad情况（前面有多少是有数据的，有数据为1，其余为0）
    # shape: train=(3030,25);dev=(215,20);test=(772,14)
    #print(sset, 'mask:')
    mask_in_one_set = np.load('saved_data_ruoyao/' + sset + '/mask.npy')
    #for j, branch_mask in enumerate(mask_in_one_set):
        #print(j, ':', branch_mask)

    # 读取该集里各branch中各tweet的label（label是pad过的，后面没有的是0）
    # shape: train=(3030,25);dev=(215,20);test=(772,14)
    #print(sset, 'label:')
    label_in_one_set = np.load('saved_data_ruoyao/' + sset + '/padlabel.npy')
    print(len(branches_in_one_set))
    #for j, branch_label in enumerate(label_in_one_set):
        #print(j, ':', branch_label)
