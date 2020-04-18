# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 19:25:33 2019

@author: dell
"""

import numpy as np
import os
import pandas as pd
import csv

def convert_A_label(label):  # task A labels
    if label == 0:
        return("support")
    elif label == 1:
        return("comment")
    elif label == 2:
        return("deny")
    elif label == 3:
        return("query")

def convert_B_label(label):  # task B labels
    if label == 0:
        return("true")
    elif label == 1:
        return("false")
    elif label == 2:
        return("unverified")


def load_whole_dataset():
    whichset = ['train', 'dev', 'test']  # 19rumor

    for sset in whichset:
        if sset=='test':
            taskA_ids = []
            taskB_ids = []
            unique_texts = []
        branches = []
        is_src = []
        ids = []
        texts = []
        position_in_branch = []
        label_As = []
        label_Bs = []
        branches_text_in_one_set = np.load('data/taskA_data_0103/' + sset + '/branch_id_A_B_label.npy',encoding='bytes')
        for j, branch in enumerate(branches_text_in_one_set):
            #print('branch:',j)
            for i, tweet in enumerate(branch):
                branches.append(j)
                if i==0:
                    is_src.append('yes')
                else:
                    is_src.append('no')
                id_str = tweet[0]
                ids.append(id_str)
                text = tweet[1]
                texts.append(text)
                position_in_branch.append(i)
                if sset!='test':
                    label_As.append(convert_A_label(int(tweet[2])))
                    if i==0:
                        label_Bs.append(convert_B_label(int(tweet[3])))
                    else:
                        label_Bs.append(None)
                else:
                    label_As.append(None)
                    label_Bs.append(None)
                if sset=='test':
                    if id_str not in taskA_ids:
                        taskA_ids.append(id_str)
                        unique_texts.append(text)
                    if i==0 and id_str not in taskB_ids:
                        taskB_ids.append(id_str)
        
        # 写入csv
        #字典中的key值即为csv中列名
        dataframe = pd.DataFrame({'branch':branches,'is_src':is_src,'position in branch':position_in_branch,'id':ids,'text':texts,'label_A':label_As,'label_B':label_Bs})

        #将DataFrame存储为csv,index表示是否显示行名，default=True
        dataframe.to_csv(sset+"_branch_text.csv",index=True,sep=',')

        '''
        if sset=='test':
            print(len(taskA_ids))  # 1827
            print(len(taskB_ids))  # 81
            for i,idd in enumerate(taskA_ids):  # 打印没有text的tweet id
                if unique_texts[i]=='':
                    print(idd)
        '''


if __name__ == "__main__":
    load_whole_dataset()