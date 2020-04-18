import os
import json
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score

from datasets import _rocstories
from datasets import _ruoyao

def rocstories(data_dir, pred_path, log_path):
    preds = pd.read_csv(pred_path, delimiter='\t')['prediction'].values.tolist()
    _, _, _, labels = _rocstories(os.path.join(data_dir, 'cloze_test_test__spring2016_-_cloze_test_ALL_test.csv'))
    test_accuracy = accuracy_score(labels, preds)*100.
    logs = [json.loads(line) for line in open(log_path)][1:]
    best_validation_index = np.argmax([log['va_acc'] for log in logs])
    valid_accuracy = logs[best_validation_index]['va_acc']
    print('ROCStories Valid Accuracy: %.2f'%(valid_accuracy))
    print('ROCStories Test Accuracy:  %.2f'%(test_accuracy))


def ruoyao(data_dir, pred_path, log_path):
    preds = pd.read_csv(pred_path, delimiter='\t')['prediction'].values.tolist()
    *_, labels = _ruoyao(os.path.join(data_dir, "test"))
    #test_accuracy = accuracy_score(labels, preds)*100.
    #print('Test Accuracy:  %.2f'%(test_accuracy))
    logs = [json.loads(line) for line in open(log_path)][1:]
    best_validation_index = np.argmax([log['va_acc'] for log in logs])
    valid_accuracy = logs[best_validation_index]['va_acc']
    print('ruoyao Valid Accuracy: %.2f'%(valid_accuracy))
    #print('ruoyao Test Accuracy:  %.2f'%(test_accuracy))
