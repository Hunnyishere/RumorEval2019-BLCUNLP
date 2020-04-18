import os
import time
import math
import json
import joblib
import random
import argparse
import numpy as np
import tensorflow as tf

from tqdm import tqdm
from functools import partial
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report

from opt import adam, warmup_cosine, warmup_linear, warmup_constant
from datasets import rocstories, ruoyao
from analysis import rocstories as rocstories_analysis
from analysis import ruoyao as ruoyao_analysis
from text_utils import TextEncoder
from utils import encode_dataset, flatten, iter_data, find_trainable_variables, convert_gradient_to_tensor, shape_list, ResultLogger, assign_to_gpu, average_grads, make_path

def gelu(x):
    return 0.5*x*(1+tf.tanh(math.sqrt(2/math.pi)*(x+0.044715*tf.pow(x, 3))))

def swish(x):
    return x*tf.nn.sigmoid(x)

opt_fns = {
    'adam':adam,
}

act_fns = {
    'relu':tf.nn.relu,
    'swish':swish,
    'gelu':gelu
}

lr_schedules = {
    'warmup_cosine':warmup_cosine,
    'warmup_linear':warmup_linear,
    'warmup_constant':warmup_constant,
}

def _norm(x, g=None, b=None, e=1e-5, axis=[1]):
    u = tf.reduce_mean(x, axis=axis, keep_dims=True)
    s = tf.reduce_mean(tf.square(x-u), axis=axis, keep_dims=True)
    x = (x - u) * tf.rsqrt(s + e)
    if g is not None and b is not None:
        x = x*g + b
    return x

def norm(x, scope, axis=[-1]):
    with tf.variable_scope(scope):
        n_state = shape_list(x)[-1]
        g = tf.get_variable("g", [n_state], initializer=tf.constant_initializer(1))
        b = tf.get_variable("b", [n_state], initializer=tf.constant_initializer(0))
        return _norm(x, g, b, axis=axis)

def dropout(x, pdrop, train):
    if train and pdrop > 0:
        x = tf.nn.dropout(x, 1-pdrop)
    return x

def mask_attn_weights(w):
    n = shape_list(w)[-1]
    b = tf.matrix_band_part(tf.ones([n, n]), -1, 0)
    b = tf.reshape(b, [1, 1, n, n])
    w = w*b + -1e9*(1-b)
    return w

def _attn(q, k, v, train=False, scale=False):
    w = tf.matmul(q, k)
    #print (w)
    if scale:
        n_state = shape_list(v)[-1]
        w = w*tf.rsqrt(tf.cast(n_state, tf.float32))
    #print (w)
    w = mask_attn_weights(w)
    w = tf.nn.softmax(w)

    w = dropout(w, attn_pdrop, train)

    a = tf.matmul(w, v)
    return a

def split_states(x, n):
    x_shape = shape_list(x)
    m = x_shape[-1]
    new_x_shape = x_shape[:-1]+[n, m//n]
    return tf.reshape(x, new_x_shape)

def merge_states(x):
    x_shape = shape_list(x)
    new_x_shape = x_shape[:-2]+[np.prod(x_shape[-2:])]
    return tf.reshape(x, new_x_shape)

def split_heads(x, n, k=False):
    if k:
        return tf.transpose(split_states(x, n), [0, 2, 3, 1])
    else:
        return tf.transpose(split_states(x, n), [0, 2, 1, 3])

def merge_heads(x):
    return merge_states(tf.transpose(x, [0, 2, 1, 3]))

def conv1d(x, scope, nf, rf, w_init=tf.random_normal_initializer(stddev=0.02), b_init=tf.constant_initializer(0), pad='VALID', train=False):
    with tf.variable_scope(scope):
        nx = shape_list(x)[-1]
        w = tf.get_variable("w", [rf, nx, nf], initializer=w_init)
        b = tf.get_variable("b", [nf], initializer=b_init)
        if rf == 1: #faster 1x1 conv
            c = tf.reshape(tf.matmul(tf.reshape(x, [-1, nx]), tf.reshape(w, [-1, nf]))+b, shape_list(x)[:-1]+[nf])
        else: #was used to train LM
            c = tf.nn.conv1d(x, w, stride=1, padding=pad)+b
        return c

def attn(x, scope, n_state, n_head, train=False, scale=False):
    assert n_state%n_head==0
    with tf.variable_scope(scope):
        #print ('from now on:')
        #print (x)
        c = conv1d(x, 'c_attn', n_state*3, 1, train=train)
        #print (c)
        q, k, v = tf.split(c, 3, 2)
        q = split_heads(q, n_head)
        k = split_heads(k, n_head, k=True)
        v = split_heads(v, n_head)
        a = _attn(q, k, v, train=train, scale=scale)
        #print(a)
        a = merge_heads(a)
        #print (a)
        a = conv1d(a, 'c_proj', n_state, 1, train=train)
        #print (a)
        a = dropout(a, resid_pdrop, train)
        return a

def mlp(x, scope, n_state, train=False):
    with tf.variable_scope(scope):
        nx = shape_list(x)[-1]
        act = act_fns[afn]
        h = act(conv1d(x, 'c_fc', n_state, 1, train=train))
        h2 = conv1d(h, 'c_proj', nx, 1, train=train)
        h2 = dropout(h2, resid_pdrop, train)
        return h2

def block(x, scope, train=False, scale=False):
    with tf.variable_scope(scope):
        nx = shape_list(x)[-1]
        a = attn(x, 'attn', nx, n_head, train=train, scale=scale)
        n = norm(x+a, 'ln_1')
        m = mlp(n, 'mlp', nx*4, train=train)
        h = norm(n+m, 'ln_2')
        return h

def embed(X, we):
    we = convert_gradient_to_tensor(we)
    e = tf.gather(we, X)
    h = tf.reduce_sum(e, 2)
    return h

def clf(x, ny, w_init=tf.random_normal_initializer(stddev=0.02), b_init=tf.constant_initializer(0), train=False):
    with tf.variable_scope('clf'):
        nx = shape_list(x)[-1]
        w = tf.get_variable("w", [nx, ny], initializer=w_init)
        b = tf.get_variable("b", [ny], initializer=b_init)
        return tf.matmul(x, w)+b

def model(X, M, Y, train=False, reuse=False):
    num_classes = 4
    #num_classes = 3
    with tf.variable_scope('model', reuse=reuse):
        we = tf.get_variable("we", [n_vocab+n_special+n_ctx, n_embd], initializer=tf.random_normal_initializer(stddev=0.02))
        
        we = dropout(we, embd_pdrop, train)
        
        X = tf.reshape(X, [-1, n_ctx, 2])
        M = tf.reshape(M, [-1, n_ctx])

        h = embed(X, we)
        for layer in range(n_layer):
            h = block(h, 'h%d'%layer, train=train, scale=True)
           
        lm_h = tf.reshape(h[:, :-1], [-1, n_embd])
        
        
        lm_logits = tf.matmul(lm_h, we, transpose_b=True)
        
        lm_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=lm_logits, labels=tf.reshape(X[:, 1:, 0], [-1]))
        #print(lm_losses)
        lm_losses = tf.reshape(lm_losses, [shape_list(X)[0], shape_list(X)[1]-1])
        lm_losses = tf.reduce_sum(lm_losses*M[:, 1:], 1)/tf.reduce_sum(M[:, 1:], 1)
        clf_h = tf.reshape(h, [-1, n_embd])
        pool_idx = tf.cast(tf.argmax(tf.cast(tf.equal(X[:, :, 0], clf_token), tf.float32), 1), tf.int32)
        clf_h = tf.gather(clf_h, tf.range(shape_list(X)[0], dtype=tf.int32)*n_ctx+pool_idx)
        
        clf_h = tf.reshape(clf_h, [-1, 1, n_embd])
        #print(clf_h)
        if train and clf_pdrop > 0:
            shape = shape_list(clf_h)
            shape[1] = 1
            clf_h = tf.nn.dropout(clf_h, 1-clf_pdrop, shape)
        clf_h = tf.reshape(clf_h, [-1, n_embd])
        #print(Y)
        clf_logits = clf(clf_h, num_classes, train=train)
        #print(clf_logits)
        #clf_logits = clf_h
        #print(clf_logits)
        #print(clf_h)
        #print(Y)
        #clf_logits = tf.reshape(clf_logits, [-1, 1])
        clf_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=clf_logits, labels=Y)
        clf_logits = tf.nn.softmax(clf_logits)
        return clf_logits, clf_losses, lm_losses

def mgpu_train(*xs):
    gpu_ops = []
    gpu_grads = []
    xs = (tf.split(x, n_gpu, 0) for x in xs)
    for i, xs in enumerate(zip(*xs)):
        do_reuse = True if i > 0 else None
        with tf.device(assign_to_gpu(i, "/gpu:0")), tf.variable_scope(tf.get_variable_scope(), reuse=do_reuse):
            clf_logits, clf_losses, lm_losses = model(*xs, train=True, reuse=do_reuse)
            if lm_coef > 0:
                train_loss = tf.reduce_mean(clf_losses) + lm_coef*tf.reduce_mean(lm_losses)
            else:
                train_loss = tf.reduce_mean(clf_losses)
            params = find_trainable_variables("model")
            grads = tf.gradients(train_loss, params)
            grads = list(zip(grads, params))
            gpu_grads.append(grads)
            gpu_ops.append([clf_logits, clf_losses, lm_losses])
    #print(gpu_ops)
    #print(*gpu_ops)
    ops = [tf.concat(op, 0) for op in zip(*gpu_ops)]
    #print(ops)
    grads = average_grads(gpu_grads)
    #print(grads)
    grads = [g for g, p in grads]
    #print(grads)
    # 优化函数
    #train = opt_fns[opt](params, grads, lr, partial(lr_schedules[lr_schedule], warmup=lr_warmup), n_updates_total, l2=l2, max_grad_norm=max_grad_norm, vector_l2=vector_l2, b1=b1, b2=b2, e=e)
    # 新优化器
    train = tf.train.AdadeltaOptimizer(learning_rate=0.001).apply_gradients(zip(grads, params))
    #train = tf.train.AdagradOptimizer(learning_rate=0.001).apply_gradients(zip(grads, params))
    #print([train])
    return [train]+ops

def mgpu_predict(*xs):
    gpu_ops = []
    xs = (tf.split(x, n_gpu, 0) for x in xs)
    for i, xs in enumerate(zip(*xs)):
        with tf.device(assign_to_gpu(i, "/gpu:0")), tf.variable_scope(tf.get_variable_scope(), reuse=True):
            clf_logits, clf_losses, lm_losses = model(*xs, train=False, reuse=True)
            gpu_ops.append([clf_logits, clf_losses, lm_losses])
    ops = [tf.concat(op, 0) for op in zip(*gpu_ops)]
    return ops

def transform_roc(X1, X2, X3):
    n_batch = len(X1)
    xmb = np.zeros((n_batch, 2, n_ctx, 2), dtype=np.int32)
    mmb = np.zeros((n_batch, 2, n_ctx), dtype=np.float32)
    start = encoder['_start_']
    delimiter = encoder['_delimiter_']
    for i, (x1, x2, x3), in enumerate(zip(X1, X2, X3)):
        x12 = [start]+x1[:max_len]+[delimiter]+x2[:max_len]+[clf_token]
        x13 = [start]+x1[:max_len]+[delimiter]+x3[:max_len]+[clf_token]
        l12 = len(x12)
        l13 = len(x13)
        xmb[i, 0, :l12, 0] = x12
        xmb[i, 1, :l13, 0] = x13
        mmb[i, 0, :l12] = 1
        mmb[i, 1, :l13] = 1
    xmb[:, :, :, 1] = np.arange(n_vocab+n_special, n_vocab+n_special+n_ctx)
    return xmb, mmb

def transform_ruoyao(source, src_feature, other_tw, last_tw, last_feature, this_tw, this_feature):
    n_batch = len(source)
    xmb = np.zeros((n_batch, 1, n_ctx, 2), dtype=np.int32)
    mmb = np.zeros((n_batch, 1, n_ctx), dtype=np.float32)
    start = encoder['_start_']
    delimiter = encoder['_delimiter_']
    for i, (x1, x2, x3, x4, x5, x6, x7) in enumerate(zip(source,src_feature,other_tw,last_tw,last_feature,this_tw,this_feature)):
        #print(len(x1[:max_len]),len(x2[:max_len]),len(x3[:max_len]),len(x4[:max_len]))
        #print('max_len:',max_len)
        if len(x1)+len(x2)+len(x3)+len(x4)+len(x5)+len(x6)+len(x7)>n_ctx-n_special:
            x12 = [start] + x1[:max_len] + [delimiter] + x2 + [delimiter] + x3[:max_len] + [delimiter] + x4[:max_len] + [delimiter] + x5 + [delimiter]+ x6[:max_len] +[delimiter] + x7 + [clf_token]
        else:
            x12 = [start] + x1 + [delimiter] +x2+ [delimiter] + x3 + [delimiter] +x4+ [delimiter] + x5 + [
                      delimiter] +x6+ [delimiter] + x7 + [clf_token]
        l12 = len(x12)
        xmb[i, 0, :l12, 0] = x12
        mmb[i, 0, :l12] = 1
    xmb[:, :, :, 1] = np.arange(n_vocab + n_special, n_vocab + n_special + n_ctx)
    return xmb, mmb


'''
def transform_ruoyao(X1, X2):
    n_batch = len(X1)
    xmb = np.zeros((n_batch, 1, n_ctx, 2), dtype=np.int32)
    mmb = np.zeros((n_batch, 1, n_ctx), dtype=np.float32)
    start = encoder['_start_']
    delimiter = encoder['_delimiter_']
    for i,(x1,x2) in enumerate(zip(X1,X2)):
        x12 = [start]+x1[:max_len]+[delimiter]+x2[:max_len]+[clf_token] 
        l12 = len(x12)
        xmb[i, 0, :l12,0] = x12
        mmb[i, 0, :l12] = 1
    xmb[:, :, :, 1] = np.arange(n_vocab+n_special, n_vocab+n_special+n_ctx)
    return xmb,mmb
'''

def iter_apply(Xs, Ms, Ys):
    fns = [lambda x:np.concatenate(x, 0), lambda x:float(np.sum(x))]
    results = []
    for xmb, mmb, ymb in iter_data(Xs, Ms, Ys, n_batch=n_batch_train, truncate=False, verbose=True):
        n = len(xmb)
        if n == n_batch_train:
            res = sess.run([eval_mgpu_logits, eval_mgpu_clf_loss], {X_train:xmb, M_train:mmb, Y_train:ymb})
        else:
            res = sess.run([eval_logits, eval_clf_loss], {X:xmb, M:mmb, Y:ymb})
        res = [r*n for r in res]
        results.append(res)
    results = zip(*results)
    return [fn(res) for res, fn in zip(results, fns)]

def iter_predict(Xs, Ms):
    logits = []
    for xmb, mmb in iter_data(Xs, Ms, n_batch=n_batch_train, truncate=False, verbose=True):
        n = len(xmb)
        if n == n_batch_train:
            logits.append(sess.run(eval_mgpu_logits, {X_train:xmb, M_train:mmb}))
        else:
            logits.append(sess.run(eval_logits, {X:xmb, M:mmb}))
    logits = np.concatenate(logits, 0)
    return logits

def save(path):
    ps = sess.run(params)
    joblib.dump(ps, make_path(path))

def log():
    global best_score
    tr_logits, tr_cost = iter_apply(trX[:n_valid], trM[:n_valid], trY[:n_valid])
    va_logits, va_cost = iter_apply(vaX, vaM, vaY)
    tr_cost = tr_cost/len(trY[:n_valid])
    va_cost = va_cost/n_valid
    tr_acc = accuracy_score(trY[:n_valid], np.argmax(tr_logits, 1))*100.
    va_acc = accuracy_score(vaY, np.argmax(va_logits, 1))*100.
    tr_f1 = f1_score(trY[:n_valid], np.argmax(tr_logits, 1), average='macro')
    va_f1 = f1_score(vaY, np.argmax(va_logits, 1), average='macro')
    target_names = ['support', 'comment', 'deny', 'query']
    #target_names = ['true', 'false', 'unverified']
    #tr_report = classification_report(trY[:n_valid], np.argmax(tr_logits, 1), target_names=target_names)
    va_report = classification_report(vaY, np.argmax(va_logits, 1), target_names=target_names)
    tr_pr = precision_score(trY[:n_valid], np.argmax(tr_logits, 1), average='macro')
    va_pr = precision_score(vaY, np.argmax(va_logits, 1), average='macro')
    tr_re = recall_score(trY[:n_valid], np.argmax(tr_logits, 1), average='macro')
    va_re = recall_score(vaY, np.argmax(va_logits, 1), average='macro')
    logger.log(n_epochs=n_epochs, n_updates=n_updates, tr_cost=tr_cost, va_cost=va_cost, tr_acc=tr_acc, va_acc=va_acc)
    print('n_epochs: %d, n_updates: %d, tr_cost: %.3f, va_cost: %.3f'%(n_epochs, n_updates, tr_cost, va_cost))
    print('train_acc: %.2f, dev_acc: %.2f, train_macroF: %.4f, dev_macroF: %.4f, train_precision: %.4f, dev_precision: %.4f, train_recall: %.4f, dev_recall: %.4f'%(tr_acc, va_acc, tr_f1, va_f1, tr_pr, va_pr, tr_re, va_re))
    #print('train f1 report:\n',tr_report)
    print('dev f1 report:\n',va_report,'\n')
    if submit:
        score = va_f1
        if score > best_score:
            best_score = score
            save(os.path.join(save_dir, desc, 'best_params.jl'))

argmax = lambda x:np.argmax(x, 1)

pred_fns = {
    'rocstories':argmax,
    'ruoyao':argmax,
}

filenames = {
    'rocstories':'ROCStories.tsv',
    'ruoyao':'ruoyao.tsv'
}

label_decoders = {
    'rocstories':None,
    'ruoyao':None,
}

def predict():
    filename = filenames[dataset]
    pred_fn = pred_fns[dataset]
    label_decoder = label_decoders[dataset]
    preds = iter_predict(teX, teM)
    predictions = pred_fn(preds)
    if label_decoder is not None:
        predictions = [label_decoder[prediction] for prediction in predictions]
    path = os.path.join(submission_dir, filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        #f.write('\{\n\"{}\"\:\{\n'.format('subtaskaenglish'))
        f.write('{}\t{}\t{}\n'.format('id', 'prediction', 'probability'))
        for i, prediction in enumerate(predictions):
            #f.write('\"{}\":\"{}\",\n'.format(nums[i], label[prediction]))
            f.write('{}\t{}\t{}\n'.format(nums[i], prediction, preds[i]))
        #f.write('\}\n\}')

def count2():
    print ('!!!the size of model is :')
    print( np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--desc', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--log_dir', type=str, default='log/')
    parser.add_argument('--save_dir', type=str, default='save/')
    parser.add_argument('--data_dir', type=str, default='data/')
    parser.add_argument('--submission_dir', type=str, default='submission/')
    parser.add_argument('--submit', action='store_true')
    parser.add_argument('--analysis', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_iter', type=int, default=3)
    parser.add_argument('--n_batch', type=int, default=8)
    parser.add_argument('--max_grad_norm', type=int, default=1)
    parser.add_argument('--lr', type=float, default=6.25e-5)
    parser.add_argument('--lr_warmup', type=float, default=0.002)
    parser.add_argument('--n_ctx', type=int, default=512)
    parser.add_argument('--n_embd', type=int, default=768)
    parser.add_argument('--n_head', type=int, default=12)
    parser.add_argument('--n_layer', type=int, default=12)
    parser.add_argument('--embd_pdrop', type=float, default=0.1)
    parser.add_argument('--attn_pdrop', type=float, default=0.1)
    parser.add_argument('--resid_pdrop', type=float, default=0.1)
    parser.add_argument('--clf_pdrop', type=float, default=0.1)
    parser.add_argument('--l2', type=float, default=0.01)
    parser.add_argument('--vector_l2', action='store_true')
    parser.add_argument('--n_gpu', type=int, default=4)
    parser.add_argument('--opt', type=str, default='adam')
    parser.add_argument('--afn', type=str, default='gelu')
    parser.add_argument('--lr_schedule', type=str, default='warmup_linear')
    parser.add_argument('--encoder_path', type=str, default='model/encoder_bpe_40000.json')
    parser.add_argument('--bpe_path', type=str, default='model/vocab_40000.bpe')
    parser.add_argument('--n_transfer', type=int, default=12)
    parser.add_argument('--lm_coef', type=float, default=0.5)
    parser.add_argument('--b1', type=float, default=0.9)
    parser.add_argument('--b2', type=float, default=0.999)
    parser.add_argument('--e', type=float, default=1e-8)

    args = parser.parse_args()

    print(args)
    globals().update(args.__dict__)
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)

    logger = ResultLogger(path=os.path.join(log_dir, '{}.jsonl'.format(desc)), **args.__dict__)
    text_encoder = TextEncoder(encoder_path, bpe_path)
    encoder = text_encoder.encoder
    n_vocab = len(text_encoder.encoder)

    #(trX1, trX2, trX3, trY), (vaX1, vaX2, vaX3, vaY), (teX1, teX2, teX3) = encode_dataset(rocstories(data_dir), encoder=text_encoder)
    #enco_ry = ruoyao(data_dir)
    #(trX1,trX2,tyY), (vaX1, vaX2, vaY), (teX1, teX2) = ruoyao(data_dir)
    #print(trX1[0])
    #(trX1, trX2, trY), (vaX1, vaX2, vaY), (teX1, teX2, teY) = encode_dataset(ruoyao(data_dir),encoder=text_encoder)
    # 下面这步已经经过了bpe
    (trSrc,trSrcFea,trOther,trLast,trLastFea,trThis,trThisFea,trY), (vaSrc,vaSrcFea,vaOther,vaLast,vaLastFea,vaThis,vaThisFea,vaY), (teSrc,teSrcFea,teOther,teLast,teLastFea,teThis,teThisFea, teY) = encode_dataset(ruoyao(data_dir), encoder=text_encoder)
    n = os.path.join(data_dir, 'test','A_id_set.npy')
    nums =np.load(n)
    label = ['support','comment','deny','query']
    #label = ['true', 'false', 'unverified']

    n_y = 2
    encoder['_start_'] = len(encoder)
    encoder['_delimiter_'] = len(encoder)
    encoder['_classify_'] = len(encoder)
    clf_token = encoder['_classify_']
    #n_special = 3
    n_special = 8
    #max_len = n_ctx//2-2  # n_ctx除以2的最小整数-2	即默认512/2-2
    max_len = n_ctx // 5 - 2  # 为防止重新用spacy分词后词数超过512，让每个的长度小一点
    #n_ctx = min(max([len(x1[:max_len])+len(x2[:max_len]) for x1, x2 in zip(trX1, trX2)]+[len(x1[:max_len])+len(x2[:max_len]) for x1, x2 in zip(vaX1, vaX2)]+[len(x1[:max_len])+len(x2[:max_len]) for x1, x2 in zip(teX1, teX2)])+3, n_ctx)
    #trX, trM = transform_ruoyao(trX1, trX2)
    #vaX, vaM = transform_ruoyao(vaX1, vaX2)
    # 句长：取小的那一个
    n_ctx = min(max([len(x1[:max_len])+len(x2)+len(x3[:max_len])+len(x4[:max_len])+len(x5)+len(x6[:max_len])+len(x7) for x1,x2,x3,x4,x5,x6,x7 in zip(trSrc,trSrcFea,trOther,trLast,trLastFea,trThis,trThisFea)]
        + [len(x1[:max_len])+len(x2)+len(x3[:max_len])+len(x4[:max_len])+len(x5)+len(x6[:max_len])+len(x7) for x1,x2,x3,x4,x5,x6,x7 in zip(vaSrc,vaSrcFea,vaOther,vaLast,vaLastFea,vaThis,vaThisFea)]
        + [len(x1[:max_len])+len(x2)+len(x3[:max_len])+len(x4[:max_len])+len(x5)+len(x6[:max_len])+len(x7) for x1,x2,x3,x4,x5,x6,x7 in zip(teSrc,teSrcFea,teOther,teLast,teLastFea,teThis,teThisFea)])
                + n_special, n_ctx)
    trX, trM = transform_ruoyao(trSrc,trSrcFea,trOther,trLast,trLastFea,trThis,trThisFea)
    vaX, vaM = transform_ruoyao(vaSrc,vaSrcFea,vaOther,vaLast,vaLastFea,vaThis,vaThisFea)
    if submit:
        #teX, teM = transform_ruoyao(teX1, teX2)
        teX, teM = transform_ruoyao(teSrc,teSrcFea,teOther,teLast,teLastFea,teThis,teThisFea)

    n_train = len(trY)
    n_valid = len(vaY)
    n_batch_train = n_batch*n_gpu
    n_updates_total = (n_train//n_batch_train)*n_iter

    X_train = tf.placeholder(tf.int32, [n_batch_train, 1, n_ctx, 2])
    M_train = tf.placeholder(tf.float32, [n_batch_train, 1, n_ctx])
    X = tf.placeholder(tf.int32, [None, 1, n_ctx, 2])
    M = tf.placeholder(tf.float32, [None, 1, n_ctx])

    Y_train = tf.placeholder(tf.int32, [n_batch_train])
    Y = tf.placeholder(tf.int32, [None])

    train, logits, clf_losses, lm_losses = mgpu_train(X_train, M_train, Y_train)
    clf_loss = tf.reduce_mean(clf_losses)
    params = find_trainable_variables('model')
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    sess.run(tf.global_variables_initializer())

    shapes = json.load(open('model/params_shapes.json'))
    offsets = np.cumsum([np.prod(shape) for shape in shapes])
    
    init_params = [np.load('model/params_{}.npy'.format(n)) for n in range(10)]
    init_params = np.split(np.concatenate(init_params, 0), offsets)[:-1]
    init_params = [param.reshape(shape) for param, shape in zip(init_params, shapes)]
    init_params[0] = init_params[0][:n_ctx]
    init_params[0] = np.concatenate([init_params[1], (np.random.randn(n_special, n_embd)*0.02).astype(np.float32), init_params[0]], 0)
    del init_params[1]

    if n_transfer == -1:
        n_transfer = 0
    else:
        n_transfer = 1+n_transfer*12
    sess.run([p.assign(ip) for p, ip in zip(params[:n_transfer], init_params[:n_transfer])])

    eval_mgpu_logits, eval_mgpu_clf_losses, eval_mgpu_lm_losses = mgpu_predict(X_train, M_train, Y_train)
    eval_logits, eval_clf_losses, eval_lm_losses = model(X, M, Y, train=False, reuse=True)
    eval_clf_loss = tf.reduce_mean(eval_clf_losses)
    eval_mgpu_clf_loss = tf.reduce_mean(eval_mgpu_clf_losses)

    #count2()
    # 把这里以下注释掉是用存好的GPT模型预测
    n_updates = 0
    n_epochs = 0
    if dataset != 'stsb':
        trYt = trY
    if submit:
        save(os.path.join(save_dir, desc, 'best_params.jl'))
    best_score = 0
    for i in range(n_iter):
        for xmb, mmb, ymb in iter_data(*shuffle(trX, trM, trYt, random_state=np.random), n_batch=n_batch_train, truncate=True, verbose=True):
            cost, _ = sess.run([clf_loss, train], {X_train:xmb, M_train:mmb, Y_train:ymb})
            
            n_updates += 1
            if n_updates in [1000, 2000, 4000, 8000, 16000, 32000] and n_epochs == 0:
                log()
        n_epochs += 1
        log()
    # 把这里以上注释掉是用存好的GPT模型预测

    if submit:  # 用微调后的新模型
        #sess.run([p.assign(ip) for p, ip in zip(params, joblib.load(os.path.join(save_dir, desc, 'best_params.jl')))])
        sess.run([p.assign(ip) for p, ip in zip(params, joblib.load(os.path.join(save_dir, desc, 'taskA_enlarge_word_feature_5644.jl')))])
        predict()
        if analysis:  # 存的是test的预测值
            ruoyao_analysis(data_dir, os.path.join(submission_dir, 'ruoyao.tsv'), os.path.join(log_dir, 'ruoyao.jsonl'))
