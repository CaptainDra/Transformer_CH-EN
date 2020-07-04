# -*- coding: utf-8 -*-
#/usr/bin/python2


from __future__ import print_function
import codecs
import os

import tensorflow as tf
import numpy as np

from hyperparams import Hyperparams as hp
from data_load import load_test_data, load_de_vocab, load_en_vocab
from train import Graph
from nltk.translate.bleu_score import corpus_bleu

def eval(): 
    # 加载
    g = Graph(is_training=False)
    print("Graph loaded")
    
    # 加载数据
    X, Sources, Targets = load_test_data()
    de2idx, idx2de = load_de_vocab()
    en2idx, idx2en = load_en_vocab()
     
     
    # 开始阶段       
    with g.graph.as_default():    
        sv = tf.train.Supervisor()
        with sv.managed_session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            ## 恢复参数
            sv.saver.restore(sess, tf.train.latest_checkpoint(hp.logdir))
            print("Restored!")
              
            ## 模型名取得
            mname = open(hp.logdir + '/checkpoint', 'r').read().split('"')[1] # model name
             
            ## Inference
            if not os.path.exists('results'): os.mkdir('results')
            with codecs.open("results/" + mname, "w", "utf-8") as fout:
                list_of_refs, hypotheses = [], []
                for i in range(len(X) // hp.batch_size):
                     
                    ### 获取最小Batch
                    x = X[i*hp.batch_size: (i+1)*hp.batch_size]
                    sources = Sources[i*hp.batch_size: (i+1)*hp.batch_size]
                    targets = Targets[i*hp.batch_size: (i+1)*hp.batch_size]
                     
                    ### 自回归
                    preds = np.zeros((hp.batch_size, hp.maxlen), np.int32)
                    for j in range(hp.maxlen):
                        _preds = sess.run(g.preds, {g.x: x, g.y: preds})
                        preds[:, j] = _preds[:, j]
                     
                    ### 写
                    for source, target, pred in zip(sources, targets, preds): # sentence-wise
                        got = " ".join(idx2en[idx] for idx in pred).split("</S>")[0].strip()
                        fout.write("- source: " + source +"\n")
                        fout.write("- expected: " + target + "\n")
                        fout.write("- got: " + got + "\n\n")
                        print("- source: " + source +"\n")
                        print("- expected: " + target + "\n")
                        print("- got: " + got + "\n\n")
                        fout.flush()
                          
                        #得分
                        ref = target.split()
                        hypothesis = got.split()
                        if len(ref) > 3 and len(hypothesis) > 3:
                            list_of_refs.append([ref])
                            hypotheses.append(hypothesis)
              
                ## 计算得分（这里乘以一百方便便于观察，也有不乘以一百的）
                score = corpus_bleu(list_of_refs, hypotheses)
                fout.write("Bleu Score = " + str(100*score))
                print("Bleu Score = " + str(100*score))
                                          
if __name__ == '__main__':
    eval()
    print("Done")
    
    