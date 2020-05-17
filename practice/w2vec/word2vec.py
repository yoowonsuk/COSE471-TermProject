import torch
import random
from collections import Counter
import argparse
import torch.nn as nn # use plan embeding 
import sys
sys.path.append('..')
from common.optimizer import SGD
from common.util import preprocess, create_context_target, convert_one_hot
from w2vec.CBow import CustomCBOW
from w2vec.SkipGram import CustomSkipGram

def word2vec_trainer(corpus, word2ind, mode="CBOW", dimension=64, learning_rate=0.05, iteration=50000, batch_size=500):
    window_size = 1
    vocab_size = len(word2ind)
    contexts, target = create_context_target(corpus, window_size)
    target = convert_one_hot(target, vocab_size).float()
    contexts = convert_one_hot(contexts, vocab_size).float()
    batch_size = min(batch_size, len(target))
    optimizer = SGD(lr=learning_rate)
    losses=[]
    parallel_num = contexts.shape[1]
    print("Number of words : %d" % (len(target)))
    #################### model initialization ####################
    if mode == "CBOW":
        model = CustomCBOW(vocab_size, dimension, vocab_size, parallel_num)
    elif mode == "SG":
        model = CustomSkipGram(vocab_size, dimension, vocab_size, parallel_num)
    else:
        print("Unkwnown mode : "+mode)
        exit()
    ##############################################################

    for i in range(iteration):
        
        ################## getRandomContext ##################
        index = torch.randperm(len(target))[0:batch_size]
        centerWord, contextWords = target[index], contexts[index]
        ######################################################    
        if mode=="CBOW":
            x = contextWords
            t = centerWord

        elif mode=="SG":
            x = centerWord
            t = contextWords

        loss = model.forward(x, t)
        model.backward()
        optimizer.update(model)
        W_emb = model.get_inputw()
        W_out = model.get_outputw()
        losses.append(loss)
        ################## learning rate decay ##################
        lr = learning_rate*(1-i/iteration)
        optimizer.set_lr(lr)
        #########################################################

        if i%1000==0:
        	avg_loss=sum(losses)/len(losses)
        	print("Loss : %f" %(avg_loss,))
        	losses=[]

    return W_emb, W_out

def main():
    # with open('text8', 'r') as f:
    #     text = f.read()
    text = "i say hello and you say goodbye"
	# Write your code of data processing, training, and evaluation
	# Full training takes very long time. We recommend using a subset of text8 when you debug
    corpus, word2ind, _ = preprocess(text, subset=1.0)
    print("processing completed")
    word2vec_trainer(corpus, word2ind, mode="CBOW", learning_rate=0.01, iteration=50000)
    

main()
