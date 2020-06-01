import torch
import random
import sys
import argparse
from collections import Counter
sys.path.append('..')
from common.optimizer import SGD
from common.util import preprocess, create_context_target, convert_one_hot
from w2vec.CBow import CustomCBOW
from w2vec.SkipGram import CustomSkipGram
import pickle # for save
from common.util import most_similar_byEmb
from huffman import HuffmanCoding

def word2vec_trainer(ns, corpus, word2ind, ind2node,
                     mode="CBOW", dimension=64, learning_rate=0.05, iteration=50000, batch_size=100, window_size=3):
    vocab_size = len(word2ind)
    losses = []
    parallel_num = 2 * window_size # number of parallel affine layers
    #################### model initialization ####################
    if ns == 0: # Hierarchical softmax
        if mode == "CBOW":
            model = CustomCBOW(vocab_size, dimension, vocab_size, parallel_num, hs=True)
        elif mode == "SG":
            model = CustomSkipGram(vocab_size, dimension, vocab_size, parallel_num, hs=True)
        else:
            print("Unkwnown mode : " + mode)
            exit()
    else: # Negative sampling
        if mode == "CBOW":
            model = CustomCBOW(vocab_size, dimension, vocab_size, parallel_num)
        elif mode == "SG":
            model = CustomSkipGram(vocab_size, dimension, vocab_size, parallel_num)
        else:
            print("Unkwnown mode : " + mode)
            exit()        
    optimizer = SGD(lr=learning_rate)
    ################################ Learning ################################
    print("Creating contexts and targets...")
    contexts, target = create_context_target(corpus, window_size)
    target = convert_one_hot(target, vocab_size).float()
    contexts = convert_one_hot(contexts, vocab_size).float()
    batch_size = min(batch_size, len(target))
    print("Start training...")
    for i in range(iteration+1):
        ################## getRandomContext ##################
        index = torch.randperm(len(target))[0:batch_size]
        centerWord, contextWords = target[index], contexts[index]
        ########## For Hierarchical softmax ##########
        model.set_HsSetting(ind2node)
        ################## learning ##################    
        loss = model.forward(contextWords, centerWord)
        model.backward()
        optimizer.update(model)
        losses.append(loss)
        ################## learning rate decay ##################
        lr = learning_rate*(1-i/iteration)
        optimizer.set_lr(lr)
        #########################################################
        if i%100==0 and i != 0:
            avg_loss=sum(losses)/len(losses)
            print("Iteration : %d / Loss : %f" %(i, avg_loss))
    ##########################################################################
 
    ######### Extract W matrix #########
    W_in, b_in = model.get_inputw()
    W_out, b_out = model.get_outputw()
    W_emb = W_in
    ####################################
    return W_emb, W_out

def main():
    parser = argparse.ArgumentParser(description='Word2vec')
    parser.add_argument('mode', metavar='mode', type=str,
                        help='"SG" for skipgram, "CBOW" for CBOW')
    parser.add_argument('ns', metavar='negative_samples', type=int,
                        help='0 for hierarchical softmax, the other numbers would be the number of negative samples')
    parser.add_argument('subsampling', metavar='subsampling', type=str,
                        help='"Y" for using subsampling, "N" for not')
    parser.add_argument('part', metavar='partition', type=str,
                        help='"part" if you want to train on a part of corpus, "full" if you want to train on full corpus')
    args = parser.parse_args()
    ns = args.ns
    mode = args.mode
    subsampling = args.subsampling
    part = args.part
    # Load and tokenize corpus
    print("loading...")
    if part == "part":
        # text = open('text8', mode='r').readlines()[0][:10000000]  # Load a part of corpus for debugging
        text = open('text8', mode='r').readlines()[0][:10000]
    elif part == "full":
        text = open('text8', mode='r').readlines()[0]  # Load full corpus for submission
    else:
        print("Unknown argument : " + part)
        exit()

    print("tokenizing...")
    corpus = text.split()
    frequency = Counter(corpus)
    processed = []
    # Discard rare words
    for word in corpus:
        if frequency[word] > 1:
            processed.append(word)
    vocabulary = set(processed)

    # Assign an index number to a word
    word2ind = {}
    i = 0
    for word in vocabulary:
        word2ind[word] = i
        i += 1
    ind2word = {}
    for k, v in word2ind.items():
        ind2word[v] = k

    print("Vocabulary size : %d" % len(word2ind))
    print("Corpus size : %d" % len(processed))

    # Code dict for hierarchical softmax
    freqdict = {}
    for word in vocabulary:
        freqdict[word] = frequency[word]
    codedict = HuffmanCoding().build(freqdict)
    nodedict = {}
    ind2node = {}
    i = 0
    if ns == 0:
        for word in codedict[0].keys():
            code = codedict[0][word]
            s = ""
            nodeset = []
            codeset = []
            for ch in code:
                if s in nodedict.keys():
                    nodeset.append(nodedict[s])
                else:
                    nodedict[s] = i
                    nodeset.append(i)
                    i += 1
                codeset.append(int(ch))
                s += ch
            ind2node[word2ind[word]] = (nodeset, codeset)
    id_corpus = []
    for word in processed:
        id_corpus.append(word2ind[word])
    w_emb, w_out = word2vec_trainer(ns, torch.tensor(id_corpus).int(), word2ind, ind2node,
                              mode=mode, dimension=300, learning_rate=0.025, iteration=320000)
    torch.save([w_emb, word2ind, ind2word, w_out], '{0}{1}{2}.pt'.format(mode, ns, subsampling))
    
main()
