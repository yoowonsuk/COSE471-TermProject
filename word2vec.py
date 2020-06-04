import torch
import random
import argparse
import time
from random import shuffle
from collections import Counter
from huffman import HuffmanCoding
import math

def getRandomContext(corpus, C=5):
    wordID = random.randint(0, len(corpus) - 1)

    context = corpus[max(0, wordID - C):wordID]
    if wordID + 1 < len(corpus):
        context += corpus[wordID + 1:min(len(corpus), wordID + C + 1)]

    centerword = corpus[wordID]
    context = [w for w in context if w != centerword]

    if len(context) > 0:
        return centerword, context
    else:
        return getRandomContext(corpus, C)


def CBOW(center, context, inputMatrix, outputMatrix):
    ################################  Input  ################################
    # center : Index of a centerword (type:int)                             #
    # context : Indices of contextwords (type:list(int))                    #
    # inputMatrix : Weight matrix of input (type:torch.tensor(V,D))         #
    # outputMatrix : Weight matrix of output (type:torch.tensor(V,D))       #
    #########################################################################

    ###############################  Output  ################################
    # loss : Loss value (type:torch.tensor(1))                              #
    # grad_emb : Gradient of word embedding (type:torch.tensor(1,D))        #
    # grad_out : Gradient of outputMatrix (type:torch.tensor(V,D))          #
    #########################################################################
    input_embed = inputMatrix[context]
    hidden = input_embed.sum(axis = 0).view(1, -1) / len(context)
    dot = torch.mm(hidden, outputMatrix.T).view(-1)
    softmax = torch.softmax(dot, dim=0)
    loss = -torch.log(softmax[center] + 1e-04)
    center_onehot = torch.zeros_like(dot)
    center_onehot[center] = 1
    dL = dot - center_onehot
    dL = dL.view(-1, 1)
    grad_out = torch.mm(dL, hidden)
    dx = torch.mm(dL.T, outputMatrix) / len(context)
    grad_emb = dx 
    return loss, grad_emb, grad_out

def CBOW_HS(center, context, codes, inputMatrix, outputMatrix):
    ################################  Input  ################################
    # center : Index of a centerword (type:int)                             #
    # context : Indices of contextwords (type:list(int))                    #
    # codes : List of Huffman code element (type:list)                      #
    # inputMatrix : Weight matrix of input (type:torch.tensor(V,D))         #
    # outputMatrix : Weight matrix of output (type:torch.tensor(V,D))       #
    #########################################################################

    ###############################  Output  ################################
    # loss : Loss value (type:torch.tensor(1))                              #
    # grad_emb : Gradient of word embedding (type:torch.tensor(1,D))        #
    # grad_out : Gradient of outputMatrix (type:torch.tensor(V,D))          #
    #########################################################################
    input_embed = inputMatrix[context]
    hidden = input_embed.sum(axis = 0).view(1, -1) / len(context)
    dot = torch.mm(hidden, outputMatrix.T).view(-1)
    prob = torch.zeros_like(dot)
    index_zeros = (codes == 0).nonzero()
    index_ones = (codes == 1).nonzero()
    prob[index_zeros] = torch.sigmoid(dot[index_zeros])
    prob[index_ones] = 1 - torch.sigmoid(dot[index_ones])
    loss = torch.sum(-torch.log(prob + 1e-04))
    dL = torch.zeros_like(dot)
    dL[index_zeros] = torch.sigmoid(dot[index_zeros]) - 1
    dL[index_ones] = torch.sigmoid(dot[index_ones])
    dL = dL.view(-1, 1)
    grad_out = torch.mm(dL, hidden)
    dx = torch.mm(dL.T, outputMatrix) / len(context)
    grad_emb = dx 
    return loss, grad_emb, grad_out


def CBOW_NS(center, context, inputMatrix, outputMatrix):
    ################################  Input  ################################
    # center : Index of a centerword (type:int)                             #
    # context : Indices of contextwords (type:list(int))                    #
    # inputMatrix : Weight matrix of input (type:torch.tensor(V,D))         #
    # outputMatrix : Weight matrix of output (type:torch.tensor(V,D))       #
    #########################################################################

    ###############################  Output  ################################
    # loss : Loss value (type:torch.tensor(1))                              #
    # grad_emb : Gradient of word embedding (type:torch.tensor(1,D))        #
    # grad_out : Gradient of outputMatrix (type:torch.tensor(V,D))          #
    #########################################################################

    loss = None
    grad_emb = None
    grad_out = None

    return loss, grad_emb, grad_out

def Skipgram(center, context, inputMatrix, outputMatrix):
    ################################  Input  ################################
    # center : Index of a centerword (type:int)                             #
    # context : Indices of contextwords (type:list(int))                    #
    # inputMatrix : Weight matrix of input (type:torch.tensor(V,D))         #
    # outputMatrix : Weight matrix of output (type:torch.tensor(V,D))       #
    #########################################################################

    ###############################  Output  ################################
    # loss : Loss value (type:torch.tensor(1))                              #
    # grad_emb : Gradient of word vector (type:torch.tensor(1,D))           #
    # grad_out : Gradient of outputMatrix (type:torch.tensor(V,D))          #
    #########################################################################
    hidden = inputMatrix[center].view(1, -1)
    dot = torch.mm(hidden, outputMatrix.T).view(-1)
    softmax = torch.softmax(dot, dim=0)
    loss = -torch.log(softmax[center] + 1e-04)
    center_onehot = torch.zeros_like(dot)
    center_onehot[center] = 1
    dL = dot - center_onehot
    dL = dL.view(-1, 1)
    grad_out = torch.mm(dL, hidden)
    dx = torch.mm(dL.T, outputMatrix)
    grad_emb = dx 
    return loss, grad_emb, grad_out


def Skipgram_HS(center, context, codes, inputMatrix, outputMatrix):
    ################################  Input  ################################
    # center : Index of a centerword (type:int)                             #
    # context : Indices of contextwords (type:list(int))                    #
    # codes : List of Huffman code element (type:list)                      #
    # inputMatrix : Weight matrix of input (type:torch.tensor(V,D))         #
    # outputMatrix : Weight matrix of output (type:torch.tensor(V,D))       #
    #########################################################################

    ###############################  Output  ################################
    # loss : Loss value (type:torch.tensor(1))                              #
    # grad_emb : Gradient of word vector (type:torch.tensor(1,D))           #
    # grad_out : Gradient of outputMatrix (type:torch.tensor(V,D))          #
    #########################################################################
    hidden = inputMatrix[center].view(1, -1)
    dot = torch.mm(hidden, outputMatrix.T).view(-1)
    prob = torch.zeros_like(dot)
    index_zeros = (codes == 0).nonzero()
    index_ones = (codes == 1).nonzero()
    prob[index_zeros] = torch.sigmoid(dot[index_zeros])
    prob[index_ones] = 1 - torch.sigmoid(dot[index_ones])
    loss = torch.sum(-torch.log(prob + 1e-04))
    dL = torch.zeros_like(dot)
    dL[index_zeros] = torch.sigmoid(dot[index_zeros]) - 1
    dL[index_ones] = torch.sigmoid(dot[index_ones])
    dL = dL.view(-1, 1)
    grad_out = torch.mm(dL, hidden)
    dx = torch.mm(dL.T, outputMatrix)
    grad_emb = dx 
    return loss, grad_emb, grad_out



def Skipgram_NS(center, context, inputMatrix, outputMatrix):
    ################################  Input  ################################
    # center : Index of a centerword (type:int)                             #
    # context : Index of a contextword (type:int)                           #
    # inputMatrix : Weight matrix of input (type:torch.tensor(V,D))         #
    # outputMatrix : Weight matrix of output (type:torch.tensor(V,D))       #
    #########################################################################

    ###############################  Output  ################################
    # loss : Loss value (type:torch.tensor(1))                              #
    # grad_emb : Gradient of word vector (type:torch.tensor(1,D))           #
    # grad_out : Gradient of outputMatrix (type:torch.tensor(V,D))          #
    #########################################################################

    loss = None
    grad_emb = None
    grad_out = None

    return loss, grad_emb, grad_out


def word2vec_trainer(ns, corpus, word2ind, freqdict, ind2node,
                     mode="CBOW", subsampling="N", dimension=64, learning_rate=0.05, iteration=50000):
    # initialization
    W_emb = torch.randn(len(word2ind), dimension) / (dimension ** 0.5)
    W_out = torch.randn(len(word2ind), dimension) / (dimension ** 0.5)
    W_emb = W_emb.cuda()
    W_out = W_out.cuda()
    window_size = 5

    losses = []
    for c in range(iteration):
        # Training word2vec using SGD
        while True:
            centerWord, contextWords = getRandomContext(corpus, window_size)
            if centerWord == " ":
                continue
            # subsampling
            if subsampling == "Y":
                threshold = 1e-5
                fw = freqdict[centerWord] / sum(freqdict.values())
                exclude_prob = 1 - math.sqrt(threshold / fw)
                if random.random() < exclude_prob:
                    continue
            if len(contextWords) == window_size * 2:
                break
        # to be implemented
        centerInd = torch.tensor(word2ind[centerWord])
        contextInds = torch.tensor([word2ind[context] for context in contextWords])
        lr = learning_rate * (1 - i / iteration)

        if mode == "CBOW":
            if ns == 0:
                # Only use the activated rows of the weight matrix
                nodes = torch.cuda.LongTensor(ind2node[centerInd.item()][0])
                codes = torch.cuda.LongTensor(ind2node[centerInd.item()][1])
                L, G_emb, G_out = CBOW_HS(centerInd, contextInds, codes, W_emb, W_out[nodes])
                W_emb[contextInds] -= lr * G_emb
                W_out[nodes] -= lr * G_out
                losses.append(L.item())
            elif ns > 0:
                L, G_emb, G_out = CBOW_NS(centerInd, contextInds, W_emb, W_out)
                W_emb[contextInds] -= lr * G_emb
                W_out -= lr * G_out
                losses.append(L.item())
            else:
                L, G_emb, G_out = CBOW(centerInd, contextInds, W_emb, W_out)
                W_emb[contextInds] -= lr * G_emb
                W_out -= lr * G_out
                losses.append(L.item())


        elif mode == "SG":
            if ns == 0:
                nodes = []
                codes = []
                for contextInd in list(contextInds):
                    nodes.append(torch.tensor(ind2node[contextInd.item()][0]))
                    codes.append(torch.tensor(ind2node[contextInd.item()][1]))
                # Only use the activated rows of the weight matrix
                for index, contextInd in enumerate(contextInds):
                    L, G_emb, G_out = Skipgram_HS(centerInd, contextInd, codes[index], W_emb, W_out[nodes[index]])
                    W_emb[centerInd] -= lr * G_emb.squeeze()
                    W_out[nodes[index]] -= lr * G_out
            elif ns > 0:
                for contextInd in contextInds:
                    L, G_emb, G_out = Skipgram_NS(centerInd, contextInd, W_emb, W_out)
                    W_emb[centerInd] -= lr * G_emb.squeeze()
                    W_out -= lr * G_out
            else:
                for contextInd in contextInds:
                    L, G_emb, G_out = Skipgram(centerInd, contextInd, W_emb, W_out)
                    W_emb[centerInd] -= lr * G_emb.squeeze()
                    W_out -= lr * G_out                

            losses.append(L.item())
        else:
            print("Unkwnown mode : " + mode)
            exit()
        if c % 1000 == 0:
            avg_loss = sum(losses) / len(losses)
            print("Loss : %f" % (avg_loss,))
            losses = []

    return W_emb, W_out

if __name__ == "__main__":
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
        text = open('text8', mode='r').readlines()[0][:10000000]  # Load a part of corpus for debugging
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
        else:
            processed.append(" ")

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
    # Training section
    start_time = time.time()
    emb, _ = word2vec_trainer(ns, processed, word2ind, freqdict, ind2node,
                              mode=mode, subsampling=subsampling, dimension=300, learning_rate=0.025, iteration=320000)
    end_time = time.time()
    print("Training time : %f min" % ((end_time - start_time) / 60))
    if args.ns == 0:
        save_ns = "hs"
    elif args.ns > 0:
        save_ns = "ns"
    else:
        save_ns = "neither"
    torch.save([emb, word2ind, ind2word], '{0}{1}{2}.pt'.format(mode, save_ns, subsampling))
