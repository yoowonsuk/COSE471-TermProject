import pickle
import torch
import argparse

def cos_similarity(x, y):
    eps = 1e-08
    norm_x = x / torch.sqrt(torch.sum(x**2) + eps)
    norm_y = y / torch.sqrt(torch.sum(y**2) + eps)
    return torch.dot(norm_x, norm_y)

def most_similar_byEmb(word, ind2word, word2ind, word_matrix, top=5):
    size = len(word2ind)
    similarity = torch.zeros(size)
    for i in range(size):
        similarity[i] = cos_similarity(word, word_matrix[i])
    for i in similarity.argsort(descending=True)[0:min(top, size)]:
        print("%s : %.4f" % (ind2word[i.item()], similarity[i]))

def infer_word(w1, w2, w3, ind2word, word2ind, word_matrix): # w1 - w2 + w3 = ??
    print("#### %s - %s + %s = ??? ####" % (w1, w2, w3))
    try:
        emb_test = word_matrix[word2ind[w1]] - word_matrix[word2ind[w2]] + word_matrix[word2ind[w3]]
        most_similar_byEmb(emb_test, ind2word, word2ind, word_matrix)
    except KeyError as e:
        print("Word is not embedded. Key :", e)

parser = argparse.ArgumentParser(description='Similarity test')
parser.add_argument('filename', metavar='filename', type=str)
args = parser.parse_args()
filename = args.filename
w_emb, word2ind, ind2word = torch.load(filename)
infer_word('works', 'work', 'speak', ind2word, word2ind, w_emb) # works - work + speak = ???
infer_word('work', 'works', 'speaks', ind2word, word2ind, w_emb)
infer_word('speaks', 'speak', 'work', ind2word, word2ind, w_emb)
infer_word('speak', 'speaks', 'works', ind2word, word2ind, w_emb)

infer_word('mice', 'mouse', 'dollar', ind2word, word2ind, w_emb)
infer_word('mouse', 'mice', 'dollars', ind2word, word2ind, w_emb)
infer_word('dollars', 'dollar', 'mouse', ind2word, word2ind, w_emb)
infer_word('dollar', 'dollars', 'mice', ind2word, word2ind, w_emb)

infer_word('walked', 'walking', 'swimming', ind2word, word2ind, w_emb)
infer_word('walking', 'walked', 'swam', ind2word, word2ind, w_emb)
infer_word('swam', 'swimming', 'walking', ind2word, word2ind, w_emb)
infer_word('swimming', 'swam', 'walked', ind2word, word2ind, w_emb)

infer_word('thinking', 'think', 'read', ind2word, word2ind, w_emb)
infer_word('think', 'thinking', 'reading', ind2word, word2ind, w_emb)
infer_word('reading', 'read', 'think', ind2word, word2ind, w_emb)
infer_word('read', 'reading', 'thinking', ind2word, word2ind, w_emb)

infer_word('easiest', 'easy', 'lucky', ind2word, word2ind, w_emb)
infer_word('easy', 'easiest', 'luckiest', ind2word, word2ind, w_emb)
infer_word('luckiest', 'lucky', 'easy', ind2word, word2ind, w_emb)
infer_word('lucky', 'luckiest', 'easiest', ind2word, word2ind, w_emb)

infer_word('greater', 'great', 'tough', ind2word, word2ind, w_emb)
infer_word('great', 'greater', 'tougher', ind2word, word2ind, w_emb)
infer_word('tougher', 'tough', 'great', ind2word, word2ind, w_emb)
infer_word('tough', 'tougher', 'greater', ind2word, word2ind, w_emb)

infer_word('impossibly', 'possibly', 'ethical', ind2word, word2ind, w_emb)
infer_word('possibly', 'impossibly', 'unethical', ind2word, word2ind, w_emb)
infer_word('unethical', 'ethical', 'possibly', ind2word, word2ind, w_emb)
infer_word('ethical', 'unethical', 'impossibly', ind2word, word2ind, w_emb)

infer_word('apparently', 'apparent', 'rapid', ind2word, word2ind, w_emb)
infer_word('apparent', 'apparently', 'rapidly', ind2word, word2ind, w_emb)
infer_word('rapidly', 'rapid', 'apparent', ind2word, word2ind, w_emb)
infer_word('rapid', 'rapidly', 'apparently', ind2word, word2ind, w_emb)
