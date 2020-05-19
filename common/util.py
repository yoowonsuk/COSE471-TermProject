import re
import torch
import matplotlib.pyplot as plt
def corpus2dict(corpus_raw):
    word_to_id = {}
    id_to_word = {}
    for word in corpus_raw:
        if word not in word_to_id:
            new_id = len(word_to_id)
            word_to_id[word] = new_id
            id_to_word[new_id] = word
    return word_to_id, id_to_word

def corpus2Id(corpus_raw, w2i):
    return torch.tensor([w2i[w] for w in corpus_raw], dtype=torch.int32)

def preprocess(corpus_raw, subset=1.0):
    corpus_raw = re.sub("[^\w]", " ",  corpus_raw.lower()).split()
    corpus_raw = corpus_raw[0:int(len(corpus_raw)*subset)]
    word_to_id, id_to_word = corpus2dict(corpus_raw)
    corpus_id = corpus2Id(corpus_raw, word_to_id)
    return corpus_id, word_to_id, id_to_word

def create_co_matrix(corpus_id, vocab_size, window_size=1):
    corpus_id_len = len(corpus_id)
    matrix = torch.zeros((vocab_size, vocab_size))
    words = corpus_id.unique()
    for word in words:
        index = (corpus_id == word).nonzero()
        for i in index:
            for j in range(i-window_size, i+window_size+1):
                if j < 0 or j == i or j >= corpus_id_len:
                    continue

                matrix[word,corpus_id[j]] += 1
    return matrix

def cos_similarity(x, y):
    eps = 1e-08
    norm_x = x / torch.sqrt(torch.sum(x**2) + eps)
    norm_y = y / torch.sqrt(torch.sum(y**2) + eps)
    return torch.dot(norm_x, norm_y)

def infer_word(w1, w2, w3, word_to_id, id_to_word, word_matrix): # w1 - w2 + w3 = ??
    print("#### %s - %s + %s = ??? ####" % (w1, w2, w3))
    emb_test = word_matrix[word_to_id[w1]] - word_matrix[word_to_id[w2]] + word_matrix[word_to_id[w3]]
    most_similar_byEmb(emb_test, word_to_id, id_to_word, word_matrix)

def most_similar_byEmb(word, word_to_id, id_to_word, word_matrix, top=5):
    size = len(word_to_id)
    similarity = torch.zeros(size)
    for i in range(size):
        similarity[i] = cos_similarity(word, word_matrix[i])
    for i in similarity.argsort(descending=True)[0:min(top, size)]:
        print("%s : %.4f" % (id_to_word[i.item()], similarity[i]))

def most_similar_byWord(word, word_to_id, id_to_word, word_matrix, top=5):
    assert(word in word_to_id)
    size = len(word_to_id)
    similarity = torch.zeros(size)
    for i in range(size):
        if i == word_to_id[word]:
            continue
        similarity[i] = cos_similarity(word_matrix[word_to_id[word]], word_matrix[i])
    for i in similarity.argsort(descending=True)[0:min(top, size)]:
        print("%s : %.4f" % (id_to_word[i.item()], similarity[i]))
    
def convert_one_hot(corpus, vocab_size):
    N = corpus.shape[0]
    if corpus.ndim == 1:
        one_hot = torch.zeros((N, vocab_size), dtype=torch.int32)
        for idx, word_id in enumerate(corpus):
            one_hot[idx, word_id] = 1
    elif corpus.ndim == 2:
        C = corpus.shape[1]
        one_hot = torch.zeros((N, C, vocab_size), dtype=torch.int32)
        for idx_0, word_ids in enumerate(corpus):
            for idx_1, word_id in enumerate(word_ids):
                one_hot[idx_0, idx_1, word_id] = 1

    return one_hot

def create_context_target(corpus, window_size=1):
    target = corpus[window_size:-window_size]
    context = torch.zeros(2 * window_size).int()
    for i in range(window_size, len(corpus)-window_size):
        sub = corpus[i-window_size:i+window_size+1].clone()
        sub = torch.cat([sub[0:window_size], sub[window_size+1:sub.shape[0]]])
        context = torch.cat((context, sub))
    context = context.reshape((-1, window_size*2))
    return context[1:], target
