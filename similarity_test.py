import pickle
import torch
import argparse

def cos_similarity(x, y):
    eps = 1e-08
    norm_x = x / torch.sqrt(torch.sum(x**2) + eps)
    norm_y = y / torch.sqrt(torch.sum(y**2) + eps)
    return torch.dot(norm_x, norm_y)

def most_similar_byEmb(word, answer, ind2word, word2ind, word_matrix, top=5):
    size = len(word2ind)
    similarity = torch.zeros(size)
    correct = False
    for i in range(size):
        similarity[i] = cos_similarity(word, word_matrix[i])
    for i in similarity.argsort(descending=True)[0:min(top, size)]:
        if ind2word[i.item()] == answer:
            correct = True
        print("%s : %.4f" % (ind2word[i.item()], similarity[i]))
    return correct

def infer_word(w1, w2, w3, answer, ind2word, word2ind, word_matrix): # w1 - w2 + w3 = ??
    print("#### %s - %s + %s = ??? ####" % (w1, w2, w3))
    correct = False
    try:
        emb_test = word_matrix[word2ind[w1]] - word_matrix[word2ind[w2]] + word_matrix[word2ind[w3]]
        correct = most_similar_byEmb(emb_test, answer, ind2word, word2ind, word_matrix)
    except KeyError as e:
        print("Word is not embedded. Key :", e)
    finally:
        if correct is True:
            return 1
        else:
            return 0

parser = argparse.ArgumentParser(description='Similarity test')
parser.add_argument('filename', metavar='filename', type=str)
args = parser.parse_args()
filename = args.filename
w_emb, word2ind, ind2word = torch.load(filename)

correct_cnt = 0
correct_cnt += infer_word('works', 'work', 'speak', 'speaks', ind2word, word2ind, w_emb) # works - work + speak = ???
correct_cnt += infer_word('work', 'works', 'speaks', 'speak', ind2word, word2ind, w_emb)
correct_cnt += infer_word('speaks', 'speak', 'work', 'works', ind2word, word2ind, w_emb)
correct_cnt += infer_word('speak', 'speaks', 'works', 'work', ind2word, word2ind, w_emb)

correct_cnt += infer_word('mice', 'mouse', 'dollar', 'dollars', ind2word, word2ind, w_emb)
correct_cnt += infer_word('mouse', 'mice', 'dollars', 'dollar', ind2word, word2ind, w_emb)
correct_cnt += infer_word('dollars', 'dollar', 'mouse', 'mice', ind2word, word2ind, w_emb)
correct_cnt += infer_word('dollar', 'dollars', 'mice', 'mouse', ind2word, word2ind, w_emb)

correct_cnt += infer_word('walked', 'walking', 'swimming', 'swam', ind2word, word2ind, w_emb)
correct_cnt += infer_word('walking', 'walked', 'swam', 'swimming', ind2word, word2ind, w_emb)
correct_cnt += infer_word('swam', 'swimming', 'walking', 'walk', ind2word, word2ind, w_emb)
correct_cnt += infer_word('swimming', 'swam', 'walked', 'walking', ind2word, word2ind, w_emb)

correct_cnt += infer_word('thinking', 'think', 'read', 'reading', ind2word, word2ind, w_emb)
correct_cnt += infer_word('think', 'thinking', 'reading', 'read', ind2word, word2ind, w_emb)
correct_cnt += infer_word('reading', 'read', 'think', 'thinking', ind2word, word2ind, w_emb)
correct_cnt += infer_word('read', 'reading', 'thinking', 'think', ind2word, word2ind, w_emb)

correct_cnt += infer_word('easiest', 'easy', 'lucky', 'luckiest', ind2word, word2ind, w_emb)
correct_cnt += infer_word('easy', 'easiest', 'luckiest', 'lucky', ind2word, word2ind, w_emb)
correct_cnt += infer_word('luckiest', 'lucky', 'easy', 'easiest', ind2word, word2ind, w_emb)
correct_cnt += infer_word('lucky', 'luckiest', 'easiest', 'easy', ind2word, word2ind, w_emb)

correct_cnt += infer_word('greater', 'great', 'tough', 'tougher', ind2word, word2ind, w_emb)
correct_cnt += infer_word('great', 'greater', 'tougher', 'tough', ind2word, word2ind, w_emb)
correct_cnt += infer_word('tougher', 'tough', 'great', 'greater', ind2word, word2ind, w_emb)
correct_cnt += infer_word('tough', 'tougher', 'greater', 'great', ind2word, word2ind, w_emb)

correct_cnt += infer_word('impossibly', 'possibly', 'ethical', 'unethical', ind2word, word2ind, w_emb)
correct_cnt += infer_word('possibly', 'impossibly', 'unethical', 'ethical', ind2word, word2ind, w_emb)
correct_cnt += infer_word('unethical', 'ethical', 'possibly', 'impossibly', ind2word, word2ind, w_emb)
correct_cnt += infer_word('ethical', 'unethical', 'impossibly', 'possibly', ind2word, word2ind, w_emb)

correct_cnt += infer_word('apparently', 'apparent', 'rapid', 'rapidly', ind2word, word2ind, w_emb)
correct_cnt += infer_word('apparent', 'apparently', 'rapidly', 'rapid', ind2word, word2ind, w_emb)
correct_cnt += infer_word('rapidly', 'rapid', 'apparent', 'apparently'. ind2word, word2ind, w_emb)
correct_cnt += infer_word('rapid', 'rapidly', 'apparently', 'apparent', ind2word, word2ind, w_emb)

print("Accuracy : %.3f" % (correct_cnt / 32))