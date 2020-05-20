from common.util import infer_word, cos_similarity
import pickle
mode = "CBOW"
if mode == "CBOW":
    filename = 'cbowpkl_file'
elif mode == "SG":
    filename = 'sgpkl_file'
with open(filename, 'rb') as f:
    params = pickle.load(f)

w_emb = params['word_vecs']
id2word = params['id2word']
word2id = params['word2id']

infer_word('works', 'work', 'speak', word2id, id2word, w_emb) # works - work + speak = ???
infer_word('work', 'works', 'speaks', word2id, id2word, w_emb)
infer_word('speaks', 'speak', 'work', word2id, id2word, w_emb)
infer_word('speak', 'speaks', 'works', word2id, id2word, w_emb)

infer_word('mice', 'mouse', 'dollar', word2id, id2word, w_emb)
infer_word('mouse', 'mice', 'dollars', word2id, id2word, w_emb)
infer_word('dollars', 'dollar', 'mouse', word2id, id2word, w_emb)
infer_word('dollar', 'dollars', 'mice', word2id, id2word, w_emb)

infer_word('walked', 'walking', 'swimming', word2id, id2word, w_emb)
infer_word('walking', 'walked', 'swam', word2id, id2word, w_emb)
infer_word('swam', 'swimming', 'walking', word2id, id2word, w_emb)
infer_word('swimming', 'swam', 'walked', word2id, id2word, w_emb)

infer_word('thinking', 'think', 'read', word2id, id2word, w_emb)
infer_word('think', 'thinking', 'reading', word2id, id2word, w_emb)
infer_word('reading', 'read', 'think', word2id, id2word, w_emb)
infer_word('read', 'reading', 'thinking', word2id, id2word, w_emb)

infer_word('easiest', 'easy', 'lucky', word2id, id2word, w_emb)
infer_word('easy', 'easiest', 'luckiest', word2id, id2word, w_emb)
infer_word('luckiest', 'lucky', 'easy', word2id, id2word, w_emb)
infer_word('lucky', 'luckiest', 'easiest', word2id, id2word, w_emb)

infer_word('greater', 'great', 'tough', word2id, id2word, w_emb)
infer_word('great', 'greater', 'tougher', word2id, id2word, w_emb)
infer_word('tougher', 'tough', 'great', word2id, id2word, w_emb)
infer_word('tough', 'tougher', 'greater', word2id, id2word, w_emb)

infer_word('impossibly', 'possibly', 'ethical', word2id, id2word, w_emb)
infer_word('possibly', 'impossibly', 'unethical', word2id, id2word, w_emb)
infer_word('unethical', 'ethical', 'possibly', word2id, id2word, w_emb)
infer_word('ethical', 'unethical', 'impossibly', word2id, id2word, w_emb)

infer_word('apparently', 'apparent', 'rapid', word2id, id2word, w_emb)
infer_word('apparent', 'apparently', 'rapidly', word2id, id2word, w_emb)
infer_word('rapidly', 'rapid', 'apparent', word2id, id2word, w_emb)
infer_word('rapid', 'rapidly', 'apparently', word2id, id2word, w_emb)
