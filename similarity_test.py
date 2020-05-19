from common.util import infer_word
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

infer_word('work', 'works', 'speak', word2id, id2word, w_emb) # any - act + that = ???


################## similarity test ##################
# emb_test = w_emb[word2id['works']] - w_emb[word2id['work']] + w_emb[word2id['speak']]
# most_similar_byEmb(emb_test, word2id, id2word, w_emb, top=5)
# emb_test = w_emb[[0][word2id["works"]] - w_emb[word2id["works"]] + w_emb[word2id["speaks"]]
# most_similar_byEmb(emb_test, word2id, id2word, w_emb, top=5)
# emb_test = w_emb[word2id["speaks"]] - w_emb[word2id["speak"]] + w_emb[word2id["work"]]
# most_similar_byEmb(emb_test, word2id, id2word, w_emb, top=5)
# emb_test = w_emb[word2id["speak"]] - w_emb[word2id["speaks"]] + w_emb[word2id["works"]]
# most_similar_byEmb(emb_test, word2id, id2word, w_emb, top=5)

# emb_test = w_emb[word2id["mice"]] - w_emb[word2id["mouse"]] + w_emb[word2id["dollar"]]
# most_similar_byEmb(emb_test, word2id, id2word, w_emb, top=5)
# emb_test = w_emb[word2id["mice"]] - w_emb[word2id["mouse"]] + w_emb[word2id["dollars"]]
# most_similar_byEmb(emb_test, word2id, id2word, w_emb, top=5)
# emb_test = w_emb[word2id["speaks"]] - w_emb[word2id["speak"]] + w_emb[word2id["mouse"]]
# most_similar_byEmb(emb_test, word2id, id2word, w_emb, top=5)
# emb_test = w_emb[word2id["speaks"]] - w_emb[word2id["speak"]] + w_emb[word2id["mice"]]
# most_similar_byEmb(emb_test, word2id, id2word, w_emb, top=5)

# emb_test = w_emb[word2id["walked"]] - w_emb[word2id["walking"]] + w_emb[word2id["swimming"]]
# most_similar_byEmb(emb_test, word2id, id2word, w_emb, top=5)
# emb_test = w_emb[word2id["walked"]] - w_emb[word2id["walking"]] + w_emb[word2id["swam"]]
# most_similar_byEmb(emb_test, word2id, id2word, w_emb, top=5)
# emb_test = w_emb[word2id["swam"]] - w_emb[word2id["swimming"]] + w_emb[word2id["walking"]]
# most_similar_byEmb(emb_test, word2id, id2word, w_emb, top=5)
# emb_test = w_emb[word2id["swam"]] - w_emb[word2id["swimming"]] + w_emb[word2id["walked"]]
# most_similar_byEmb(emb_test, word2id, id2word, w_emb, top=5)

# emb_test = w_emb[word2id["thinking"]] - w_emb[word2id["think"]] + w_emb[word2id["read"]]
# most_similar_byEmb(emb_test, word2id, id2word, w_emb, top=5)
# emb_test = w_emb[word2id["thinking"]] - w_emb[word2id["think"]] + w_emb[word2id["reading"]]
# most_similar_byEmb(emb_test, word2id, id2word, w_emb, top=5)
# emb_test = w_emb[word2id["reading"]] - w_emb[word2id["read"]] + w_emb[word2id["think"]]
# most_similar_byEmb(emb_test, word2id, id2word, w_emb, top=5)
# emb_test = w_emb[word2id["reading"]] - w_emb[word2id["read"]] + w_emb[word2id["thinking"]]
# most_similar_byEmb(emb_test, word2id, id2word, w_emb, top=5)

# emb_test = w_emb[word2id["easiest"]] - w_emb[word2id["easy"]] + w_emb[word2id["lucky"]]
# most_similar_byEmb(emb_test, word2id, id2word, w_emb, top=5)
# emb_test = w_emb[word2id["easiest"]] - w_emb[word2id["easy"]] + w_emb[word2id["luckiest"]]
# most_similar_byEmb(emb_test, word2id, id2word, w_emb, top=5)
# emb_test = w_emb[word2id["luckiest"]] - w_emb[word2id["lucky"]] + w_emb[word2id["easy"]]
# most_similar_byEmb(emb_test, word2id, id2word, w_emb, top=5)
# emb_test = w_emb[word2id["luckiest"]] - w_emb[word2id["lucky"]] + w_emb[word2id["easiest"]]
# most_similar_byEmb(emb_test, word2id, id2word, w_emb, top=5)

# emb_test = w_emb[word2id["greater"]] - w_emb[word2id["great"]] + w_emb[word2id["tough"]]
# most_similar_byEmb(emb_test, word2id, id2word, w_emb, top=5)
# emb_test = w_emb[word2id["greater"]] - w_emb[word2id["great"]] + w_emb[word2id["tougher"]]
# most_similar_byEmb(emb_test, word2id, id2word, w_emb, top=5)
# emb_test = w_emb[word2id["tougher"]] - w_emb[word2id["tough"]] + w_emb[word2id["great"]]
# most_similar_byEmb(emb_test, word2id, id2word, w_emb, top=5)
# emb_test = w_emb[word2id["tougher"]] - w_emb[word2id["tough"]] + w_emb[word2id["greater"]]
# most_similar_byEmb(emb_test, word2id, id2word, w_emb, top=5)

# emb_test = w_emb[word2id["impossibly"]] - w_emb[word2id["possibly"]] + w_emb[word2id["ethical"]]
# most_similar_byEmb(emb_test, word2id, id2word, w_emb, top=5)
# emb_test = w_emb[word2id["impossibly"]] - w_emb[word2id["possibly"]] + w_emb[word2id["unethical"]]
# most_similar_byEmb(emb_test, word2id, id2word, w_emb, top=5)
# emb_test = w_emb[word2id["unethical"]] - w_emb[word2id["ethical"]] + w_emb[word2id["possibly"]]
# most_similar_byEmb(emb_test, word2id, id2word, w_emb, top=5)
# emb_test = w_emb[word2id["unethical"]] - w_emb[word2id["ethical"]] + w_emb[word2id["impossibly"]]
# most_similar_byEmb(emb_test, word2id, id2word, w_emb, top=5)

# emb_test = w_emb[word2id["apparently"]] - w_emb[word2id["apparent"]] + w_emb[word2id["rapid"]]
# most_similar_byEmb(emb_test, word2id, id2word, w_emb, top=5)
# emb_test = w_emb[word2id["apparently"]] - w_emb[word2id["apparent"]] + w_emb[word2id["rapidly"]]
# most_similar_byEmb(emb_test, word2id, id2word, w_emb, top=5)
# emb_test = w_emb[word2id["rapidly"]] - w_emb[word2id["rapid"]] + w_emb[word2id["apparent"]]
# most_similar_byEmb(emb_test, word2id, id2word, w_emb, top=5)
# emb_test = w_emb[word2id["rapidly"]] - w_emb[word2id["rapid"]] + w_emb[word2id["apparently"]]
# most_similar_byEmb(emb_test, word2id, id2word, w_emb, top=5)
# example #
# emb_test = w_emb[word2id["works"]] - w_emb[word2id["works"]] + w_emb[word2id["speak"]]
# most_similar_byEmb(emb_test, word2id, id2word, w_emb, top=5)
#########################################################
