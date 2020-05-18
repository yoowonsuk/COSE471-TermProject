from common.util import most_similar_byEmb
with open('text8', 'r') as f:
    text = f.read()
# Write your code of data processing, training, and evaluation
# Full training takes very long time. We recommend using a subset of text8 when you debug
#_, word2id, id2word = preprocess(text, subset=1e-4)
with open('cbow_params.pkl', 'rb') as f:
    w_emb, w_out, word2id, id2word = pickle.load(f)
close(f)


################## similarity test ##################
most_similar_byEmb(word, word2id, id2word, w_emb, top=5)
# example #
# emb_test = w_emb[word2id["works"]] - w_emb[word2id["works"]] + w_emb[word2id["speak"]]
# most_similar_byEmb(emb_test, word2id, id2word, w_emb, top=5)
#########################################################
