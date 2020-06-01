
from huffman import *
from common.util import preprocess, create_context_target, convert_one_hot
import pickle # for save
with open('text8', 'r') as f:
    text = f.read()
corpus, word2id, id2word = preprocess(text, subset=1.0)
freq = {}
for _id, _ in id2word.items():
    freq[_id] = int((corpus == _id).sum())
huffman = HuffmanCoding()
key2code, code2id = huffman.build(freq)

with open("freq", 'wb') as f:
    pickle.dump(freq, f, protocol=3)
with open("key2code", 'wb') as f:
    pickle.dump(key2code, f, protocol=3)
with open("code2id", 'wb') as f:
    pickle.dump(code2id, f, protocol=3)