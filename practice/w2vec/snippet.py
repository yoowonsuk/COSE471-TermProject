# coding: utf-8
import sys
sys.path.append('..')  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
from common.optimizer import SGD
from common.util import preprocess, create_context_target, convert_one_hot
from common.trainer import Trainer
import torch
from w2vec.CBow import CustomCBOW
from data import ptb

window_size = 5
hidden_size = 5
batch_size = 3
max_epoch = 5000

with open('text8', 'r') as f:
    text = f.read()
corpus, word_to_id, id_to_word = preprocess(text, subset=1e-4)
print("complete")
vocab_size = len(word_to_id)
contexts, target = create_context_target(corpus, window_size)
target = convert_one_hot(target, vocab_size)
contexts = convert_one_hot(contexts, vocab_size).float()
model = CustomCBOW(vocab_size, hidden_size, vocab_size, contexts.shape[1])
optimizer = SGD()
trainer = Trainer(model, optimizer)

trainer.fit(contexts, target, max_epoch, batch_size)
trainer.plot()

word_vecs = model.word_vecs
for word_id, word in id_to_word.items():
    print(word, word_vecs[word_id])