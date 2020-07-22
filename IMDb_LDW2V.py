from keras.datasets import imdb
import tensorflow as tf
from ladder_net import get_ladder_network_fc

import pandas as pd
import numpy as np
import keras
import random
import time
import pickle
import gensim
import os

NUM_WORDS = 5000
INDEX_FROM = 3
DIM_W2V = 300
LEN_REVIEW = 300

# Load Google's pre-trained Word2Vec model.
model = gensim.models.KeyedVectors.load_word2vec_format('./embedding_src/GoogleNews-vectors-negative300.bin', binary=True)
print(model)


# Load the dataset
(x_train, y_train), (x_test, y_test) = imdb.load_data(path="imdb.npz", num_words=NUM_WORDS, skip_top=0, maxlen=None,
                                                      seed=113, start_char=1, oov_char=2, index_from=INDEX_FROM)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')
print('x_train shape:', x_train.shape)
print('y_train shape:', y_train.shape)
print('x_test shape:', x_test.shape)
print('y_test shape:', y_test.shape)

print(x_train[0])
print(y_train[0:8])
print(x_test)
print(y_test)

# Make Word to ID dictionary
word_to_id = imdb.get_word_index()
word_to_id = {k: (v + INDEX_FROM) for k, v in word_to_id.items()}
word_to_id["[PAD]"] = 0
word_to_id["[🏃]"] = 1  # START
word_to_id["[❓]"] = 2  # UNKNOWN

# Make ID to Word dictionary
id_to_word = {value: key for key, value in word_to_id.items()}

def restore_original_text(index_no):
    return (' '.join(id_to_word[id] for id in x_train[index_no]))

index_list = range(5)
original_text_list = []
y_train_list = []

for i in index_list:
    original_text_list.append(restore_original_text(i))
    y_train_list.append(y_train[i])

df = pd.DataFrame({'index': index_list, 'x_train original_text': original_text_list, 'y_train': y_train_list})
print(df)

# Print Original Texts
for i in index_list:
    print('<< index >> :', i)
    print(restore_original_text(i))
    print('y_train:', y_train[i])
    print('-------------------------------------------------------------------------------')

# one hot
def vectorize_sequences(sequences, dimension=NUM_WORDS):
    results = np.zeros((len(sequences), dimension))
    print('sequences.shape:', sequences.shape)
    print('results.shape:', results.shape)
    for i, sequence in enumerate(sequences):
        # print(i,sequence)
        # print('sequence.shape: ', sequence.shape)
        results[i, sequence] = 1.
        print(results[i])
    return results

# word2vec
def vectorize_sequences_w2v(sequences, model, dimension=DIM_W2V):
    results = np.zeros((len(sequences), dimension))
    zeropad = np.zeros(DIM_W2V)
    for i, sequence in enumerate(sequences):
        review_vector = np.zeros(dimension)
        review_vector_tmp = np.zeros(dimension)
        len_review_words = 0
        for k in sequence: 
            if id_to_word.get(k) != None and model.wv.vocab.get(id_to_word[k]) != None:
                review_vector_tmp += model.wv[id_to_word[k]]
                prev_k = k
                len_review_words += 1
        results[i] = review_vector_tmp[0:dimension] / np.max(np.absolute(review_vector_tmp)) * 30
    return results

x_train0 = x_train
x_test0 = x_test

# x_train = vectorize_sequences(x_train0)
# x_test = vectorize_sequences(x_test0)
x_train = vectorize_sequences_w2v(x_train0, model)
x_test = vectorize_sequences_w2v(x_test0, model)

#save and load the embedding output
'''with open('./data/imdb/imdb_x.pickle','wb') as f1:
        pickle.dump((x_train, x_test),f1,pickle.HIGHEST_PROTOCOL)
f1.closed
with open('./data/imdb/imdb_x.pickle','rb') as f1:
    (x_train, x_test) = pickle.load(f1)
f1.closed'''

y_train0 = y_train
y_test0 = y_test
y_train = np.asarray(y_train0).astype('float32')
y_test = np.asarray(y_test0).astype('float32')
print('y_train:', y_train)
print('y_train:.shape', y_train.shape)

# data

len_review = 300
l_train = max(map(len, x_train.tolist()))
print("l_train=%d" % l_train)
x_train = np.array(list(map(lambda x: x + [0] * (l_train - len(x)), x_train.tolist())))
x_train = x_train[0:, :]
print(x_train[0])
l_test = max(map(len, x_test.tolist()))
print("l_test=%d" % l_test)
print("x_test.shape:", x_test.shape)
x_test = np.array(list(map(lambda x: x + [0] * (l_test - len(x)), x_test.tolist())))
print("x_test.shape:", x_test.shape)
x_test = x_test[0:, :]
print("x_test[0]:", x_test[0])

# label
y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)
print("y_test[9999]:", y_test[9999])

# only select 500 training samples
idxs_annot = range(x_train.shape[0])
print("x_train.shape[0]:", x_train.shape[0])
print("idxs_annot:", idxs_annot)
random.seed(0)
random.shuffle(np.array(idxs_annot))
idxs_annot = idxs_annot[:500]
print("idxs_annot:", idxs_annot)

x_train_unlabeled = x_train
x_train_labeled = x_train[idxs_annot]
y_train_labeled = y_train[idxs_annot]

# Repeat the labeled dataset to match the shapes

n_rep = int(x_train_unlabeled.shape[0] / x_train_labeled.shape[0])
print("x_train_unlabeled.shape[0]", x_train_unlabeled.shape[0])
print("x_train_labeled.shape[0]", x_train_labeled.shape[0])
print("n_rep:", n_rep)
x_train_labeled_rep = np.concatenate([x_train_labeled] * n_rep)
y_train_labeled_rep = np.concatenate([y_train_labeled] * n_rep)
print(y_train_labeled_rep)
print(y_train_labeled_rep.shape)

# Initialize the model

inp_size = len_review 
n_classes = 2
model = get_ladder_network_fc(layer_sizes=[inp_size, 500, 500, 250, 250, 100, n_classes])

# Train the model

t1 = time.time()
print("x_train_labeled_rep.shape:", x_train_labeled_rep.shape)
print("x_train_unlabeled.shape:", x_train_unlabeled.shape)
print("y_train_labeled.shape:", y_train_labeled_rep.shape)
model.fit([x_train_labeled_rep, x_train_unlabeled], y_train_labeled_rep, epochs=50)
t2 = time.time()

# get the test accuracy

from sklearn.metrics import accuracy_score
y_test_pr = model.test_model.predict(x_test, batch_size=1024)
print("test accuracy", accuracy_score(y_test.argmax(-1), y_test_pr.argmax(-1)))
elapsed_time = t2 - t1
print("Elapsed time: %5.3f" % elapsed_time)