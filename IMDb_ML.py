from sklearn.metrics import confusion_matrix
import tensorflow as tf
import tensorflow_datasets
from transformers import *
from keras.datasets import imdb

physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_visible_devices(physical_devices[1:], 'GPU')

import pickle
import numpy as np
import pandas as pd
import keras
import random
import time
import numpy as np
import gensim
import os

NUM_WORDS = 5000
INDEX_FROM = 3
DIM_BERT = 768
LEN_REVIEW = 300
NUM_REVIEW=25000

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
    
# for  vectorize_sequences_bert 
def restore_text(x_data, index_no):
    return (' '.join(id_to_word[id] for num, id in enumerate(x_data[index_no]) if num<450 ))

# Print Original Texts on pandas ( index 0 - 19)

index_list = range(5)
original_text_list = []
y_train_list = []
for i in index_list:
    original_text_list.append(restore_original_text(i))
    y_train_list.append(y_train[i])
df = pd.DataFrame({'index': index_list, 'x_train original_text': original_text_list, 'y_train': y_train_list})

# bert
def vectorize_sequences_bert(sequences, dimension=DIM_BERT):
    results = np.zeros((len(sequences), dimension))
    zeropad = np.zeros(DIM_BERT)
    extractBertFeature = pipeline('feature-extraction', model='bert-large-cased', tokenizer='bert-large-cased')
    # xlnet-base-cased albert-xxlarge-v2 albert-xlarge-v2 albert-large-v1 albert-large-v2 albert-base-v2  albert-base-v1 bert-large-cased bert-base-cased
    for i, sequence in enumerate(sequences):
        review_vector = np.zeros(dimension)
        review_vector_tmp = np.zeros(dimension)
        len_review_words = 0
        sentence = restore_text(sequences,i)
        sentence=sentence.replace("[🏃]", "")
        sentence=sentence.replace("[❓]", "")
        review_vector_tmp=np.array(extractBertFeature(sentence))[0][0]
        results[i] = review_vector_tmp[0:dimension]*10
    return results

x_train0 = x_train
x_test0 = x_test

# x_train = vectorize_sequences(x_train0)
# x_test = vectorize_sequences(x_test0)
x_train = vectorize_sequences_bert(x_train0)
x_test = vectorize_sequences_bert(x_test0)

#save and load the embedding output
'''
with open('./data/imdb/imdb_x.pickle','wb') as f1:
        pickle.dump((x_train, x_test),f1,pickle.HIGHEST_PROTOCOL)
f1.closed'''

with open('./data/imdb/imdb_x.pickle','rb') as f1:
    (x_train, x_test) = pickle.load(f1)
f1.closed

x_train = x_train[:NUM_REVIEW]
x_test = x_test[:NUM_REVIEW]

y_train0 = y_train[:NUM_REVIEW]
y_test0 = y_test[:NUM_REVIEW]

y_train = np.asarray(y_train0).astype('float32')
y_test = np.asarray(y_test0).astype('float32')

# data

len_review = 768
l_train = max(map(len, x_train.tolist()))
x_train = np.array(list(map(lambda x: x + [0] * (l_train - len(x)), x_train.tolist())))
x_train = x_train[0:, :]
l_test = max(map(len, x_test.tolist()))
x_test = np.array(list(map(lambda x: x + [0] * (l_test - len(x)), x_test.tolist())))
x_test = x_test[0:, :]

# label

y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

# only select 1%  training samples
idxs_annot = range(x_train.shape[0])
random.seed(0)
random.shuffle(np.array(idxs_annot))
idxs_annot = idxs_annot[:int(NUM_REVIEW*0.01)]

x_train_unlabeled = x_train
x_train_labeled = x_train[idxs_annot]
y_train_labeled = y_train[idxs_annot]

# Repeat the labeled dataset to match the shapes

import numpy as np

n_rep = int(x_train_unlabeled.shape[0] / x_train_labeled.shape[0])
x_train_labeled_rep = np.concatenate([x_train_labeled] * n_rep)
y_train_labeled_rep = np.concatenate([y_train_labeled] * n_rep)
y_train_labeled_rep_new = np.array(np.argmax(y_train_labeled_rep, axis=1))
y_test_new=np.array(np.argmax(y_test, axis=1))

#################
#naive bayes
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
y_pred = gnb.fit(x_train_labeled_rep, y_train_labeled_rep_new).predict(x_test)
print(y_test_new)
print(y_pred)
print("Number of mislabeled points out of a total %d points : %d" % (x_test.shape[0], (y_test_new!= y_pred).sum()))
conf_mat=confusion_matrix(y_test_new, y_pred)
print(conf_mat)

#################
#decision tree
from sklearn import tree

clf = tree.DecisionTreeRegressor()
y_pred=clf.fit(x_train_labeled_rep, y_train_labeled_rep_new).predict(x_test)
print(y_test_new)
print(y_pred)
print("Number of mislabeled points out of a total %d points : %d" % (x_test.shape[0], (y_test_new != y_pred).sum()))
conf_mat=confusion_matrix(y_test_new, y_pred)
print(conf_mat)

#################
#svm
from sklearn import svm
clf = svm.LinearSVC()#muilt-class
#.SVC(gamma='scale', decision_function_shape='ovo')2 classes
y_pred=clf.fit(x_train_labeled_rep, y_train_labeled_rep_new).predict(x_test)
print(y_test_new)
print(y_pred)
print("Number of mislabeled points out of a total %d points : %d" % (x_test.shape[0], (y_test_new != y_pred).sum()))

conf_mat=confusion_matrix(y_test_new, y_pred)
print(conf_mat)
