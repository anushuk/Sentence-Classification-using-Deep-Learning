#importing libraries
import pandas as pd
import numpy as np
import os
import sys
from textprep import text_prep
from keras.layers import Dense, Dropout
from keras.models import Model
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from sklearn import preprocessing
import json
import pickle

#getting the right path
path1 = os.path.normpath(os.getcwd() + os.sep + os.pardir)
filepath=os.path.join(path1, "training.xlsx")

#reading the  training data
dataset = pd.read_excel(filepath)

#data preprocessing
dataset.sentence = dataset.sentence.apply(lambda x: text_prep(x))


embeddings_index={}
#reading embeddings
#remember to download "glove.6B.50d.txt" from the given link in the readme.md file
with open('glove.6B.50d.txt','r',encoding='utf-8') as embd_file:
    for line in embd_file:
        val = line.split()
        word = val[0]
        cfs = np.asarray(val[1:], dtype='float32')
        embeddings_index[word] = cfs

#saving the embeddings
pickle.dump(embeddings_index, open('embeddings_index.pkl', 'wb'))

#word embedding for a sentence by averaging it out.
def avg_vectors(sentence):
    sentence = [token for token in sentence.split() if  token in embeddings_index]
    embedding_size = 50
    vs = np.zeros(embedding_size)
    sentence_length = len(sentence)
    for word in sentence:
        vs = np.add(vs, embeddings_index.get(word))
    vs = np.divide(vs, sentence_length)
    return vs
    
#getting sentence vectors
x_train_vec= dataset.sentence.apply(lambda x: avg_vectors(x))
x_train=np.array(x_train_vec.values.tolist())

#  label encoding the variables
y_train = list(dataset['label'])
le = preprocessing.LabelEncoder()
le.fit(y_train)

#saving label encodings
pickle.dump(le, open('LabelEncoder_wrd2vec.pkl', 'wb'))

#encoder
def encode(le, labels):
    enc = le.transform(labels)
    return to_categorical(enc)

#decoder
def decode(le, one_hot):
    dec = np.argmax(one_hot, axis=1)
    return le.inverse_transform(dec)

y_train = encode(le, y_train)
y_train = np.asarray(y_train)

#model building
model = Sequential()
model.add(Dense(128, activation='sigmoid',input_shape=(50,)))
model.add(Dropout(0.3))
model.add(Dense(64, activation='sigmoid'))
model.add(Dropout(0.3))
model.add(Dense(3, activation="softmax"))
model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])

#model training
history = model.fit(x_train, y_train, batch_size = 32, epochs = 100, validation_split = .2)

#saving the trained model
model_json = model.to_json()
with open("word2vecmodel.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("word2vec_model.h5")

print("Training Done")
print("Model Saved")
