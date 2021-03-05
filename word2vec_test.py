#importing libraries
import pandas as pd
import numpy as np
import os
import sys
from textprep import text_prep
from keras.models import model_from_json
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix
import json
import pickle

#getting the right path
path1 = os.path.normpath(os.getcwd() + os.sep + os.pardir)
filepath=os.path.join(path1, "testing.xlsx")

#reading the testing file
dataset_test = pd.read_excel(filepath)
dataset_test_new=dataset_test

#data preprocessing
dataset_test.sentence = dataset_test.sentence.apply(lambda x: text_prep(x))

#loading the saved embeddings
pickle_in = open('embeddings_index.pkl',"rb")
embeddings_index = pickle.load(pickle_in)

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
x_test_vec= dataset_test.sentence.apply(lambda x: avg_vectors(x))
x_test=np.array(x_test_vec.values.tolist())

#encoder
def encode(le, labels):
    enc = le.transform(labels)
    return to_categorical(enc)

#decoder
def decode(le, one_hot):
    dec = np.argmax(one_hot, axis=1)
    return le.inverse_transform(dec)

#getting label encoders
y_test = list(dataset_test['label'])
pickle_in = open('LabelEncoder_wrd2vec.pkl',"rb")
le_pickle = pickle.load(pickle_in)
y_test = encode(le_pickle, y_test)
y_test = np.asarray(y_test)

#loading the saved model
json_file = open("word2vecmodel.json", 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("word2vec_model.h5")

#model prediction
predicts = loaded_model.predict(x_test, batch_size=32,verbose=2)
y_test = decode(le_pickle, y_test)
y_preds = decode(le_pickle, predicts)

#results
print("Accuracy : ",accuracy_score(y_test, y_preds))
print("F1 Score : ",f1_score(y_test, y_preds,average='weighted'))
print("confusion_matrix : ",confusion_matrix(y_test, y_preds))

dataset_test_new['predicted_label']=y_preds

#saving the results
dataset_test_new.to_csv("testing_result_word2vec.csv",index=False)

print("Testing Done")
print("Output Created")
