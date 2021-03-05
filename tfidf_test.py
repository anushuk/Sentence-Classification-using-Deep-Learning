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

#reading the testing data
dataset_test = pd.read_excel(filepath)
dataset_test_new=dataset_test

#data preprocessing
dataset_test.sentence = dataset_test.sentence.apply(lambda x: text_prep(x))

#column to list
sent=dataset_test['sentence'].tolist()

#vectorizing the test data with saved vectors
pickle_in = open('tfidf_vectorizer.pkl',"rb")
tfidf_vectorizer = pickle.load(pickle_in)
x_test = tfidf_vectorizer.transform(sent).toarray()


#encoder
def encode(le, labels):
    enc = le.transform(labels)
    return to_categorical(enc)

#decoder
def decode(le, one_hot):
    dec = np.argmax(one_hot, axis=1)
    return le.inverse_transform(dec)

#loading the labels back
y_test = list(dataset_test['label'])
pickle_in = open('LabelEncoder_tfidf.pkl',"rb")
le_pickle = pickle.load(pickle_in)
y_test = encode(le_pickle, y_test)
y_test = np.asarray(y_test)

#loading the model
json_file = open("tfidfmodel.json", 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("tfidf_model.h5")

#prediction
predicts = loaded_model.predict(x_test, batch_size=32,verbose=2)
y_test = decode(le_pickle, y_test)
y_preds = decode(le_pickle, predicts)

#results
print("Accuracy : ",accuracy_score(y_test, y_preds))
print("F1 Score : ",f1_score(y_test, y_preds,average='weighted'))
print("confusion_matrix : ",confusion_matrix(y_test, y_preds))

dataset_test_new['predicted_label']=y_preds

#saving the results
dataset_test_new.to_csv("testing_results_tfidf.csv",index=False)

print("Testing Done")
print("Output Created")
