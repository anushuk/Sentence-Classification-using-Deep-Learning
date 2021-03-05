import pandas as pd
import numpy as np
import os
import pickle
import sys
from textprep import text_prep
from keras.utils import to_categorical
from keras.models import model_from_json
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix

#getting tensorflow for elmo
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
import tensorflow_hub as hub

#downloading the elmo model
url = "https://tfhub.dev/google/elmo/2"
embed = hub.Module(url)

#getting the right path
path1 = os.path.normpath(os.getcwd() + os.sep + os.pardir)
filepath=os.path.join(path1, "testing.xlsx")

#reading the  testing data
dataset_test = pd.read_excel(filepath)
dataset_test_new=dataset_test

#data preprocessing
dataset_test.sentence = dataset_test.sentence.apply(lambda x: text_prep(x))

y_test = list(dataset_test['label'])
x_test = list(dataset_test['sentence'])

#label encoding
pickle_in = open('LabelEncoder_elmo.pkl',"rb")
le_pickle = pickle.load(pickle_in)

#encoder
def encode(lb, labels):
    enc = lb.transform(labels)
    return to_categorical(enc)

#decoder
def decode(lb, one_hot):
    dec = np.argmax(one_hot, axis=1)
    return lb.inverse_transform(dec)

y_test = encode(le_pickle, y_test)

x_test = np.asarray(x_test)
y_test = np.asarray(y_test)

#prediction
with tf.Session() as session:
    tf.compat.v1.keras.backend.set_session(session)
    session.run(tf.global_variables_initializer())
    session.run(tf.tables_initializer())
    json_file = open('elmomodel.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("elmo_model.h5")
    predicts = loaded_model.predict(x_test, batch_size=32,verbose=2)

y_test = decode(le_pickle, y_test)
y_preds = decode(le_pickle, predicts)

#results
print("Accuracy : ",accuracy_score(y_test, y_preds))
print("F1 Score : ",f1_score(y_test, y_preds,average='weighted'))
print("confusion_matrix : ",confusion_matrix(y_test, y_preds))

dataset_test_new['predicted_label']=y_preds

#saving the results
dataset_test_new.to_csv("testing_result_elmo.csv",index=False)

print("Testing Done")
print("Output Created")
