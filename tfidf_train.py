#importing  libraries
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
from sklearn.feature_extraction.text import TfidfVectorizer
import json
import pickle

#getting the right path
path1 = os.path.normpath(os.getcwd() + os.sep + os.pardir)
filepath=os.path.join(path1, "training.xlsx")

#reading the  training data
dataset = pd.read_excel(filepath)

#data preprocessing
dataset.sentence = dataset.sentence.apply(lambda x: text_prep(x))

#column to list
sent=dataset['sentence'].tolist()

#vectorizing the data
tfidf_vectorizer = TfidfVectorizer()

#preparing train data
x_train = tfidf_vectorizer.fit_transform(sent).toarray()

#saving vectorized data
pickle.dump(tfidf_vectorizer, open('tfidf_vectorizer.pkl', 'wb'))

# getting dependent variable
y_train = list(dataset['label'])

#  label encoding the variables
le = preprocessing.LabelEncoder()
le.fit(y_train)

# saving the labels
pickle.dump(le, open('LabelEncoder_tfidf.pkl', 'wb'))

#encoding the labels
def encode(le, labels):
    enc = le.transform(labels)
    return to_categorical(enc)

#decoding the labels
def decode(le, one_hot):
    dec = np.argmax(one_hot, axis=1)
    return le.inverse_transform(dec)

y_train = encode(le, y_train)
y_train = np.asarray(y_train)


#model building
model = Sequential()
model.add(Dense(128, activation='sigmoid',input_shape=(9908,)))
model.add(Dropout(0.5))
model.add(Dense(64, activation='sigmoid'))
model.add(Dropout(0.3))
model.add(Dense(3, activation="softmax"))
model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])

#model Training
history = model.fit(x_train, y_train, batch_size = 32, epochs = 50, validation_split = .2)
model_json = model.to_json()

#saving the model
with open("tfidfmodel.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("tfidf_model.h5")

print("Training Done")
print("Model Saved")
