#importing libraries
import pandas as pd
import os
import numpy as np
import pickle
import sys
from textprep import text_prep
from keras.models import Model
from keras.models import model_from_json
from keras.utils import to_categorical
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation,Lambda
from keras.models import Model,model_from_json, Sequential
from sklearn import preprocessing

#getting tensorflow for elmo
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
import tensorflow_hub as hub

#getting the right path
path1 = os.path.normpath(os.getcwd() + os.sep + os.pardir)
filepath=os.path.join(path1, "training.xlsx")

#reading the  training data
dataset = pd.read_excel(filepath)

#data preprocessing
dataset.sentence = dataset.sentence.apply(lambda x: text_prep(x))

y_train = list(dataset['label'])
x_train = list(dataset['sentence'])

#label encoding
lb = preprocessing.LabelEncoder()
lb.fit(y_train)
pickle.dump(lb, open('LabelEncoder_elmo.pkl', 'wb'))

#encoder
def encd(lb, labels):
    enc = lb.transform(labels)
    return to_categorical(enc)

#decoder
def decd(lb, one_hot):
    dec = np.argmax(one_hot, axis=1)
    return lb.inverse_transform(dec)

y_train = encd(lb, y_train)

x_train = np.asarray(x_train)
y_train = np.asarray(y_train)

#downloading the elmo model
url = "https://tfhub.dev/google/elmo/2"
embed = hub.Module(url)

#model building
def elmoembd(x):
    return embed(tf.squeeze(tf.cast(x, tf.string)), signature="default", as_dict=True)["default"]

input_text = Input(shape=(1,), dtype=tf.string)
embedding = Lambda(elmoembd, output_shape=(1024, ))(input_text)
dense1 = Dense(128, activation='relu')(embedding)
dense2= Dropout(0.5)(dense1)
dense3 = Dense(256, activation='relu')(dense2)
dense4= Dropout(0.3)(dense3)
pred = Dense(3, activation='softmax')(dense4)
model = Model(inputs=[input_text], outputs=pred)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#model training
with tf.Session() as session:
    tf.compat.v1.keras.backend.set_session(session)
    session.run(tf.global_variables_initializer())
    session.run(tf.tables_initializer())
    history = model.fit(x_train, y_train, epochs=5, batch_size=32,verbose=1, validation_split=0.2)
    #saving the model
    model_json = model.to_json()
    with open("elmomodel.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("elmo_model.h5")

print("Training Done")
print("Model Saved")
