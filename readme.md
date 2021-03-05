![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)
![License](https://img.shields.io/badge/license-Apache%202-red.svg)
![BUILD STATUS](https://img.shields.io/badge/Build-passing-purple.svg)
![dependencies](https://img.shields.io/badge/dependencies-up%20to%20date-green.svg)
![obsesve](https://img.shields.io/badge/observatory-A%2B-yellow.svg)

## Introduction
Three models are built to classify sentence vectors with mulitple classes using various Deep Learning Techniques.
For now there are two baseline models using TF-IDF and  word2vec embeddings, and one proposed model, which is build using ELMO(Embeddings from Language Models).
Let's look into given files into more details.


### Data
There are two data files:

1. <b>training.py</b> > used as training data
2. <b>testing.py</b>  > used as testing data

### TF-IDF Vectorizer
![tfidf](images/tfidf.png){:height="200px" width="500px"}
For this model, respective files are as follow:

1.<b>tfidf_train.py:</b> It will get TF-IDF vector and train the neural network and will save the model in the current directory
A deep dive into the technical parts

There is one data preprocessing file ("textprep.py")

We have two baseline models within folders " Baseline_tfidf" and "Basline_word2vec"

1) In folder "Baseline_tfidf"  you will find 'tfidf_train.py' and 'tfidf_test.py' for training and testing the model respectively and command to train is
 "python tfidf_train.py"
and to test already trained saved model run
" python tfidf_test.py"
You will also find output file here named "testing_output_testidf.csv"

2) In folder "Basline_word2vec"  you will find  'word2vec_train.py' and 'word2vec_test.py' for training and testing the model respectively and command to train is
 "python word2vec_train.py"
and to test already trained saved model run
" python word2vec_test.py"
You will also find output file here named "testing_output_word2vec.csv"

3) In folder "Propesed_elmo"  you will find  'elmo_train.py' and 'elmo_test.py' for training and testing the model respectively and command to train is
 "python elmo_train.py"
and to test already trained saved model run
" python elmo_test.py"
You will also find output file here named "testing_output_elmo.csv"



<b>Created by :</b>
<b><i> Anubhav Shukla </i></b>
</br>
</br>
![anubhav](https://img.shields.io/badge/Anubhav-%402021-blue.svg)
![status](https://img.shields.io/badge/Status-up-green.svg)
