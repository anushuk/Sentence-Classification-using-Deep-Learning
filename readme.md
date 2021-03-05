![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)
![License](https://img.shields.io/badge/license-Apache%202-red.svg)
![BUILD STATUS](https://img.shields.io/badge/Build-passing-purple.svg)
![dependencies](https://img.shields.io/badge/dependencies-up%20to%20date-green.svg)
![obsesve](https://img.shields.io/badge/observatory-A%2B-yellow.svg)

<h3>Introduction</h3>
Three models are built to classify sentence vectors with mulitple classes using various Deep Learning Techniques.
For now there are two baseline models using TF-IDF and  word2vec embeddings, and one proposed model, which is build using ELMO(Embeddings from Language Models).
Let's look into given files into more details.


<h3>Data</h3>
There are two data files:

1. <b>training.py</b> > used as training data
2. <b>testing.py</b>  > used as testing data

<h3>Text Pre-processingr</h3>
<img src="images/text_steps.png" width="400" height="200">

Read more about <b>Text Pre-processing</b> [here](https://towardsdatascience.com/text-preprocessing-in-natural-language-processing-using-python-6113ff5decd8#:~:text=In%20NLP%2C%20text%20preprocessing%20is,Stop%20words%20removal)  </br>

In this repository you will find <b>textprep.py</b>, which contains all the text pre-processing steps and is imported by all other python files here. </br>
This python script also contains all the necessary NLTK packages and will downloaded automatically. 

<h3>Baseline TF-IDF Vectorizer Model </h3>

<img src="images/tfidf.png" width="400" height="300">

Read more about <b>TF-IDF</b> [here](https://medium.com/@cmukesh8688/tf-idf-vectorizer-scikit-learn-dbc0244a911a)  </br>

For this model, respective files are as follow:

1.<b>tfidf_train.py:</b> It will preprocess the text, it will get TF-IDF vector, create and save labels for dependent variable, train the neural network, and will save the model in the current directory.</br>
To run it `python tfidf_train.py`

2.<b>tfidf_test.py:</b> It will preprocess the text, it will load the saved vector, load the trained model, performe predictions, and will create the result output(testing_result_testidf.csv).</br>
To run it `python tfidf_test.py`

<h3>Baseline Word2Vec Model </h3>

<img src="images/word2vec.jpeg" width="400" height="300">

Read more about <b>Word2Vec</b> [here](https://towardsdatascience.com/introduction-to-word-embedding-and-word2vec-652d0c2060fa)  </br>

For this model, respective files are as follow:

1.<b>word2vec_train.py:</b> It will preprocess the text, it will get  sentence vector by averageing out word vectors and save it as embeddings, create and save labels for dependent variable, train the neural network, and will save the model in the current directory.</br>
To run it `python tfidf_train.py`</br></br>
<b> Note :</b> Before running train file you to download `glove.6B.50d.txt` file. Download from [here](https://nlp.stanford.edu/projects/glove/)

2.<b>word2vec_test.py:</b> It will preprocess the text, it will load the embeddings, load the trained model, performe predictions, and will create the result output(testing_result_word2vec.csv).</br>
To run it `python word2vec_test.py`






<b>Created by :</b>
<b><i> Anubhav Shukla </i></b>
</br>
</br>
![anubhav](https://img.shields.io/badge/Anubhav-%402021-blue.svg)
![status](https://img.shields.io/badge/Status-up-green.svg)
