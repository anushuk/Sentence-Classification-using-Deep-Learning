#importing libraries
import pandas as pd
import numpy as np
import pickle
import re
import string
import nltk
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import WhitespaceTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

# all downloads
nltk.download('stopwords')

nltk.download('averaged_perceptron_tagger')

nltk.download('wordnet')

nltk.download('punkt')

#word tagging
def get_wordnet_pos(pos_tag):
    if pos_tag.startswith('J'):
        return wordnet.ADJ
    elif pos_tag.startswith('V'):
        return wordnet.VERB
    elif pos_tag.startswith('N'):
        return wordnet.NOUN
    elif pos_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

#text preprocessing
def text_prep(input_str):

    input_str=str(input_str)

    input_str=''.join([i for i in input_str if not i.isdigit()])

    #lower case string
    input_str = input_str.lower()


    #panctuation
    input_str = [word.strip(string.punctuation) for word in input_str.split(" ")]


    #stopwords
    stop = stopwords.words('english')
    input_str = [x for x in input_str if x not in stop]

    #singleletter
    input_str = [t for t in input_str if len(t) > 0]

    #Part-of-speech tagging and lemmatization
    pos_tags = pos_tag(input_str)

    input_str = [WordNetLemmatizer().lemmatize(t[0], get_wordnet_pos(t[1])) for t in pos_tags]

    #twoletter
    input_str = [t for t in input_str if len(t) > 1]

    #list to string
    input_str = " ".join(input_str)

    return input_str
