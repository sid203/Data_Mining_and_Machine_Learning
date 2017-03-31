# -*- coding: utf-8 -*-
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

def convert_lower(sentence):
    return str.lower(sentence)

def remove_punctuation(sentence):
    tokenizer = RegexpTokenizer(r'\w+')
    l=tokenizer.tokenize(sentence)
    return (" ".join(l))

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def remove_stop_words(word_list):
    array=[]
    for word in word_list:
        if word not in stopwords.words('english'):
            array.append(word)
    return array


def stemming(word):
    return PorterStemmer().stem_word(word)
