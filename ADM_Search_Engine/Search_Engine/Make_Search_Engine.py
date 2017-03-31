import numpy as np
import pandas as pd
import re
from collections import Counter
from math import log10
import math
def get_unique_terms(file_name):
    f = open(file_name, 'r', encoding = 'utf-8')
    l = [i for i in f.readlines()]
    s = list(set([word for line in open(file_name, 'r', encoding = 'utf-8') for word in line.split()]))
    f.close()
    return s,l
def create_disctionary_of_words(file_name):
    unique_words,lines=get_unique_terms(file_name)
    d = {}
    
    for i in range(len(lines)//9):
        if i == 0:
            d[i] = lines[:9]
        else:
            d[i] = lines[(i-1)*9:i*9]
            
    for i in range(len(d)):
        for j in range(len(d[i])):
            d[i][j] = d[i][j][:-1]
            
    return unique_words,d
def cal_doc_freq(file_name):
    s,d=create_disctionary_of_words(file_name)
    doc_freq = np.zeros(len(s))
    for i in range(len(s)):
        for j in range(len(d)):  
            if len(re.findall(r'\b{}\b'.format(s[i]),  str(set(' '.join(d[j]).split())))) > 0:
                doc_freq[i] += 1
    return doc_freq
def word_score(term, Dic, Key, Doc_Freq, file_name): 
    term_freq = len(re.findall(r'\b{}\b'.format(term),  ' '.join(Dic[Key])))
    s = get_unique_terms(file_name)[0]
    tot_n_recipes = len(Dic)
    doc_freq = Doc_Freq[s.index(term)]
    if term_freq == 0 or doc_freq == 0:
        W = 0
    else:
        W = (1+ log10(term_freq))*log10(tot_n_recipes)/doc_freq
    return W
def save_inverted_index(file_name):
    s,d=create_disctionary_of_words(file_name)
    doc_freq=cal_doc_freq(file_name)
    dic = {}
    for i in range(len(s)):
        dic[s[i]] = pd.Series([word_score(s[i], d, j, doc_freq, file_name) for j in range(len(d))], index = range(len(d)))
    df = pd.DataFrame(dic)
    df.to_csv('Output.txt', sep='\t', encoding='utf-8')
    
def run_inverted_index(file_name):
    save_inverted_index(file_name)
