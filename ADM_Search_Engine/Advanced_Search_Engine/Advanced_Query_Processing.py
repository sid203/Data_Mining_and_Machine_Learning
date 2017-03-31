import numpy as np
import pandas as pd
import re
from collections import Counter
from math import log10
import math
import sys
sys.path.insert(0, 'C:\\Users\\conne\\Videos\\ADM\Advanced_Search_Engine\\')
import Make_Search_Engine as SE
import Help_Clean as he

def clean_query(query):
    query=he.convert_lower(query)
    query=he.remove_punctuation(query)
    query=he.tokenize(query)
    query=he.remove_stop_words(query)
    final_words=[]
    for word in query:
        final_words.append(he.stemming(word))
    #final_words=" ".join(final_words)
    return final_words

def cosine_similarity(v1,v2):
    sumxx, sumxy, sumyy = 0, 0, 0
    for i in range(len(v1)):
        x = v1[i]; y = v2[i]
        sumxx += x*x
        sumyy += y*y
        sumxy += x*y
    return sumxy/math.sqrt(sumxx*sumyy)

def word_score_query(vector, dictionary):
    set_vector = sorted(set(vector))
    term_freq = np.zeros(len(set_vector))
    tot_n_recipes = len(dictionary)+1
    doc_freq = np.ones(len(set_vector))
    for i in range(len(set_vector)):
        term_freq[i] = len(re.findall(r'\b{}\b'.format(set_vector[i]),  ' '.join(vector)))
    for i in range(len(set_vector)):
        for j in range(len(dictionary)):  
            if len(re.findall(r'\b{}\b'.format(set_vector[i]),  str(set(' '.join(dictionary[j]).split())))) > 0:
                doc_freq[i] += 1
    output = np.zeros(len(set_vector))
    for i in range(len(output)):
        output[i] = ((1+ log10(term_freq[i]))*log10(tot_n_recipes)/doc_freq[i]) if (term_freq[i] > 0 and doc_freq[i] > 0) else 0
    return output


def query_processing(Dict,file):
    query = input()
    query=clean_query(query)
    vector1 = word_score_query(query, Dict)
    f=pd.read_csv(file, delimiter='\t', index_col=0)
    ranking={k: cosine_similarity(vector1,f.iloc[v,:].values) for k,v in enumerate(range(len(Dict)))}
    sorted_ranking = sorted(ranking, reverse=True, key = ranking.__getitem__)
    return (sorted_ranking[:10])
    
def run_query():
    input_file="sample1.txt"
    Dict=SE.create_disctionary_of_words(input_file)[1]
    return query_processing(Dict,"Output_Small.txt")
