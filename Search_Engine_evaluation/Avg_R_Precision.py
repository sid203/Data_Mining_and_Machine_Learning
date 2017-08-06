import numpy as np 
import pandas as pd
import fagin as fa

def read_all_files(file_names,k):
    all_files=[pd.read_csv(file_name,sep='\t') for file_name in file_names[:-2]]
    path_text=file_names[-2]
    path_title=file_names[-1]
    all_files.append(fa.write_file(path_text,path_title,k))
    return all_files


def relevance(doc_id,ground_truth):
    if doc_id in ground_truth.Relevant_Doc_id.values:
        return 1
    else:
        return 0    


def avg_rprecision(file_names,ground_truth):
    k=int(input('Enter the value of k which will be used to generate top-k file from fagins algo'))
    all_files=read_all_files(file_names,k)
    rprec_df=pd.DataFrame(columns=['query_id']+file_names[:-2]+['agg_file_BM25_EngStemmerstopwords_text&title'])
    rprec_df.index.name='r_precision'

    all_rprecsn=[]
    
    for file in all_files:
        all_queryids=[]
        all_queryidscore=[]
        for query_id in np.unique(ground_truth.Query_id):
            tmp_grnd_truth=ground_truth.loc[ground_truth.Query_id==query_id]
            tmp_file=file.loc[file.Query_ID==query_id]
            total_docs=tmp_grnd_truth.shape[0]
            r_precsn=0
            for doc_id in tmp_file.Doc_ID[:total_docs].values:
                r_precsn=r_precsn+(relevance(doc_id,tmp_grnd_truth))
            
            r_precsn=r_precsn/total_docs
            all_queryids.append(query_id)
            all_queryidscore.append(r_precsn)
        
        all_rprecsn.append(all_queryidscore)
    
    #assign values in dataframe
    itr=0
    for col in rprec_df.columns[1:].values:
        rprec_df[col]=all_rprecsn[itr]
        itr=itr+1
     
    rprec_df.query_id=all_queryids


    return rprec_df
             

ground_truth_path='cran_Ground_Truth.tsv'

file_names=['output_cran__EnglishStemmer__CountScorer.tsv',
 'output_cran__defaultStemmer__Count_scorer.tsv',
 'output_cran__defaultStemmer__BM25Scorer.tsv',
 'output_cran__EnglishStemmer_stopwords__TfIdfScorer.tsv',
 'output_cran__defaultStemmer__TfIdfscorer.tsv',
 'output_cran__EnglishStemmer_stopwords__CountScorer.tsv',
 'output_cran__EnglishStemmer_stopwords__BM25Scorer.tsv',
 'output_cran__EnglishStemmer__BM25Scorer.tsv',
 'output_cran__EnglishStemmer__TfIdfScorer.tsv',
 'output_cran__EnglishStemmer_stopwords__BM25Scorer_text.tsv',
 'output_cran__EnglishStemmer_stopwords__BM25Scorer_title.tsv']

print('Please put these files in your working directory \n')
for name in file_names:
    print(name+'\n')

print(ground_truth_path+'\n')

print('press y to continue')
while(input()!='y'):
    print('')



ground_truth=pd.read_csv(ground_truth_path,sep = '\t')


#Dataframe consisting of R-Precision score of all files. 
r_prcsn_df=avg_rprecision(file_names,ground_truth)

Avg_Rprecision=r_prcsn_df.iloc[:,1:].mean(axis=0)
print(Avg_Rprecision)
Avg_Rprecision.to_csv('Avg_Rprecision.tsv',sep='\t')
print('Average R precision written to disk with the file name:'+'\n')
print('Avg_Rprecision')
