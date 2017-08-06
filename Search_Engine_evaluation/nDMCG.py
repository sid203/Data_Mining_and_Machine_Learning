import numpy as np 
import pandas as pd
import fagin as fa

def read_search_result_file(file_name,k):
    file=pd.read_csv(file_name,sep='\t')
    df_lst=[]
    for query_id in np.unique(file.Query_ID):
        tmp_df=file.loc[file.Query_ID==query_id].iloc[:k,:]
        df_lst.append(tmp_df)
    
    final_file=pd.concat(df_lst,ignore_index=True)
    return final_file


def relevance(doc_id,ground_truth):
    if doc_id in ground_truth.Relevant_Doc_id.values:
        return 1
    else:
        return 0    
    
    
def ndmcg(rank_files,ground_truth):
    
    ndcmg_df=pd.DataFrame(columns=['query_id','k=1','k=3','k=5','k=10'])
    ndcmg_df.index.name='nDMCG_score'
    
    all_mcdgscore=[]

    for rank_file in rank_files:
        all_queryid_score=[]
        all_queryids=[]

        for query_id in np.unique(ground_truth.Query_id):
            tmp_grnd_truth=ground_truth.loc[ground_truth.Query_id==query_id]
            tmp_rank_file=rank_file.loc[rank_file.Query_ID==query_id]
            mcdg_score=0
            max_mcdg=0
            for doc_id,rank in zip(tmp_rank_file.Doc_ID[1:].values,tmp_rank_file.Rank[1:].values):
                mcdg_score=mcdg_score+(relevance(doc_id,tmp_grnd_truth)/np.log2(rank))
                max_mcdg=max_mcdg+(1/np.log2(rank))

            mcdg_score=mcdg_score+relevance(tmp_rank_file.Doc_ID[:1].values[0],tmp_grnd_truth)
            max_mcdg=max_mcdg+1
            all_queryid_score.append(mcdg_score/max_mcdg)
            all_queryids.append(query_id)
            
        all_mcdgscore.append(all_queryid_score)
    
    
    #assign values in dataframe
    itr=0
    for col in ndcmg_df.columns[1:].values:
        ndcmg_df[col]=all_mcdgscore[itr]
        itr=itr+1
     
    ndcmg_df.query_id=all_queryids

    return ndcmg_df

def avg_ndcmg(all_df_lst):
    
    avg_df_lst=[]
    for df in all_df_lst:
        avg_df_lst.append(pd.DataFrame(df.iloc[:,1:].mean(axis=0)).T)
        
    avg_df=pd.concat(avg_df_lst,ignore_index=True)
    
    index_names=['agg_text&title','output_cran__EnglishStemmer__CountScorer.tsv',
                 'output_cran__defaultStemmer__Count_scorer.tsv',
                 'output_cran__defaultStemmer__BM25Scorer.tsv',
                 'output_cran__EnglishStemmer_stopwords__TfIdfScorer.tsv',
                 'output_cran__defaultStemmer__TfIdfscorer.tsv',
                 'output_cran__EnglishStemmer_stopwords__CountScorer.tsv',
                 'output_cran__EnglishStemmer_stopwords__BM25Scorer.tsv',
                 'output_cran__EnglishStemmer__BM25Scorer.tsv',
                 'output_cran__EnglishStemmer__TfIdfScorer.tsv']
    
    avg_df.index=index_names
    return avg_df
    

    

path_text='output_cran__EnglishStemmer_stopwords__BM25Scorer_text.tsv'
path_title='output_cran__EnglishStemmer_stopwords__BM25Scorer_title.tsv'
ground_truth_path='cran_Ground_Truth.tsv'

file_names=['output_cran__EnglishStemmer__CountScorer.tsv',
 'output_cran__defaultStemmer__Count_scorer.tsv',
 'output_cran__defaultStemmer__BM25Scorer.tsv',
 'output_cran__EnglishStemmer_stopwords__TfIdfScorer.tsv',
 'output_cran__defaultStemmer__TfIdfscorer.tsv',
 'output_cran__EnglishStemmer_stopwords__CountScorer.tsv',
 'output_cran__EnglishStemmer_stopwords__BM25Scorer.tsv',
 'output_cran__EnglishStemmer__BM25Scorer.tsv',
 'output_cran__EnglishStemmer__TfIdfScorer.tsv']


print('Please put these files in your working directory \n')
for name in file_names:
    print(name+'\n')

print(path_text+'\n'+path_title+'\n'+ground_truth_path+'\n')

print('press y to continue')
while(input()!='y'):
    print('')


ground_truth=pd.read_csv(ground_truth_path,sep = '\t')


#Calculate Score

list_of_k=[1,3,5,10]

rank_file=[fa.write_file(path_text,path_title,i) for i in list_of_k]
rank_file_df=ndmcg(rank_file,ground_truth)
#print(rank_file_df)

all_df_lst=[]
all_df_lst.append(rank_file_df)

for file_name in file_names:
    top_k_file=[read_search_result_file(file_name,i) for i in list_of_k]
    df=ndmcg(top_k_file,ground_truth)
    all_df_lst.append(df)


#Call average function
average_values=avg_ndcmg(all_df_lst)

average_values.to_csv('average_values.tsv',sep = '\t')
print(average_values)
print('\n')
print('average_values written to disk with the filename: average_values.tsv')
