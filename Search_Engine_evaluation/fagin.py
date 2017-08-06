import numpy as np
import pandas as pd

def read_files(text_file_path,title_file_path):
    
    BM25Scorer_text=pd.read_csv(text_file_path,sep = '\t')
    BM25Scorer_title=pd.read_csv(title_file_path,sep = '\t')
    
    return BM25Scorer_text,BM25Scorer_title


def check_both_list(element_lst,seen_elements):
    for i in seen_elements:
        chk_lst=[]
        for j in i:
            chk_lst.append(j)
        if chk_lst==element_lst:
            return True
        elif chk_lst==element_lst:
            return True
    
    return False
    


def fagin_algo(tmp_text,tmp_title,k):
    seen_elements=[]
    seen_elements_both=[]
    for i,j in zip(tmp_text.Doc_ID,tmp_title.Doc_ID):
        if check_both_list([i,0,1],seen_elements)==True:
            seen_elements_both.append(i)
        if check_both_list([j,1,0],seen_elements)==True:
            seen_elements_both.append(j)
        
        #seen elements    
        seen_elements.append([i,1,0])
        seen_elements.append([j,0,1])

        if len(seen_elements_both)>=k:
            break
        
    over_all_score = []
    for elmt in seen_elements_both:
        over_all_score.append(tmp_text.loc[tmp_text.Doc_ID==elmt].Score.values[0]+tmp_title.loc[tmp_title.Doc_ID==elmt].Score.values[0])
    
    df = pd.DataFrame(over_all_score,seen_elements_both,columns=['score']).sort(columns='score',ascending=False)
    return list(df.index)[:k]

def write_file(path_text,path_title,k):
    BM25Scorer_text,BM25Scorer_title=read_files(path_text,path_title)

    agg_file=pd.DataFrame(columns=BM25Scorer_text.columns.values) #output file
    
    all_scores=[]
    all_docids=[]
    all_queries=[]
    all_ranks=[]
    for queryid in np.unique(BM25Scorer_text.Query_ID):
        tmp_text=BM25Scorer_text.loc[BM25Scorer_text.Query_ID==queryid]
        tmp_title=BM25Scorer_title.loc[BM25Scorer_title.Query_ID==queryid]
        top_k=fagin_algo(tmp_text,tmp_title,k)
        rank=0
        for elmnt in top_k:
            txt_score=tmp_text.loc[tmp_text.Doc_ID==elmnt].Score.values[0]
            title_score=tmp_title.loc[tmp_title.Doc_ID==elmnt].Score.values[0]
            cmbnd_score=2*title_score+txt_score
            rank=rank+1
            all_scores.append(cmbnd_score)
            all_docids.append(elmnt)
            all_queries.append(queryid)
            all_ranks.append(rank)
    
    agg_file['Query_ID']=all_queries
    agg_file['Doc_ID']=all_docids
    agg_file['Score']=all_scores
    agg_file['Rank']=all_ranks
    
    return agg_file



path_text='output_cran__EnglishStemmer_stopwords__BM25Scorer_text.tsv'
path_title='output_cran__EnglishStemmer_stopwords__BM25Scorer_title.tsv'

print('Please put these files in your working directory \n')
print('1.'+path_text+'\n'+'2.'+path_title+'\n')

k=int(input('enter the value of k'))
agg_file=write_file(path_text,path_title,k)
print(agg_file)
print('\n')
agg_file.to_csv('agg_file.tsv',sep = '\t')
print('agg_file written to disk with filename:agg_file.tsv')


