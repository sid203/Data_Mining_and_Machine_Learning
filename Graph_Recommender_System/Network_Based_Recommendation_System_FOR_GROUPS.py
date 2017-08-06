import csv
import time
import pprint as pp
import networkx as nx

import Network_Based_Recommendation_System_FUNCTIONS as homework_2

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
    
from operator import itemgetter, attrgetter

def fagin_algo(tmp_text,tmp_title,k):
    seen_elements=[]
    seen_elements_both=[]
    for i,j in zip(tmp_text.movie_id,tmp_title.movie_id):
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
        over_all_score.append(tmp_text.loc[tmp_text.movie_id==elmt].score.values[0]+tmp_title.loc[tmp_title.movie_id==elmt].score.values[0])
    
    df = pd.DataFrame(over_all_score,seen_elements_both,columns=['score']).sort(columns='score',ascending=False)
    return list(df.index)[:k]

def create_preference_vector_for_teleporting_group(group_dct, graph_users_items):
    
    pref_vec_list=[]
    sum_group_val = 0
    for user_id in group_dct.keys():
        sum_group_val=sum_group_val + group_dct[user_id]
        #create the preference vector for a given user in group
        pref_vec = homework_2.create_preference_vector_for_teleporting(user_id,graph_users_items)
        #weight the the ratings according to the importance of this user
        for key in pref_vec.keys():
            pref_vec[key] = pref_vec[key] * group_dct[user_id]
            
        pref_vec_list.append(pref_vec) #create a list of weighted preference vectors and then add them up

    # merge these preference vectors
    merged = merge3(pref_vec_list)
    for key in merged.keys():
        merged[key]=merged[key]/sum_group_val

    
    return(merged)

    
def merge3(dicts):
    merged = {}
    for d in dicts:
        for k in d.keys():
            if k in merged.keys():
                merged[k] = merged[k] + d[k]
            else:
                merged[k]=d[k]
    return merged


print()
print ("Current time: " + str(time.asctime(time.localtime())))
print()
print()


all_groups = [
    {1701: 1, 1703: 1, 1705: 1, 1707: 1, 1709: 1}, ### Movie night with friends.
    {1701: 1, 1702: 4}, ### First appointment scenario: the preferences of the girl are 4 times more important than those of the man.
    {1701: 1, 1702: 2, 1703: 1, 1704: 2}, ### Two couples scenario: the preferences of girls are still more important than those of the men...
    {1701: 1, 1702: 1, 1703: 1, 1704: 1, 1705: 1, 1720:10}, ### Movie night with a special guest.
    {1701: 1, 1702: 1, 1703: 1, 1704: 1, 1705: 1, 1720:10, 1721:10, 1722:10}, ### Movie night with 3 special guests.
]
print()
pp.pprint(all_groups)
print()


graph_file = "./input_data/u_data_homework_format.txt"

pp.pprint("Load Graph.")
print ("Current time: " + str(time.asctime(time.localtime())))
graph_users_items = homework_2.create_graph_set_of_users_set_of_items(graph_file)
print (" #Users in Graph= " + str(len(graph_users_items['users'])))
print (" #Items in Graph= " + str(len(graph_users_items['items'])))
print (" #Nodes in Graph= " + str(len(graph_users_items['graph'])))
print (" #Edges in Graph= " + str(graph_users_items['graph'].number_of_edges()))
print ("Current time: " + str(time.asctime(time.localtime())))
print()
print()


pp.pprint("Create Item-Item-Weighted Graph.")
print ("Current time: " + str(time.asctime(time.localtime())))
item_item_graph = homework_2.create_item_item_graph(graph_users_items)
print (" #Nodes in Item-Item Graph= " + str(len(item_item_graph)))
print (" #Edges in Item-Item Graph= " + str(item_item_graph.number_of_edges()))
print ("Current time: " + str(time.asctime(time.localtime())))
print()
print()


### Conversion of the 'Item-Item-Graph' to a scipy sparse matrix representation.
### This reduces a lot the PageRank running time ;)
print()
print (" Conversion of the 'Item-Item-Graph' to a scipy sparse matrix representation.")
N = len(item_item_graph)
nodelist = item_item_graph.nodes()
M = nx.to_scipy_sparse_matrix(item_item_graph, nodelist=nodelist, weight='weight', dtype=float)
print (" Done.")
print()
#################################################################################################


output_file = open("./Output_Recommendation_for_Group.tsv", 'w')
output_file_csv_writer = csv.writer(output_file, delimiter='\t', quotechar='"', quoting=csv.QUOTE_NONE)
print()
for current_group in all_groups:
    print ("Current group: ")
    pp.pprint(current_group)
    print ("Current time: " + str(time.asctime(time.localtime())))
    
    sorted_list_of_recommended_items_for_current_group = []
    # Your code here ;)
    #
    tmp = create_preference_vector_for_teleporting_group(current_group,graph_users_items)
    personalized_pagerank_vector_of_items = homework_2.pagerank(M, N, nodelist, alpha=0.85, personalization=tmp)
    sorted_list_of_recommended_items_for_current_group=sorted(personalized_pagerank_vector_of_items, key=lambda key: personalized_pagerank_vector_of_items[key],reverse=True)

    
    print ("Recommended Sorted List of Items:")
    print(str(sorted_list_of_recommended_items_for_current_group[:30]))
    print()
    output_file_csv_writer.writerow(sorted_list_of_recommended_items_for_current_group)

output_file.close()




print()
print()
print ("Current time: " + str(time.asctime(time.localtime())))
print ("Done ;)")
print()

