# -*- coding: utf-8 -*-
import Help_Clean as help
def clean_single_recipe(Single_recipe):
    new_recipe=[]
    for recipe_element in Single_recipe:
        temp=recipe_element
        temp=help.convert_lower(temp)
        temp=help.remove_punctuation(temp)
        temp=help.tokenize(temp)
        temp=help.remove_stop_words(temp)
        final_words=[]
        for word in temp:
            final_words.append(help.stemming(word))
        final_words=" ".join(final_words)
        new_recipe.append(final_words)
    return new_recipe

def write_to_file(Recipe,file_name):
    target=open(file_name,'a',encoding='utf-8')
    for element in Recipe:
        target.write(str(element))
        target.write('\n')
    target.close()
    
def clean_all_recipes(recipe_file):
    target=open(recipe_file,'r',encoding='utf-8')
    counter=0
    Single_Recipe=[]
    for ind_element in target:
        if ind_element=="\n":
            Single_Recipe.append("No Information")
        else:
            Single_Recipe.append(ind_element)
        counter+=1
        if counter>=3:
            nice_recipe=clean_single_recipe(Single_Recipe)
            write_to_file(nice_recipe,"Clean_Items.txt")
            counter=0
            Single_Recipe[:]=[]
    target.close()
