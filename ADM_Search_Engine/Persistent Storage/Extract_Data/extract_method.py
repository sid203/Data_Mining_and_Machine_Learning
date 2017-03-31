# -*- coding: utf-8 -*-
from bs4 import BeautifulSoup
import requests

def extract_method(recipe_url):
    recipe=requests.get(recipe_url)
    rsoup = BeautifulSoup(recipe.text, "lxml")
    result = []
    for tag in rsoup.find_all(itemprop='recipeInstructions'):
        result.append(tag.contents)
    result=clean_method(result)
    final_output=[]
    for instruction in result:
        temp=clean_ind_instruction(instruction)
        final_output.append(temp)
    return " ".join(final_output)
    
def clean_method(instructions):
    temp=[]
    for i in range(len(instructions)):
        temp.append(str(instructions[i][1]))
        
    return temp
    
def clean_ind_instruction(string):
    l=string.split('<')
    l=l[1].split('>')
    string=l[1]
    return string
