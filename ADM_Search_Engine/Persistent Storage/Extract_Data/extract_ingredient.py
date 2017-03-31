# -*- coding: utf-8 -*-
from bs4 import BeautifulSoup
import requests
import time

def extract_ingredients(recipe_url):
    recipe=requests.get(recipe_url)
    rsoup = BeautifulSoup(recipe.text, "lxml")
    result = []
    for tag in rsoup.find_all(itemprop='ingredients'):
        if len(tag.contents)>1:
            result.append(tag.contents)
        else:
            result.append([tag.contents[0]])
    result=clean_ingredients(result)
    result=join_ingredients(result)
    return " ".join(result)

def clean_ingredients(ingredients):
    
    for i in range(len(ingredients)):
        for j in range(len(ingredients[i])):
            if 'class' in str(ingredients[i][j]) or 'href' in str(ingredients[i][j]):
                string=str(ingredients[i][j])
                string=clean_string(string)
                ingredients[i][j]=string
    return ingredients
                
def clean_string(string):
    l=string.split('<')
    string=l[1]
    l=string.split('>')
    string=l[1]
    return string

def join_ingredients(ingredients):
    results=[]
    for ingredient in ingredients:
        string=add_ingredient(ingredient)
        results.append(string)
    return results
def add_ingredient(ingredient):
    string=''
    for element in ingredient:
        string=str.join('',(string,element))
    return string
