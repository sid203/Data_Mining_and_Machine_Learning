# -*- coding: utf-8 -*-
from bs4 import BeautifulSoup
import requests

def extract_cooking_time(recipe_url):
    recipe=requests.get(recipe_url)
    rsoup = BeautifulSoup(recipe.text, "lxml")
    result = ''
    for tag in rsoup.find_all(itemprop='cookTime'):
        result = tag.contents[0]
    return result
