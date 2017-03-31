# -*- coding: utf-8 -*-
from bs4 import BeautifulSoup
import requests

def extract_author(recipe_url):
    recipe=requests.get(recipe_url)
    rsoup = BeautifulSoup(recipe.text, "lxml")
    result = ''
    for tag in rsoup.find_all(itemprop='author'):
        result = tag.contents[0]
    return result
