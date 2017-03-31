# -*- coding: utf-8 -*-
import requests
from bs4 import BeautifulSoup
def Extract_Dietary_Information(url):
    recipe = requests.get(url)
    soup = BeautifulSoup(recipe.text, "lxml")
    s="Vegetarian"
    items = soup.find_all("div", {"class":"recipe-metadata__dietary"})
    if bool(items)==True:
        for i in items:
            x = i.find("p", {"class": "recipe-metadata__dietary-vegetarian-text"}).get_text()
            if s in x:
                return(s)
         
        
    else:
        return("No information")
    
