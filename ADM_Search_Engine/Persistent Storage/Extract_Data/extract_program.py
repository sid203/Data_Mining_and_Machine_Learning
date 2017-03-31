# -*- coding: utf-8 -*-
from bs4 import BeautifulSoup
import requests
def Extract_Program_Information(url):
    recipe1 = requests.get(url)
    soup1 = BeautifulSoup(recipe1.text, "lxml")
    items = soup1.find_all("div", {"class":"recipe-is-from-widget__about"})
    if bool(items)==True:
        for i in items:
            x = i.find("p", {"class": "recipe-is-from-widget__programme-series-title"}).get_text()
            return(x.strip())
        
    else:
        return("No information")
