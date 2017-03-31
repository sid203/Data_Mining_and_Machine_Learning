# -*- coding: utf-8 -*-
from bs4 import BeautifulSoup
import requests
import sys
sys.path.insert(0, 'C:\\Users\\conne\\Videos\\ADM\\Extract_Data\\')
import extract_author as author
import extract_cook_time as ct
import extract_prep_time as pt
import extract_ingredient as ingredient
import extract_method as method
import extract_serves as serves
import extract_dietary as dt
import extract_program as pg
import extract_title as title

def collect_all_information(url):
    ## Use the Functions to get the Data #
    Single_Recipe_Data=[]
    auth=author.extract_author(url)
    p_time=pt.extract_prep_time(url)
    c_time=ct.extract_cooking_time(url)
    sv=serves.extract_serves(url)
    dietary=dt.Extract_Dietary_Information(url)
    progrm=pg.Extract_Program_Information(url)
    ing=ingredient.extract_ingredients(url)
    m=method.extract_method(url)
    t=title.extract_title(url)
    # Save the Data together in one, Variable , and return that Variable
    Single_Recipe_Data.append(t)
    Single_Recipe_Data.append(auth)
    Single_Recipe_Data.append(p_time)
    Single_Recipe_Data.append(c_time)
    Single_Recipe_Data.append(dietary)
    Single_Recipe_Data.append(progrm)
    Single_Recipe_Data.append(sv)
    Single_Recipe_Data.append(ing)
    Single_Recipe_Data.append(m)
    return Single_Recipe_Data
