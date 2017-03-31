# -*- coding: utf-8 -*-
import random
from retrying import retry
from bs4 import BeautifulSoup
import requests
import time
import sys
sys.path.insert(0, 'C:\\Users\\conne\\Videos\\ADM\\Persistent Storage\\Extract_Data\\')
import extract_single_recipe as recipe

def write_to_file(Recipe,file_name):
    target=open(file_name,'a',encoding='utf-8')
    for element in Recipe:
        target.write(str(element))
        target.write('\n')
    target.close()
    

def read_all_links(file_name):
    links=[]
    target=open(file_name, 'r',encoding='utf-8')
    for link in target:
        if 'search' not in link:
            if 'shopping' not in link:
                links.append(link)
    target.close()
    return links


def extract_all_information():
    file_name='Machine1.txt'
    links=read_all_links(file_name)
    path='C:\\Users\\conne\\Videos\\ADM\\Links And Downloaded Recipes\\Downloaded Recipes\\Machine1_Recipes.txt'
    for link in links:
        single_recipe=recipe.collect_all_information(link)
        write_to_file(single_recipe,path)
        time.sleep(1)

