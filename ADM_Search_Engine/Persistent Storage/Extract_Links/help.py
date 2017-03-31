from bs4 import BeautifulSoup
import requests

######################
def initial_data(url):
    food_links=[]
    r  = requests.get("http://" +url)
    data = r.text 
    soup = BeautifulSoup(data,'lxml')
    for link in soup.find_all('a'):
        if(link.get('href').startswith('/food/')):
            food_links.append(link.get('href'))
    return food_links
def extract_recipes_links(home_link,urls):
    food_links=[]
    for link in urls:
        url=home_link+link
        r  = requests.get("http://" +url)
        data = r.text 
        soup = BeautifulSoup(data,'lxml')
        for link in soup.find_all('a'):
            if(link.get('href').startswith('/food/recipes')):
                food_links.append(link.get('href'))
    food_links=set(food_links)
    food_links=list(food_links)
    return food_links
def extract_other_links(url):
    food_links=[]
    r  = requests.get("http://" +url)
    data = r.text 
    soup = BeautifulSoup(data,'lxml')
    for link in soup.find_all('a'):
        if(link.get('href').startswith('/food/')):
            food_links.append(link.get('href'))
    return food_links
def add_list(L1,L2):
    for element in L2:
        L1.append(element)
    return L1

############################
