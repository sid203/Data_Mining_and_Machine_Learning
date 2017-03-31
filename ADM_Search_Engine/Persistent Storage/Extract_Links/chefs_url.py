from bs4 import BeautifulSoup
import requests
import help
def extract_chef_urls():
    initial_link='www.bbc.co.uk/food/chefs/'
    home_link='www.bbc.co.uk'
    links=help.initial_data(initial_link)
    every_link=[]
    for link in links:
        if "/food/chefs/by/letters/" in link:
            chef_url=home_link+link
            array=help.extract_other_links(chef_url)
            every_link=help.add_list(every_link,array)
        
    every_link=set(every_link)
    every_link=list(every_link)
    return every_link

def save_all_links_chefs():
    home_link='www.bbc.co.uk'
    urls=extract_chef_urls()
    l=help.extract_recipes_links(home_link,urls)
    file_name='links2.txt'
    target=open(file_name,'w')
    for link in l:
        target.write(link)
        target.write('\n')
    target.close()


