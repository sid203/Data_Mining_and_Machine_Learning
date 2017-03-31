from bs4 import BeautifulSoup
import requests
import help

def extract_season_urls():
    months=['january','february','march','april',
           'may','june','july','august','september',
           'october','november','december']
    initial_link='www.bbc.co.uk/food/seasons/'
    home_link='www.bbc.co.uk'
    links=help.initial_data(initial_link)
    every_link=[]
    for link in links:
        for month in months:
            if month in link:
                month_url=home_link+link
                array=help.extract_other_links(month_url)
                every_link=help.add_list(every_link,array)
        
    every_link=set(every_link)
    every_link=list(every_link)
    return every_link

def save_all_links_months():
    home_link='www.bbc.co.uk'
    urls=extract_season_urls()
    l=help.extract_recipes_links(home_link,urls)
    file_name='links2.txt'
    target=open(file_name,'w')
    for link in l:
        target.write(link)
        target.write('\n')
    target.close()
    

