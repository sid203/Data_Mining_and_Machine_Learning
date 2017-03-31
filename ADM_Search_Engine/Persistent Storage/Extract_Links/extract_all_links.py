from bs4 import BeautifulSoup
import requests
import chefs_url as chef
import cuisine_url as cuisine
import dishes_url as dishes
import ingredient_url as ingredient
import occassion_url as oc
import season_url as season

def run_everything():
    chef.save_all_links_chefs()
    cuisine.save_all_links_cuisines()
    dishes.save_all_links_dishes()
    ingredient.ave_all_links_ingredients()
    oc.save_all_links_occasions()
    season.save_all_links_months()
