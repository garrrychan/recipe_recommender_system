import requests
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import re
import json
from time import sleep, time
import random

class util:
    def __init__(self):
        pass
    def clean(ingredients):
        '''Remove numbers, punctuation and non alpha numeric chars from ingredient list'''
        new_lst = []
        for i in ingredients:
            new_lst.append(re.sub(r'([12345678910]\s*| \(.*\?\)|[:()-,".\/])','', i))
        return new_lst

    def convert_to_mins(str):
        '''Returns time in all minutes
        convert_to_mins('2 h 5 m') >>> 125'''
        lst = re.sub('d',"1440",re.sub('m',"1",re.sub('h',"60",str))).split()
        if len(lst) == 6: # days, hours and mins
            return f'{int(lst[0])*int(lst[1])+int(lst[2])*int(lst[3])+int(lst[4])}'
        if len(lst) == 4: # hours and mins
            return f'{int(lst[0])*int(lst[1])+int(lst[2])}'
        if len(lst) == 2: # only mins
            return f'{int(lst[0])}'
        else: # fix this
            raise ValueError('Does not handle this str')

##### recipe_ids #####
def get_page(page=''):
    '''Parse HTML of page with BeautifulSoup
    get_page(page='58')
    '''
    URL = "https://www.allrecipes.com/recipes/16791/everyday-cooking/special-collections/web-show-recipes/food-wishes/"
    params = {'page': page}
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.131 Safari/537.36'}
    response = requests.get(URL, headers=headers, params=params)
    soup = BeautifulSoup(response.content,'lxml')
    return soup

def get_recipe_ids(soup):
    '''Return all recipe_ids from a page
    get_recipe_ids(get_page("58"))
    '''
    recipe_ids = []
    for i in soup.find_all("article", {"class" : "fixed-recipe-card"}):
        recipe_ids.append(i.find("ar-save-item").attrs["data-id"])
    return recipe_ids

def all_recipe_ids(pages=58+1):
    '''Return all recipes for Foodwishes, max 59 pages!'''
    all_recipe_ids = []
    page_nums = range(1, pages)
    for i in page_nums:
        print(f'Getting page {i+1}')
        all_recipe_ids.extend(get_recipe_ids(get_page(page=i)))
        # prevent server issues
        sleep(random.randint(0,2))
    return all_recipe_ids

def get_recipe(id):
    '''Return a recipe for one ID dictionary format
    get_recipe(266825)'''
    URL = "https://www.allrecipes.com/recipe/"
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.131 Safari/537.36'}
    response = requests.get(URL+f'{id}',headers=headers)
    soup = BeautifulSoup(response.content,"lxml")
    recipe = {}
    try: # try if html soup is in the right format
        recipe["recipe_id"] = id
        recipe['reviews'] = soup.find("span", {"class":"review-count"}).text.split(' ')[0]
        recipe["title"] = soup.find('h1').text
        recipe["ratings"] = soup.find("meta", {"property": "og:rating"})["content"]
        try:
            recipe["calories"] = soup.find('span',{"class":"calorie-count"}).text.split()[0]
        except: # recipe with missing calories
            recipe["calories"] = 0
        category = []
        for i in soup.find_all('meta',{"itemprop":"recipeCategory"}):
            category.append(i["content"])
        recipe["category"] = category
        try:
            recipe["total_mins"] = util.convert_to_mins(soup.find('span',{'class': 'ready-in-time'}).text)
        except: # no time data
            recipe["total_mins"] = 0
    except: # skip with empty recipes
        raise ValueError('Does not handle this html page')
    ingredients = []
    # only get checklist ingredients
    for i in soup.find_all("label",{"ng-class":"{true: 'checkList__item'}[true]"}):
        ingredients.append(i.find("span",{"class":"recipe-ingred_txt added"}).text)
        recipe["ingredients"] = ingredients
    return recipe

def get_recipe_2(id):
    '''Return a recipe for one ID dictionary format
    get_recipe_2(247092), slightly different page format'''
    URL = "https://www.allrecipes.com/recipe/"
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.131 Safari/537.36'}
    response = requests.get(URL+f'{id}',headers=headers)
    soup = BeautifulSoup(response.content,"lxml")
    recipe = {}
    try: # try if html soup is in the right format
        recipe["recipe_id"] = id
        #  many ratings
        try: #"2Ratings"
            recipe["reviews"] = int(re.sub("[\s+\n]","",soup.find("span",{"class":"ugc-ratings-item"}).text)[:-7])
        except: # only one rating "1Rating"
            recipe["reviews"] = int(re.sub("[\s+\n]","",soup.find("span",{"class":"ugc-ratings-item"}).text)[:-6])
        recipe["title"] = soup.find('h2',{"class":"recipe-title"}).text
        recipe['ratings'] = float(re.sub("[\s+\n]","",soup.find("span", {"class":"review-star-text"}).text)[7:8]) # make this an int
        recipe["calories"] = int(soup.find_all('div',{"class":"section-body"})[-1].text.split()[0]) # make this an int
        recipe["category"] = soup.find("div",{"class":"keyvals"}).attrs['data-content_cms_category'].split(",")
        # this format doesn't have time data
        recipe["total_mins"] = 0
    except: # skip with empty recipes
        recipe = {'recipe_id': id,
         'reviews': 0,
         'title': 'Recipe webpage is unconventional',
         'ratings': '0',
         'calories': '0',
         'category': 'n/a',
         'total_mins': '0',
         'ingredients': []}
    ingredients = []
    # only get checklist ingredients
    for i in soup.find_all("span",{"class":"ingredients-item-name"}):
        ingredients.append(re.sub('[\\n]\s*','',i.text))
        recipe["ingredients"] = ingredients
    return recipe

def get_all_recipes(ids):
    '''Return all the recipes from a list of ids'''
    df = pd.DataFrame(columns=['id', 'reviews', 'title', 'ratings', 'calories', 'category', 'total_mins', 'ingredients'])
    recipes = []
    index = 1
    for i in ids:
        start_time = time()
        sleep(random.randint(0,5))
        try: # html type 1
            recipes.append(get_recipe(i))
            print(get_recipe(i))
        except: # html type2
            recipes.append(get_recipe2(i))
            print(get_recipe2(i))
        try:
            print(f'Getting recipe {i}, {index}/{len(ids)}')
        except:
            pass
        index += 1
        elapsed_time = time() - start_time
    print(recipes)
    return pd.concat([pd.DataFrame([recipe]) for recipe in recipes])

###  get data, execute code ###
final_recipe_ids = all_recipe_ids()
pd.DataFrame(final_recipe_ids).to_csv("recipe_ids.csv", index=False)

# load recipes
final_recipe_ids = pd.read_csv("./data/recipe_ids.csv")["0"].tolist()

# export to CSV
all_recipes = get_all_recipes(final_recipe_ids)
all_recipes.to_csv("./data/recipes/all_recipes.csv",index=False)

# skipped recipes, re-scrap
skipped = [220180, 220184, 220991, 220994, 221080, 221211, 221227, 221269, 221272, 221952, 221955, 221959, 221968, 222192, 222194, 222206, 222309, 222313, 222334, 222339, 222340, 222387, 222392, 222393, 222396, 222402, 222760, 223051, 223151, 223198, 223362, 228371, 228374, 229449, 229450, 229453, 229728, 229729, 230566, 230567, 230746, 231006, 232474, 232750, 233789, 233998, 234297, 234425, 234463, 234466, 234535, 234930, 235274, 235276, 235277, 235358, 235718, 235798, 236032, 236219, 236226, 236504, 236700, 236803, 236807, 237725, 237727, 237930, 238036, 238131, 238259, 238263, 238311, 238313, 238384, 238402, 238587, 238918, 238921, 238924, 239232, 239433, 239542, 239543, 239783, 239785, 239870, 240239, 240457, 240523, 240607, 240703, 240751, 240755, 240790, 240933, 241311, 241556, 241738, 241739, 241874, 241898, 242360, 242464, 244200, 244203, 244455, 244472, 244554, 244601, 244686, 244811, 244913, 244914, 244944, 245029, 245363, 245440, 245441, 245724, 246127, 246529, 246679, 246681, 246718, 246737, 246931, 246974, 247092, 247093, 247095, 247364, 254274, 254277, 254453, 254575, 254945, 254946, 255013, 255019, 255021, 255281, 255304, 255588, 255590, 255814, 256099, 256436, 256886, 256887, 256889, 256968, 256996, 257073, 257426, 257491, 257918, 258369, 258748, 258804, 259042, 259483, 260539, 260733, 261198, 261587, 261588, 261792, 261801, 262160, 263453, 263484, 263622, 263749, 263750, 263829, 264043, 264058, 264183, 264184, 264290, 264606, 264951, 265113, 265538, 265562, 265861, 266045, 266334, 266825, 268350, 268587, 268594, 268856, 268858, 269177, 269181, 270102, 270299, 270538, 270676, 272184, 272197, 272494, 272778, 273054]

missing_recipes = get_all_recipes(skipped)
missing_recipes.to_csv("./data/recipes/missing_recipes.csv",index=False)

##### user reviews #####
# To search faster through large blocks of HTML Soups, set page size to max 25
# There are < 2500 reviews for all of Chef John's recipes -> 2500/25 = 100 max pages
def get_users(recipe_id):
    '''Returns users in a dictionary for a specific recipe
    get_users(270954) - 5 reviews
    get_users_fast(222002) - 1300+ reviews
    '''
    URL = f'https://www.allrecipes.com/recipe/getreviews/?recipeid={recipe_id}&recipeType=Recipe&sortBy=MostHelpful&pagesize=25'
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.131 Safari/537.36'}
    lst_of_users = []
    for page in range(1,100+1): # loop through all pages
        params = {'pagenumber': page}
        response = requests.get(URL,headers=headers, params=params)
        soup = BeautifulSoup(response.content,"lxml")
        num_users = len(soup.find_all("h4",{"itemprop":"author"})) # users on a page
        print(num_users)
        if num_users == 0: # if page is blank, break loop
            break
        index = 1
        for i in range(num_users): # loop through users on a page
            user = {}
            user["recipe_id"] = recipe_id
            user["user_id"] = soup.find_all("div",{"class":"recipe-details-cook-stats-container"})[i].find("a")["href"].split("/")[-2]
            user["username"] = re.sub(r'\\r\\n|\s\s','',soup.find_all("h4",{"itemprop":"author"})[i].text)
            user["rating"] = soup.find_all("div",{"class":"stars-and-date-container"})[i]["title"].split(" ")[2]
            user["date"] = soup.find_all("div",{"class":"review-date"})[i]["content"]
            user["review"] = re.sub(r'\\r\\n|\s\s','',soup.find_all("p",{"itemprop":"reviewBody"})[i].text)
            lst_of_users.append(user)
            print(f'Getting users: {index}, set {page}')
            index +=1
    return lst_of_users

# load recipes
final_recipe_ids = pd.read_csv("./data/recipe_ids.csv")["0"].tolist()

def get_all_users(final_recipe_ids):
    '''Return all the users as a dataframe
    get_all_users(final_recipe_ids[0:2])'''
    start_time = time()
    df = pd.DataFrame(columns=['date', 'rating', 'recipe_id', 'review', 'user_id', 'username'])
    index = 1
    users = []
    for i in final_recipe_ids:
        start_time = time()
        users.extend(get_users(i))
        print(f'Got users for recipe {index}/{len(final_recipe_ids)}')
        index += 1
        elapsed_time = time() - start_time
        print(elapsed_time)
        sleep(random.randint(0,3))
    # concat outside of the first loop function for better performance
    lst = []
    for user in users:
        lst.append(pd.DataFrame([user]))
    return pd.concat(lst)

# run in batches
# the first recipes with 1000+ reviews can be slow
batch1 = get_all_users(final_recipe_ids[0:5])
batch2 = get_all_users(final_recipe_ids[5:10])
batch3 = get_all_users(final_recipe_ids[10:15])
batch4 = get_all_users(final_recipe_ids[10:50])
batch5 = get_all_users(final_recipe_ids[50:100])
batch6 = get_all_users(final_recipe_ids[100:200])
batch7 = get_all_users(final_recipe_ids[200:500])
batch8 = get_all_users(final_recipe_ids[500:900])
batch9 = get_all_users(final_recipe_ids[900:1000])
batch10 = get_all_users(final_recipe_ids[1000:(1162+1)])

batch1.to_csv("./data/users/batch1.csv",index=False)
batch2.to_csv("./data/users/batch2.csv",index=False)
batch3.to_csv("./data/users/batch3.csv",index=False)
batch4.to_csv("./data/users/batch4.csv",index=False)
batch5.to_csv("./data/users/batch5.csv",index=False)
batch6.to_csv("./data/users/batch6.csv",index=False)
batch7.to_csv("./data/users/batch7.csv",index=False)
batch8.to_csv("./data/users/batch8.csv",index=False)
batch9.to_csv("./data/users/batch9.csv",index=False)
batch10.to_csv("./data/users/batch10.csv",index=False)


### rescrap for images links ###
recipe_ids = pd.read_csv("./data/recipe_ids.csv")["0"].tolist()

def get_image(id):
    '''Return image URL
    get_image(266825)
    >>> https://images.media-allrecipes.com/userphotos/560x315/5596236.jpg'''
    recipe = {}
    recipe["recipe_id"] = id
    URL = "https://www.allrecipes.com/recipe/"
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.131 Safari/537.36'}
    response = requests.get(URL+f'{id}',headers=headers)
    soup = BeautifulSoup(response.content,"lxml")
    photo_url = soup.find("img",{"class":"rec-photo"}).attrs["src"]
    recipe["photo_url"] = photo_url
    return recipe


def get_image2(id):
    '''Return image URL, slightly diff html
    get_image2(id=221269)
    ''https://images.media-allrecipes.com/userphotos/560x315/3536283.jpg'''
    recipe = {}
    recipe["recipe_id"] = id
    URL = "https://www.allrecipes.com/recipe/"
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.131 Safari/537.36'}
    response = requests.get(URL+f'{id}',headers=headers)
    soup = BeautifulSoup(response.content,"lxml")
    # append 560x315
    try:
        photo_url = re.sub('userphotos/','userphotos/560x315/', soup.find("a",{"class":"recipe-review-open-in-new"}).attrs["href"])
    except:
        try:
            photo_url = re.sub('userphotos/','userphotos/560x315/', soup.find("button",{"class":"icon icon-image-zoom"}).attrs["data-image"])
        except:
            print(f'Photo not found for {id}')
            return recipe
    recipe["photo_url"] = photo_url
    return recipe

def get_all_photos(ids):
    '''Return all the photo url from a list of ids'''
    df = pd.DataFrame(columns=['id', 'photo_url'])
    photos = []
    index = 1
    for i in ids:
        start_time = time()
        try: # html type 1
            sleep(3)
            image = get_image(i)
            photos.append(image)
            print(image)
        except: # html type2
            sleep(2)
            image = get_image2(i)
            photos.append(image)
            print(image)
        index += 1
        elapsed_time = time() - start_time
    print(photos)
    return pd.concat([pd.DataFrame([photo]) for photo in photos])

photo_100 = get_all_photos(recipe_ids[:100])
photo_100.to_csv("./data/photo_url/100.csv",index=False)
photo_200 = get_all_photos(recipe_ids[100:200])
photo_200.to_csv("./data/photo_url/200.csv",index=False)
photo_300 = get_all_photos(recipe_ids[200:300])
photo_300.to_csv("./data/photo_url/300.csv",index=False)
photo_400 = get_all_photos(recipe_ids[300:400])
photo_400.to_csv("./data/photo_url/400.csv",index=False)
photo_500 = get_all_photos(recipe_ids[400:500])
photo_500.to_csv("./data/photo_url/500.csv",index=False)
photo_600 = get_all_photos(recipe_ids[500:600])
photo_600.to_csv("./data/photo_url/600.csv",index=False)
photo_700 = get_all_photos(recipe_ids[600:700])
photo_700.to_csv("./data/photo_url/700.csv",index=False)
photo_800 = get_all_photos(recipe_ids[700:800])
photo_800.to_csv("./data/photo_url/800.csv",index=False)
photo_900 = get_all_photos(recipe_ids[800:900])
photo_900.to_csv("./data/photo_url/900.csv",index=False)
photo_1000 = get_all_photos(recipe_ids[900:1000])
photo_1000.to_csv("./data/photo_url/1000.csv",index=False)
photo_1100 = get_all_photos(recipe_ids[1000:1100])
photo_1100.to_csv("./data/photo_url/1100.csv",index=False)
photo_end = get_all_photos(recipe_ids[1100:])
photo_end.to_csv("./data/photo_url/1200.csv",index=False)

# missing
photo_missing = get_all_photos([220184, 221005, 221006, 221007, 221053, 221054, 221068, 221069, 221070, 221071, 221079, 221080, 221089, 221090, 221091, 221092, 221093, 222237, 223043, 223045, 223046, 223047, 223048, 223049, 223050, 223051, 223069, 223150, 223152, 223153, 223154, 229270, 229271, 229272, 229273, 229450, 231350, 231352, 231353, 231354, 233294, 233295, 233327, 233397, 233398, 233399, 235348, 235355, 235356, 235357, 235358, 235359, 235360, 235366, 235414, 235415, 236700, 237471, 237472, 237473, 237474, 237475, 237476, 237477, 237496, 237498, 237499, 239465, 239466, 239540, 239541, 241498, 241499, 241500, 241553, 241554, 241555, 244913, 245029, 245584, 245585, 245617, 245619, 254945, 255823, 255863, 255864, 255935, 255936, 256886, 256968, 257871, 257918, 257919, 257938, 260013, 260014, 260018, 261983, 262044, 264043, 264058, 264059, 266085, 266147, 268136, 268180]
photo_missing.to_csv("./data/photo_url/missing.csv",index=False)
