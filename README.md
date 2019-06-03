# recipe_recommender_system
Driven by my curiousity of how Netflix, Youtube and Spotify serve personalized recommendations, I decided to learn how to create my own recommender system.



*Machine Learning Problem*: Given a person’s preferences in past recipes, could I predict other new recipes they might enjoy?



I created Seasonings, a Recipe Recommender System. The motivation behind this app is to help users discover personalized and new recipes, and prepare for grocery runs! I received a lot early positive feedback and plan future improvements to the UX and model.



I had a lot of fun making this, and plan to use this whenever I need a jolt of inspiration in the kitchen!



# Data 

Data was scraped from AllRecipes.com (Scraped with BeautifulSoup), as there was no public API. I narrowed the scope to focus on Chef John's recipes. 

- Content Data
  - ```all_recipes.csv```
  - 1100+ Recipes from 
  - 460+ Cuisines & Categories
- User Data
  - ```all_users.csv```
  - 55K Users
  - 73K Ratings



# Tech Stack

1. *Web Scraping*: BeautifulSoup, requests

- Please refer to ```web_scraper.py``` for more details



2. *Model*: scikit-learn, scipy, numpy

- See ```requirements.txt```



*3. Web Framework*: Flask

- Run ```app.py``` on [localhost:5000](localhost:5000/) ```



*4. Front End*: HTML & CSS



*5. Cloud Platform*: Heroku



# Models

Please refer to ```model.py``` 

1. Cosine Similarity
2. Content Based Filtering
3. Matrix Factorization



<img src="static/images/hybrid.png">

# Model Evaluation

<img src="static/images/evaluation.png">

My final model was a hybrid recommender that tackled the cold-start problem with a content recommender, augmented with user preferences, and factorization to rank recipes based on a voting classifier rule.



# Screenshots

#### Onboarding

<img src="static/images/onboard_1.png">

<img src="static/images/onboard_2.png">

#### Results (Hybrid, Collaborative Filtering & Content Filtering)

<img src="static/images/results_1.png">

<img src="static/images/results_2.png">

<img src="static/images/results_3.png">



# References

Special thanks to Kim Falk's book and also Maciej's GitHub for reference during this journey.

1. https://www.manning.com/books/practical-recommender-systems

2. https://github.com/lyst/lightfm

3. https://github.com/maciejkula/spotlight