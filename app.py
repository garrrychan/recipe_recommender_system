import pandas as pd
import numpy as np
import random
from flask import request, Flask, render_template
from model import recommenders, utils

app = Flask(__name__)

@app.route('/', methods=['GET','POST'])
def quiz():
    # choose sample to show for quiz
    most_popular = recommenders.sample_popular()

    if request.method == 'POST':
        quiz_results = request.form.getlist("my_checkbox")

        sample = random.sample(quiz_results,2) # for two categories
        title = sample[0]

        print(f'quiz results: {quiz_results}')
        print(f'title: {title}')

        ### People with Similar Tastes Also Liked ###
        user_recommended = recommenders.quiz_user_user_recommender(utils.create_new_user(quiz_results))
        print(f'user_recommended: {user_recommended}')

        ### Because you liked X ###
        item_recommended = recommenders.item_item_recommender(title=title, new_user=utils.create_new_user(quiz_results))
        # from quiz results, get category, and randomly select 2 to return recipes in that category
        print(f'item_recommended: {item_recommended}')
        ### categories ###
        cat1 = utils.get_category(sample[0])
        cat1_recommended = [utils.recipe_id_to_title(recipe) for recipe in utils.similar_to_cat(cat1)]

        cat2 = utils.get_category(sample[1])
        cat2_recommended = [utils.recipe_id_to_title(recipe) for recipe in utils.similar_to_cat(cat2)]

        cats_recommended = list([cat1_recommended,cat2_recommended])
        print(f'cats_recommended: {cats_recommended}')

        ### tastebreaker ###
        tastebreaker = recommenders.item_item_recommender(title=title, new_user=utils.create_new_user(quiz_results), opposite=True)
        print(f'tastebreaker: {tastebreaker}')

        # svd_recommended = recommenders.svd_recommender(8888888, new_user=utils.create_new_user(quiz_results))
        # print(f'svd_recommended: {svd_recommended}')
        all = user_recommended + item_recommended
        # + svd_recommended

        all = set(all) # remove duplicates
        # remove recipes user has tried & sample 6
        hybrid_recommended = random.sample([x for x in all if x not in utils.known_positives(8888888,new_user=utils.create_new_user(quiz_results))],6)

        print(f'hybrid_recommended: {hybrid_recommended}')

        return render_template("result.html",
        title = title,

        cats = (list([cat1,cat2]), cats_recommended, [utils.get_url(utils.title_to_id(recipe)) for recipe in cats_recommended[0]], [utils.get_url(utils.title_to_id(recipe)) for recipe in cats_recommended[1]]),
        # tuple, second element is the image url
        most_popular=([utils.strip_filler(recipe) for recipe in most_popular],
        [utils.get_url(utils.title_to_id(recipe)) for recipe in most_popular]),

        quiz_results=([utils.strip_filler(recipe) for recipe in quiz_results],
        [utils.get_url(utils.title_to_id(recipe)) for recipe in quiz_results]),

        user_recommended=([utils.strip_filler(recipe) for recipe in user_recommended],
        [utils.get_url(utils.title_to_id(recipe)) for recipe in user_recommended]),

        item_recommended=([utils.strip_filler(recipe) for recipe in item_recommended],
        [utils.get_url(utils.title_to_id(recipe)) for recipe in item_recommended],utils.strip_filler(title))
        ,

        tastebreaker=([utils.strip_filler(recipe) for recipe in tastebreaker],
        [utils.get_url(utils.title_to_id(recipe)) for recipe in tastebreaker],utils.strip_filler(title)),

        hybrid_recommended = ([utils.strip_filler(recipe) for recipe in hybrid_recommended],
        [utils.get_url(utils.title_to_id(recipe)) for recipe in hybrid_recommended])
        )

    # landing screen
    return render_template("quiz.html",
    most_popular=(most_popular,[utils.get_url(utils.title_to_id(recipe)) for recipe in most_popular])
    )

if __name__ == '__main__':
    app.run(debug=True)
