# EDA
import pandas as pd
import numpy as np
from random import sample
from collections import Counter
import ast
import re
import random
import pickle

# modelling & evaluation
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import TruncatedSVD
from sklearn.utils.extmath import randomized_svd
from sklearn.metrics import roc_auc_score

# scientific notation off
np.set_printoptions(suppress=True)
pd.options.display.float_format = '{:.2f}'.format

# load data / global variables
all_users = pd.read_csv("./data/users/all_users.csv")
all_users.drop_duplicates(inplace=True)
all_recipes = pd.read_csv("./data/recipes/all_recipes.csv")
all_recipes.drop_duplicates(inplace=True)
photo_urls = pd.read_csv("./data/photo_url/photo_urls.csv")
photo_urls.drop_duplicates(inplace=True)
recipe_lookup = all_recipes[["recipe_id","title"]]

# Collaborative filtering for those with at least 3 reviews
# ratings_by_user = all_users.groupby(["user_id","username"])[["rating"]].count().sort_values("rating",ascending=False)
# at_least_3_ids = list(ratings_by_user[ratings_by_user["rating"]>=3].reset_index().user_id)
# users3 = all_users[all_users.user_id.isin(at_least_3_ids)][["user_id","recipe_id","rating"]]
# pickle.dump(users3, open("users3.pkl", "wb"))
users3 = pickle.load(open("./pickle/users3.pkl","rb"))

class utils:
    def __init__(self,all_recipes):
        pass

    def get_category(title):
        '''get_category("Chef John's Chicken Cacciatore")
        Return multiple categories to ensure it's not too niche'''
        df = all_recipes[["category","title"]]
        my_category = df[df.title == title].category
        categories = ast.literal_eval(my_category.values[0])
        return categories

    def recipe_id_to_title(recipe_id):
        ''' recipe_id_to_title(223042) >>> 'Chicken Parmesan'
        '''
        df = all_recipes[["recipe_id","title"]]
        my_title = df[df.recipe_id == recipe_id].title
        return my_title.values[0]

    def title_to_id(title):
        ''' title_to_id('Chicken Parmesan') >>> '223042'
        '''
        df = all_recipes[["recipe_id","title"]]
        my_recipe = df[df.title == title].recipe_id
        return my_recipe.values[0]

    def strip_filler(str):
        '''Remove filler words'''
        stop = ["chef", "john's"]
        words = [i for i in str.split() if i.lower() not in stop]
        return " ".join(words)

    def known_positives(user_id,threshold=4,new_user=None):
        '''Return known positives, by default no new_user input
        new_user is a dictionary

        {'user_id': [8888888], 'recipe_id': [219936], 'rating': [5]}'''

        users = all_users[["user_id","recipe_id","rating"]]
        users = pd.concat([users,pd.DataFrame(new_user)])

        user_preferences = pd.merge(users, recipe_lookup, on='recipe_id', how='left')
        known_positives = user_preferences[(user_preferences["user_id"] == user_id)&(user_preferences["rating"] >= threshold)]
        known_positives_list = list(known_positives.title)
        return known_positives_list

    def create_new_user(quiz_results):
        '''quiz_results = ['Spicy Chicken Thai Soup']
        create_new_user(quiz_results)'''

        input = [utils.title_to_id(recipe) for recipe in quiz_results]
        new_user_id = [8888888] * len(input)
        new_user_recipe_ids = input
        new_user_ratings = [5] * len(input)

        new_user = {'user_id': new_user_id,
        'recipe_id': new_user_recipe_ids,
        'rating': new_user_ratings}

        return new_user

    def count_categories(all_recipes_df):
        ''' Returns a string of all unique categories of recipes
        count_categories(all_recipes)
        '''
        all_recipes_df.dropna(axis=0,how='any',inplace=True)
        recipe_categories = all_recipes.drop(["title","category","ingredients"],axis=1)
        categories = []
        # ast.literal turns str rep of list into list
        # dropna otherwise we will experience errors with eval!
        for i in [ast.literal_eval(j) for j in all_recipes_df.category.dropna()] :
            categories.extend(i)
        categories = list(set(categories))
        for category in categories:
            recipe_categories[category] = all_recipes_df["category"].apply(lambda row: int(category in row))

        return recipe_categories.drop(["calories", "ratings", "reviews", "total_mins"],axis=1)

    def get_url(recipe_id):
        '''url(220854)'''
        try:
            url = photo_urls.query(f'recipe_id=={recipe_id}').photo_url.values[0]
        except: #image does not exist
            url = "https://png.pngtree.com/element_origin_min_pic/17/08/09/28d3afc4b9471eba6f908caf6943d473.jpg"
        return url

    def similar_to_cat(categories, top_N=10, all_recipes=all_recipes):
        '''Return a sample of 6 of the top_N new recipes most similar to the chosen
        categories

        similar_to_cat(['Main Dish', 'Chicken', 'Chicken Cacciatore'])

        '''
        sample_list = []
        matrix = utils.count_categories(all_recipes)
        for category in categories:
            try:
                recipes = list(matrix[matrix[category]==1].recipe_id)
                sample_list.extend(recipes)
            except:
                pass
        return random.sample(sample_list,6)

def create_X(df):
    """
    Generates a compressed sparse matrix "csm" dataframe.

    Args:
        df: pandas dataframe containing 3 columns (user_id, recipe_id, rating)

    Returns:
        X: sparse user-item matrix
        user_mapper: dict that maps user id's to user indices
        user_inv_mapper: dict that maps user indices to user id's
        recipe_mapper: dict that maps recipe id's to recipe indices
        recipe_inv_mapper: dict that maps recipe indices to movie id's

    X, user_mapper, recipe_mapper, user_inv_mapper, recipe_inv_mapper = create_X(users3)

    """
    M = df['user_id'].nunique()
    N = df['recipe_id'].nunique()

    user_mapper = dict(zip(np.unique(df["user_id"]), list(range(M))))
    recipe_mapper = dict(zip(np.unique(df["recipe_id"]), list(range(N))))

    user_inv_mapper = dict(zip(list(range(M)), np.unique(df["user_id"])))
    recipe_inv_mapper = dict(zip(list(range(N)), np.unique(df["recipe_id"])))

    user_index = [user_mapper[i] for i in df['user_id']]
    item_index = [recipe_mapper[i] for i in df['recipe_id']]

    X = csr_matrix((df["rating"], (user_index, item_index)), shape=(M,N))

    return X, user_mapper, recipe_mapper, user_inv_mapper, recipe_inv_mapper

X, user_mapper, recipe_mapper, user_inv_mapper, recipe_inv_mapper = create_X(users3)

# pre-calculate for speed for Heroku website

# 1
# pickle.dump(similarity_matrix, open("similarity_matrix.pkl", "wb"))
# similarity_matrix = cosine_similarity(X,X)

# 2
# recipe_categories = utils.count_categories(all_recipes).iloc[:,1:]
# A = csr_matrix(recipe_categories)
# del recipe_categories
# cosine_sim = cosine_similarity(A, A)
# pickle.dump(cosine_sim, open("cosine_sim.pkl","wb"))

class recommenders:
    def __init__(self):
        pass

    def sample_popular(n=24):
        '''Return a sample of 12 of top 1000/1000+ recipes'''
        df = all_users[["rating","recipe_id"]].groupby("recipe_id").count().sort_values(by="rating",ascending=False).reset_index()
        top_1000 = [utils.recipe_id_to_title(thing) for thing in df[0:500].recipe_id]
        return sample(top_1000,n)

    def user_user_recommender(top_N, user_id, threshold=4, X_sparse=X, user_mapper=user_mapper, recipe_lookup = recipe_lookup, all_users=all_users,new_user=None):
        '''Return a sample of 6 of top_N new recipes based on similar users

        X: sparse user-item utility matrix, not normalized

        recommenders.user_user_recommender(top_N=20, user_id=3936048)

        '''

        similarity_matrix = pickle.load(open("./pickle/similarity_matrix.pkl", "rb"))
        user = user_mapper[user_id]
        # negate for most similar
        similar_users = np.argsort(-similarity_matrix[user])[1:11] # remove original user, peak at top 10 similar users
        sorted(-similarity_matrix[user])[1:]
        recommended_recipes = []
        # returns enough recipes ~100, so good coverage
        # loop through all users to get top_N items, only if the recipes > threshold
        for i in similar_users:
            similar_user = (all_users[all_users["user_id"]==user_inv_mapper[i]])
            recommended_recipes.extend(list(similar_user[similar_user.rating>=threshold].recipe_id))

        picks = recommended_recipes[0:25]
        # convert recipe_id to title
        picks = [recipe_lookup.query(f'recipe_id=={i}').title.values[0] for i in picks]

        # remove already tried items
        new_picks = [pick for pick in picks if pick not in utils.known_positives(user_id,threshold,new_user)]

        # remove duplicates & sample 6
        return sample(set(new_picks),6)

    def quiz_user_user_recommender(new_user):
        '''
        Accept new user input and applies user_user_recommender function
        recommenders.quiz_user_user_recommender(utils.create_new_user(quiz_results))'''

        pd_new_user = pd.DataFrame(new_user)
        # concat new_user rows
        new_user_df = pd.concat([users3,pd_new_user])
        # create a X_new
        X_new, user_mapper_new, recipe_mapper_new, user_inv_mapper_new, recipe_inv_mapper_new = create_X(new_user_df)

        return recommenders.user_user_recommender(top_N=30, user_id=8888888, threshold=4, X_sparse=X_new,
        user_mapper=user_mapper_new, recipe_lookup = recipe_lookup, all_users=all_users,new_user=new_user)


    def item_item_recommender(title, top_N=10, opposite=False, threshold=4, all_recipes=all_recipes, new_user=None, user_id=8888888):
        '''Return a sample of 6 of top_N new recipes most similar to chosen recipe,
        by default top_N is 10, so items are very similar. Opposite=False by default.

        recommenders.item_item_recommender(title="Khachapuri (Georgian Cheese Bread)", opposite=True)
        recommenders.item_item_recommender(title="Khachapuri (Georgian Cheese Bread)", opposite=False)
        recommenders.item_item_recommender(title="Chef John's Italian Meatballs", new_user=utils.create_new_user(quiz_results))
        '''

        cosine_sim = pickle.load(open("./pickle/cosine_sim.pkl","rb"))

        recipe_idx = dict(zip(all_recipes['title'], list(all_recipes.index)))
        idx = recipe_idx[title]

        sim_scores = list(enumerate(cosine_sim[idx]))
        if opposite:
            sim_scores.sort(key=lambda x: x[1], reverse=False)
            sim_scores = sim_scores[1:(top_N+1)] # taking the first top_N makes it run a lot faster
            dissimilar_recipes_idx = [i[0] for i in sim_scores]
            picks = list(all_recipes['title'].iloc[dissimilar_recipes_idx])
            new_picks = [pick for pick in picks if pick not in utils.known_positives(user_id,threshold,new_user)]
            return sample(new_picks[0:100],6)

        else:
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            sim_scores = sim_scores[1:(top_N+1)]
            similar_recipes_idx = [i[0] for i in sim_scores]
            picks = list(all_recipes['title'].iloc[similar_recipes_idx])
            # filter out items chosen, by default filter out new user 8888888
            new_picks = [pick for pick in picks if pick not in utils.known_positives(user_id,threshold,new_user)]
            # choose the top 6 from ranked new_picks to display
            return sample(new_picks[0:10],6)

    def svd_recommender(user_id, new_user=None, threshold=3):
        '''
        Returns recommended sample of 6 of the top 10 recipes, over threshold rating for a particular user,
        using matrix factorization to find latent factors.

        Requires a lot vector to be filled out for non-zero results.

        ELI5: It makes a prediction of the rating (into lower dimensional space)

        top_N: number of similar recipe to retrieve, int | X_norm: user-item utility matrix |  user_id: original user_id, int
        user_mapper: user_mapper, df | user_preferences: df | threshold = rating threshold between 1 to 5 inclusive

        # existing user
        recommenders.svd_recommender(3936048)

        # new user
        recommenders.svd_recommender(8888888, new_user=new_user)
        '''
        # for new user predictions
        if new_user != None:
            pd_new_user = pd.DataFrame(new_user)
            # concat new_user rows
            new_user_df = pd.concat([users3,pd_new_user])
            # create a X_new
            X_new, user_mapper_new, recipe_mapper_new, user_inv_mapper_new, recipe_inv_mapper_new = create_X(new_user_df)

            X = X_new
            user_mapper = user_mapper_new
            recipe_inv_mapper = recipe_inv_mapper_new
        else:
            X, user_mapper, recipe_mapper, user_inv_mapper, recipe_inv_mapper = create_X(users3)
            X = X
            user_mapper = user_mapper
            recipe_inv_mapper = recipe_inv_mapper

        # calculate mean_rating_per_recipe
        sum_ratings_per_recipe = X.sum(axis=0)
        n_ratings_per_recipe = X.getnnz(axis=0)
        mean_rating_per_recipe = sum_ratings_per_recipe/n_ratings_per_recipe
        X_mean_recipe = np.tile(mean_rating_per_recipe, (X.shape[0],1))

        user = user_mapper[user_id]
        # remove item bias
        X_norm = X - csr_matrix(X_mean_recipe)

        svd = TruncatedSVD(n_components=100,random_state=42)
        Z = svd.fit_transform(X_norm)
        X_new = svd.inverse_transform(Z)
        # add back the mean so it's interpretable
        X_new = X_new + np.array(X_mean_recipe)

        top_N_indices = X_new[user].argsort()[::-1]

        # check that it's greater than a threshold
        mask = X_new[user][top_N_indices]>threshold
        num_greater_than = len(X_new[user][top_N_indices][mask])
        # indices in order
        top_N_indices = top_N_indices[:num_greater_than]
        recommended = []
        for i in top_N_indices:
            recipe_id = recipe_inv_mapper[i]
            recommended.append((recipe_id, utils.recipe_id_to_title(recipe_id)))

        picks = [recipe[1] for recipe in recommended[0:10]]

        # there may not always be more than 6 things > 4 rating
        if len(picks) >= 6:
            return sample(picks,6)
        else:
            return picks

### evaluation ###
sum_ratings_per_recipe = X.sum(axis=0)
n_ratings_per_recipe = X.getnnz(axis=0)
mean_rating_per_recipe = sum_ratings_per_recipe/n_ratings_per_recipe
X_mean_recipe = np.tile(mean_rating_per_recipe, (X.shape[0],1))
X_norm = X - csr_matrix(X_mean_recipe)

X_train_norm = X_norm.todense()[0:2400]
X_test_norm = X_norm.todense()[2400:]

### Naive model ###
naive_preds = np.tile(0,(3211,1053))
# print(f'RMSE Train for naive model: {np.sqrt(((np.array(X_train_norm)-naive_preds[0:2400])**2).mean())}')
# print(f'RMSE Test for naive model: {np.sqrt(((np.array(X_test_norm)-naive_preds[2400:])**2).mean())}')

### Matrix Factorization Metric Calculation ###

svd = TruncatedSVD(n_components=100, random_state=42)
Z_train = svd.fit_transform(X_train_norm)
X_train_new = svd.inverse_transform(Z_train)
svd.explained_variance_ratio_.sum()

# Sklearn Truncated SVD uses randomized_svd
U, Sigma, VT = randomized_svd(X_norm,
                              n_components=100,
                              n_iter=10,
                              random_state=None)

def rank_k(k):
    U_reduced = np.mat(U[:,:k])
    VT_reduced = np.mat(VT[:k,:])
    Sigma_reduced = Sigma_reduced = np.eye(k)*Sigma[:k]
    Sigma_sqrt = np.sqrt(Sigma_reduced)
    return U_reduced*Sigma_sqrt, Sigma_reduced, Sigma_sqrt*VT_reduced

U_reduced, Sigma_reduced, VT_reduced = rank_k(30)

U_new = np.dot(X_test_norm, np.dot(VT_reduced.T,np.linalg.inv(Sigma_reduced)))
M_hat = np.dot(U_new,VT_reduced)

#print(f'RMSE Train for SVD: {np.sqrt(((np.array(X_train_norm)-X_train_new)**2).mean())}')
#print(f'RMSE Test for SVD: {np.sqrt(((np.array(X_test_norm)-np.array(M_hat))**2).mean())}')

### Ranking Metrics ###
def precision_and_recall_at_k(predictions, targets, k=6):
    '''Returns a tuple of (precision, recall) of the top k items

    precision = TP/(TP+FP) -> what fraction of recommended items did the user consume?
    recall = TP/(TP+FN) -> What out of all the items the user consumed, was recommended?

    precision_and_recall_at_k(list(all_recipes.title.sample(10)), utils.known_positives(3936048),k=6)
    precision_and_recall_at_k(recommenders.svd_recommender(3936048), utils.known_positives(3936048),k=6)
    precision_and_recall_at_k(recommenders.user_user_recommender(20,3936048), utils.known_positives(3936048),k=6)
    precision_and_recall_at_k(recommenders.item_item_recommender("Chef John's Italian Meatballs"), utils.known_positives(3936048),k=6)

    '''
    predictions = predictions[:k]
    num_hit = len(set(predictions).intersection(set(targets)))
    try:
        return float(num_hit) / len(predictions), float(num_hit) / len(targets)
    except:
        return "Predictions must be greater than length 1"
