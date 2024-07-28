import os
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Loading Data
ratings = pd.read_csv("data/ratings.csv")
movies = pd.read_csv("data/movies.csv")

# Computing Bayesian Average
movie_stats = ratings.groupby('movieId')['rating'].agg(['count', 'mean'])
C = movie_stats['count'].mean()
m = movie_stats['mean'].mean()

def bayesian_avg(ratings):
    bayesian_avg = (C * m + ratings.sum())/(C + ratings.count())
    return round(bayesian_avg, 3)

bayesian_avg_ratings = ratings.groupby('movieId')['rating'].agg(bayesian_avg).reset_index()
bayesian_avg_ratings.columns = ['movieId', 'bayesian_avg']
movie_stats = movie_stats.merge(bayesian_avg_ratings, on ='movieId')
movie_stats = movie_stats.merge(movies[['movieId', 'title']])

# Create X
def create_X(df):
    n_user = df['userId'].nunique()
    n_movie = df['movieId'].nunique()

    user_mapper = dict(zip(np.unique(df["userId"]), list(range(n_user))))
    movie_mapper = dict(zip(np.unique(df["movieId"]), list(range(n_movie))))

    user_inv_mapper = dict(zip(list(range(n_user)), np.unique(df["userId"])))
    movie_inv_mapper = dict(zip(list(range(n_movie)), np.unique(df["movieId"])))

    user_index = [user_mapper[i] for i in df['userId']]
    item_index = [movie_mapper[i] for i in df['movieId']]

    X = csr_matrix((df["rating"], (user_index, item_index)), shape=(n_user, n_movie))
    return X, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper

X, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper = create_X(ratings)

# Collaborative filtering using KNN
def find_similar_movies(movie_id, k=10, X=X, movie_mapper=movie_mapper, movie_inv_mapper=movie_inv_mapper, metric='cosine'):
    X = X.T
    neighbour_ids = []

    movie_ind = movie_mapper[movie_id]
    movie_vec = X[movie_ind]
    if isinstance(movie_vec, (np.ndarray)):
        movie_vec = movie_vec.reshape(1, -1)
    kNN = NearestNeighbors(n_neighbord=k+1, algorithm="brute", metric=metric)
    kNN.fit(X)
    neighbour = kNN.kneighbors(movie_vec, return_distance = False)
    for i in range (0, k):
        n=neighbour.item(i)
        neighbour_ids.append(movie_inv_mapper[n])
    neighbour_ids.pop(0)
    return neighbour_ids


# Content-based filtering
movie_titles = dict(zip(movies['movieId'], movies['title']))

genres = set(g for G in movies['genres'] for g in G)
for g in genres:
    movies[g] = movies.genres.transform(lambda x: int(g in x))
movie_genres = movies.drop(columns=['movieId', 'title', 'genres'])

cosine_sim = cosine_similarity(movie_genres, movie_genres)

def movie_finder(title):
    all_titles = movies['title'].tolist()
    closest_match = process.extractOne(title, all_titles)
    return closest_match[0]

movie_idx = dict(zip(movies['title'], list(movies.index)))

def get_content_based_recommendations(title_string, n_recommendations=10):
    title = movie_finder(title_string)
    idx = movie_idx[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores_df = pd.DataFrame(sim_scores, columns = ['movieId', 'sim_score'])
    sim_scores_df = sim_scores_df.sort_values(by=["sim_score"], ascending=False)
    sim_scores_mId = sim_scores_df["movieId"].tolist()
    similar_movies = []
    for i in sim_scores_mId[0:n_recommendations+1]:
        print(movies['title'].iloc[i])
        if (movies['title'].iloc[i] != title):
            similar_movies.append(i)
    return movies['title'].iloc[similar_movies], title
