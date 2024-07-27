import pandas as pd
import numpy as np
import tf_funct

ratings = pd.read_csv("data/ratings.csv")
movies = pd.read_csv("data/movies.csv")
movieList_df = ratings.groupby('movieId')['rating'].agg(['count', 'mean'])
movieList_df.columns = ['Rating Count', 'Mean Rating']
movieList_df = movieList_df.merge(movies[['movieId', 'title']], on='movieId')

R, Y, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper = tf_funct.create_matrices(ratings)

num_movies, num_users = Y.shape

my_ratings, my_rated = tf_funct.add_ratings(num_movies, [], [])
Y = np.c_[my_ratings, Y]
R = np.c_[(my_ratings != 0).astype(int), R]

num_movies, num_users = Y.shape

Ynorm, Ymean = tf_funct.normalizeRatings(Y, R)

X, W, b = tf_funct.train(Ynorm, R, num_users, num_movies, 100, iterations = 200)

tf_funct.predict(X, W, b, Ymean, my_rated, my_ratings, movieList_df)