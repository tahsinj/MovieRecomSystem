import os
import numpy as np
import tensorflow as tf
from tensorflow import keras

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Create Y
def create_matrices(df):
    
    user_ids = df['userId'].unique()
    movie_ids = df['movieId'].unique()

    n_user = len(user_ids)
    n_movie = len(movie_ids)
    
    user_mapper = dict(zip(np.unique(df["userId"]), list(range(n_user))))
    movie_mapper = dict(zip(np.unique(df["movieId"]), list(range(n_movie))))

    user_inv_mapper = dict(zip(list(range(n_user)), np.unique(df["userId"])))
    movie_inv_mapper = dict(zip(list(range(n_movie)), np.unique(df["movieId"])))

    R = np.zeros((len(movie_ids), len(user_ids)), dtype=int)
    Y = np.zeros((len(movie_ids), len(user_ids)))

    for row in df.itertuples():
        user_index = user_mapper[row.userId]
        movie_index = movie_mapper[row.movieId]
        R[movie_index, user_index] = 1
        Y[movie_index, user_index] = row.rating
    
    return R, Y, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper

# Mean Normalization
def normalizeRatings(Y, R):
    Ymean = np.sum(Y, axis=0) / np.sum(R, axis=0)
    Ynorm = Y - Ymean
    Ynorm[R == 0] = 0

    return Ynorm, Ymean

# Cost Function J for collaborative filtering with vectorization and regularization parameter lambda
def cofi_cost_func_v(X, W, b, Y, R, lambda_):
    j = (tf.linalg.matmul(X, tf.transpose(W)) + b - Y) * R
    J = 0.5 * tf.reduce_sum(j**2) + (lambda_/2) * (tf.reduce_sum(X**2) + tf.reduce_sum(W**2))

    return J

# Train values of W, X and b
def train(Ynorm, R, num_users, num_movies, num_features, lambda_ = 1, iterations = 200):
    tf.random.set_seed(1234)
    W = tf.Variable(tf.random.normal((num_users,  num_features),dtype=tf.float64),  name='W')
    X = tf.Variable(tf.random.normal((num_movies, num_features),dtype=tf.float64),  name='X')
    b = tf.Variable(tf.random.normal((1, num_users),   dtype=tf.float64),  name='b')

    optimizer = keras.optimizers.Adam(learning_rate=1e-1)

    for iter in range(iterations):
        with tf.GradientTape() as tape:
            cost_value = cofi_cost_func_v(X, W, b, Ynorm, R, lambda_)

        grads = tape.gradient(cost_value, [X,W,b] )

        optimizer.apply_gradients(zip(grads, [X,W,b]) )

        if iter % 20 == 0:
            print(f"Training loss at iteration {iter}: {cost_value:0.1f}")
    
    return X, W, b

def add_ratings(num_movies, idxToRate, ratingGiven):
    my_ratings = np.zeros(num_movies)
    for i in range(len(idxToRate)):
        my_ratings[idxToRate[i]] = ratingGiven[i]
    my_rated = [i for i in range(len(my_ratings)) if my_ratings[i] > 0]
    return my_ratings, my_rated
    

# Prediction
def predict(X, W, b, Ymean):
    p = np.matmul(X.numpy(), np.transpose(W.numpy())) + b.numpy()

    pm = p + Ymean

    my_predictions = pm[:, 0]

    ix = tf.argsort(my_predictions, direction='DESCENDING')

    return my_predictions, ix

def filtered_output(my_ratings, my_rated, my_predictions, ix):
    filter=(movieList_df["Rating Count"] > 20)
    movieList_df["pred"] = my_predictions
    movieList_df = movieList_df.reindex(columns=["pred", "Mean Rating", "Rating Count", "title"])
    movieList_df.loc[ix[:300]].loc[filter].sort_values("Mean Rating", ascending=False)

    for i in range(17):
        j = int(ix[i])
        if j not in my_rated:
            print(f'Predicting rating {my_predictions[j]:0.2f} for movie {movieList_df.iloc[j,3]}')

    print('\n\nOriginal vs Predicted ratings:\n')
    for i in range(len(my_ratings)):
        if my_ratings[i] > 0:
            print(f'Original {my_ratings[i]}, Predicted {my_predictions[i]:0.2f} for {movieList_df.iloc[i,3]}')
    
    