import numpy as np
import pandas as pd
import math
import time
import gc
import argparse
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import correlation
# utils import
from fuzzywuzzy import fuzz
# for drawing the histograms
import matplotlib.pyplot as plt
import seaborn as sns
# for creating sparse matrices
import scipy.sparse as sparse
from scipy.sparse import csr_matrix

#------------------------------------------------------------------
# Importing the data
#------------------------------------------------------------------
ratings_data = pd.read_csv("ratings.csv")
ratings_data_org = ratings_data
movies_data = pd.read_csv("movies.csv")
df_movies = movies_data[['movieId', 'title']]


x = [1,3]
y = [2,5]
euclidean_distance = math.sqrt((x[0]-y[0])**2 + (x[1]-y[1])**2)

#number of unique users and items
n_users = ratings_data.userId.unique().shape[0]
n_items = ratings_data.movieId.unique().shape[0]
print("Number of unique users: " + str(n_users))
print("Number of unique movies: " +str(n_items))


#------------------------------------------------------------------------------------
# Data Preprocessing for descriptive Statistics
#------------------------------------------------------------------------------------

# combining rating data with movie data
combined_data = pd.merge(ratings_data, movies_data, on='movieId')
combined_data.head()
# unsorted average ratings per movie
combined_data.groupby('userId')['rating'].mean().head()
# sorted average ratings per movie
combined_data.groupby('title')['rating'].mean().sort_values(ascending=False).head()
# sorted average ratings per movie with number of ratings (movies that are both often and highly rated)
combined_data.groupby('title')['rating'].count().sort_values(ascending=False).head()
# column of average rating of each movie
ratings_avg_count = pd.DataFrame(combined_data.groupby('title')['rating'].mean())
# column of number of ratings of that movie
ratings_avg_count['ratings_counts'] = pd.DataFrame(combined_data.groupby('title')['rating'].count())

# percentage of movies with less than 10 ratings
x = ratings_avg_count['ratings_counts']
count = 0
for i in range(len(x)):
    if x[i] <= 10:
        count += 1
percrat10 = count * 100 / len(x)
print(percrat10)

for i in range(1,101):
    if(i%3==0 and i%5==0):
        print("FuzzBuzz")
    elif(i%3==0):
        print("Fuzz")
    elif (i % 5 == 0):
        print("Buzz")
    else:
        print(i)

num_primes = 1
for i in range(2,1000):
    count = 0
    for j in range(2,i):
        if(i%j==0):
          count+=1
    if(count==0):
        print(str(i) + "is prime number")
        num_primes += 1

df_ratings_count_temp = pd.DataFrame(ratings_data.groupby('rating').size(), columns=['count'])
total_count = n_users * n_items
rating_zero_count = total_count - ratings_data.shape[0]
# including 0 ratings
df_ratings_count = df_ratings_count_temp.append(pd.DataFrame({'count': rating_zero_count}, index=[0.0]),verify_integrity=True,).sort_index()
# adding log count to make sure 0's are also included
df_ratings_count['log_count'] = np.log(df_ratings_count['count'])
# get rating frequency
df_movies_count = pd.DataFrame(ratings_data.groupby('movieId').size(), columns=['count'])


# pop_threshold should be smaller than act_threshold
# filtering the data
N = 50
pop_movies = list(set(df_movies_count.query('count >= @N').index))
ratings_data = ratings_data[ratings_data.movieId.isin(pop_movies)]

# get number of ratings given by every user
df_users_count = pd.DataFrame(ratings_data.groupby('userId').size(), columns=['count'])
M = 50
active_users = list(set(df_users_count.query('count >= @M').index))
ratings_data = ratings_data[ratings_data.userId.isin(active_users)]
print('Original ratings data: ', ratings_data_org.shape)
print('Ratings data after excluding both unpopular movies and inactive users: ', ratings_data.shape)

#updating row indices to avoid errors in data splitting process (where last row index = (shape of the data-1))
ratings_data = ratings_data.reset_index(drop=True)

#------------------------------------------------------------------------------------
# Plotting the figures
#------------------------------------------------------------------------------------
# Number of ratings counts
plt.figure(figsize=(8,5))
plt.rcParams['patch.force_edgecolor'] = True
ratings_avg_count['ratings_counts'].hist(bins=100,color = 'orange')
plt.xlabel('Number of Ratings', fontweight='bold')
plt.ylabel('Counts', fontweight='bold')
plt.xlim([0,2000])

# Histogram of movie ratings
plt.figure(figsize=(8,5))
plt.rcParams['patch.force_edgecolor'] = True
ratings_avg_count['rating'].hist(bins=50, color = 'orange')
plt.xlabel('Movie Ratings', fontweight='bold')
plt.ylabel('Counts', fontweight='bold')

# Scatter plot of ratings and counts corresponding to those ratings
plt.figure(figsize=(8,10))
plt.rcParams['patch.force_edgecolor'] = True
figure = sns.jointplot(x='rating', y='ratings_counts', data=ratings_avg_count, kind="scatter", color = 'orange')
plt.subplots_adjust(bottom=0.15, top=0.99, left=0.15)
figure.set_axis_labels("Ratings", "Counts", fontweight='bold')

ax = df_movies_count[['count']].reset_index().rename(columns={'index': 'rating score'}).plot(
    x='rating score',y='count', kind='bar', figsize=(8, 5), logy=True, fontsize=12,color = 'orange',)
ax.set_xlabel("Movie rating", fontweight='bold')
ax.set_ylabel("Number of ratings in log scale",fontweight='bold')

# rating frequency of all movies
ax = df_movies_count \
    .sort_values('count', ascending=False) \
    .reset_index(drop=True) \
    .plot(figsize=(8, 5),fontsize=14,color = 'orange',)
ax.set_xlabel("Movies", fontweight='bold')
ax.set_ylabel("Number of ratings", fontweight='bold')

# plot rating frequency of all movies in log scale
ax = df_movies_count \
    .sort_values('count', ascending=False) \
    .reset_index(drop=True) \
    .plot(figsize=(8, 5),fontsize=12,logy=True,color = 'Orange')
ax.set_xlabel("Movies", fontweight='bold')
ax.set_ylabel("Number of ratings (log scale)", fontweight='bold')

# plot rating frequency of all movies
ax = df_movies_count \
    .sort_values('count', ascending=False) \
    .reset_index(drop=True) \
    .plot(figsize=(8, 5),fontsize=14,color = 'orange')
ax.set_xlabel("Users", fontweight='bold')
ax.set_ylabel("Number of ratings", fontweight='bold')


#------------------------------------------------------------------------------------
#Function for transforming data to a sparse movie-user matrix
#------------------------------------------------------------------------------------
# creating User/Item/Rating sparse matrices matrices
def sparsematrix(data):
    # Getting the rating matrix
    n_users = data.userId.unique().shape[0]
    n_items = data.movieId.unique().shape[0]
    users_locations = data.groupby(by=['movieId', 'userId', 'rating']).apply(lambda x: 1).to_dict()
    row, col, value = zip(*(users_locations.keys()))
    map_u = dict(zip(data['movieId'].unique(), range(n_items)))
    map_l = dict(zip(data['userId'].unique(), range(n_users)))
    row_idx = [map_u[u] for u in row]
    col_idx = [map_l[l] for l in col]
    datar = np.array(value)
    sparse_csr = csr_matrix((datar, (row_idx, col_idx)), shape=(n_items, n_users))
    # coo_format_sparse = sparse_csr.tocoo([False])
    # csc_format_sparse = sparse_csr.tocsc([False])
    return(sparse_csr, data)

#------------------------------------------------------------------------------------
#Function for splitting the data
#------------------------------------------------------------------------------------
def get_data(data,test_size):
    unique_users = sorted(pd.unique(data['userId']))
    unique_items = sorted(pd.unique(data['movieId']))
    n_ratings = data.shape[0]
    test = []
    for userid in unique_users:
        # getting this users rating data
        dat = data[data['userId'] == userid]
        # sorting this users data based on time
        dat = dat.sort_values(['timestamp'], ascending=True)
        # test_size*100% of this users ratings
        num = int(dat.shape[0]*test_size)
        # selecting last num% of all ratings of this user
        indext = np.array(dat[-num:].index)
        test.append(indext)
    # (combining  indices of all users) list containing the test element indices
    test = np.concatenate(test, axis=None)
    test_items = np.zeros(n_ratings, dtype=bool)
    test_items[test] = True
    test_df = data[['userId', 'movieId', 'rating']][test_items]
    train_df = data[['userId', 'movieId', 'rating']][~test_items]

    # determining the sparse user-item rating matrices in three formats using above datasets
    train_sparse_csr = sparsematrix(train_df)[0]
    test_sparse_csr = sparsematrix(test_df)[0]

    print("Size of the training set")
    print(len(train_df) / (len(train_df)+len(test_df)))
    print("Size of the test set")
    print(len(test_df) / (len(train_df)+len(test_df)))

    return(train_sparse_csr, test_sparse_csr, train_df, test_df)

#------------------------------------------------------------------------------------
# Getting the sets and sparse matrices
#------------------------------------------------------------------------------------
data_object = get_data(ratings_data,0.3)
train_sparse_csr = data_object[0]
test_sparse_csr = data_object[1]
train_df = data_object[2]
test_df = data_object[3]
movie_user_matrix_train = train_df.pivot(index='movieId', columns='userId', values='rating').fillna(0)
movie_user_matrix_test = test_df.pivot(index='movieId', columns='userId', values='rating').fillna(0)


#------------------------------------------------------------------------------------
# Function matching the movies to the titles (title to lowercase)
#------------------------------------------------------------------------------------
def fuzzy_mapper(movie_mapper, favorite_movie, bool = True):
    matches = []
    for title, index in movie_mapper.items():
        ratio = fuzz.ratio(title.lower(), favorite_movie.lower())
        if ratio>= 50:
            matches.append((title, index, ratio))
    # sorting the matches
    matches = sorted(matches, key = lambda x:x[2])[::-1]
    if not matches:
        print('There are no matches found')
        return
    if bool:
        print('Possible matches: {0}\n'.format([x[0] for x in matches]))
    return matches[0][1]
#------------------------------------------------------------------------------------
# Function making similar movie recommendation
#------------------------------------------------------------------------------------
def movie_RS(train_data, movie_mapper, favorite_movie, num_recom, distancesb = False):

    # defining the model: similarity, top N movies (n_jobs = -1: using all processors)
    model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=num_recom, n_jobs=-1)
    # fitting the model using sparse matrix based on the training data
    model_knn.fit(train_data)
    print('The name of the inserted (input) movie:', favorite_movie)
    # to transform the input movie to index
    index = fuzzy_mapper(movie_mapper, favorite_movie, bool=True)
    # because the first returned neighbor is always the target point itself, we add 1 to num_recom(neigbors)
    distances, indices = model_knn.kneighbors(train_data[index], num_recom + 1)
    rec_movies_indecies = \
        sorted(list(zip(indices.squeeze().tolist(), distances.squeeze().tolist())), key=lambda x: x[1])[:0:-1]
    # transforming back to the movie names
    backward_mapper = {v: k for k, v in movie_mapper.items()}
    print('Recommendations for {}:'.format(favorite_movie))
    print('-----------------------------------------------')
    #list of movie indicies recommended for the target movie id
    rec_movie_ind = []
    if distancesb:
       for i, (index, dist) in enumerate(rec_movies_indecies):
            rec_movie_ind.append(index)
            print('{0}: {1}, with distances of {2}'.format(i + 1, backward_mapper[index], dist))
    else:
        for i, (index, dist) in enumerate(rec_movies_indecies):
            print('{0}: {1}'.format(i + 1, backward_mapper[index], dist))
            rec_movie_ind.append(index)

    # testing
    precisions = []
    n = len(test_df.userId.unique())
    for userid in test_df.userId.unique():
        movieids_peruser = []
        hits = 0
        for movieid in test_df[test_df['userId'] == userid]['movieId']:
            movieids_peruser.append(movieid)
        # is there a match between these movies and the recommended movies
        for elem in movieids_peruser:
            for recom in rec_movie_ind:
                if elem == recom:
                  # number of movies that are both in the recommended and test set for this user
                  hits+=1
        precision = hits/num_recom
        precisions.append(precision)
    print('The model precision with top {0} movie recommendations is: {1}'.format(num_recom,round(sum(precisions)/n,3)))


#------------------------------------------------------------------------------------
# The Experiment
#------------------------------------------------------------------------------------
# mapper between movie title to index using the movie data
movie_to_index = {
    movie: i for i, movie in
    enumerate(list(df_movies.set_index('movieId').loc[movie_user_matrix_train.index].title))
}

favorite_movies = ['Babysitter', 'Babe','Copycat', '(500) Days of Summer ', '¡Three Amigos!', 'Things I Hate About You']
N = [3, 5, 10, 20, 30, 50]
for fav_movie in favorite_movies:
    for num_rec in N:
        print(fav_movie)
        print(num_rec)
        print(movie_RS(train_data = train_sparse_csr,favorite_movie = fav_movie, movie_mapper = movie_to_index, num_recom = num_rec, distancesb = False))


#------------------------------------------------------------------------------------
# Ploting the precision graphs per movie
#------------------------------------------------------------------------------------
# Movie 1: Babysitter
# N = 3 0.017
# N = 5 0.028
# N = 10 0.020
# N = 20 0.014
# N = 30 0.012
# N = 50 0.012

bars = [0.017, 0.028, 0.02, 0.014, 0.012, 0.012]
# r1 = [1, 2, 3, 4, 5]
barWidth = 0.4
r1 = np.arange(len(bars))
r2 = [x + barWidth for x in r1]
plt.bar(r2, bars, color='black', width=barWidth, edgecolor='white')
plt.xlabel('Top-N Recommendation', fontweight='bold')
plt.ylabel('Precision', fontweight='bold')
plt.title('Babysitter', fontweight='bold')
plt.xticks([r + barWidth for r in range(len(bars))],['N 3', 'N 5', 'N 10', 'N 20', 'N 30','N 50'], rotation='vertical')
plt.subplots_adjust(bottom=0.23, top=0.92, left=0.15)
plt.legend()
plt.show()

# Movie 2: Babe
# N = 3 0.019
# N = 5 0.017
# N = 10 0.013
# N = 20 0.013
# N = 30 0.014
# N = 50 0.013

bars = [0.019, 0.017, 0.013, 0.013, 0.014, 0.013]
# r1 = [1, 2, 3, 4, 5]
barWidth = 0.4
r1 = np.arange(len(bars))
r2 = [x + barWidth for x in r1]
plt.bar(r2, bars, color='black', width=barWidth, edgecolor='white')
plt.xlabel('Top-N Recommendation', fontweight='bold')
plt.ylabel('Precision', fontweight='bold')
plt.title('Babe', fontweight='bold')
plt.xticks([r + barWidth for r in range(len(bars))],['N 3', 'N 5', 'N 10', 'N 20', 'N 30','N 50'], rotation='vertical')
plt.subplots_adjust(bottom=0.23, top=0.92, left=0.15)
plt.legend()
plt.show()


# Movie 3: Copycat
# N = 3 0.034
# N = 5 0.030
# N = 10 0.019
# N = 20 0.018
# N = 30 0.018
# N = 50 0.017

bars = [0.034, 0.030, 0.019, 0.018, 0.018, 0.017]
# r1 = [1, 2, 3, 4, 5]
barWidth = 0.4
r1 = np.arange(len(bars))
r2 = [x + barWidth for x in r1]
plt.bar(r2, bars, color='black', width=barWidth, edgecolor='white')
plt.xlabel('Top-N Recommendation', fontweight='bold')
plt.ylabel('Precision', fontweight='bold')
plt.title('Copycat', fontweight='bold')
plt.xticks([r + barWidth for r in range(len(bars))],['N 3', 'N 5', 'N 10', 'N 20', 'N 30','N 50'], rotation='vertical')
plt.subplots_adjust(bottom=0.23, top=0.92, left=0.15)
plt.legend()
plt.show()

# Movie 4: (500) Days of Summer
# N = 3 0.035
# N = 5 0.021
# N = 10 0.023
# N = 20 0.016
# N = 30 0.015
# N = 50 0.015
bars = [0.035, 0.021, 0.023, 0.016, 0.015, 0.015]
# r1 = [1, 2, 3, 4, 5]
barWidth = 0.4
r1 = np.arange(len(bars))
r2 = [x + barWidth for x in r1]
plt.bar(r2, bars, color='black', width=barWidth, edgecolor='white')
plt.xlabel('Top-N Recommendation', fontweight='bold')
plt.ylabel('Precision', fontweight='bold')
plt.title('(500) Days of Summer', fontweight='bold')
plt.xticks([r + barWidth for r in range(len(bars))],['N 3', 'N 5', 'N 10', 'N 20', 'N 30','N 50'], rotation='vertical')
plt.subplots_adjust(bottom=0.23, top=0.92, left=0.15)
plt.legend()
plt.show()

# Movie 5: Three Amigos
# N = 3 0.043
# N = 5 0.029
# N = 10 0.02
# N = 20 0.011
# N = 30 0.012
# N = 50 0.013

bars = [0.043, 0.029, 0.02, 0.011, 0.012, 0.013]
# r1 = [1, 2, 3, 4, 5]
barWidth = 0.4
r1 = np.arange(len(bars))
r2 = [x + barWidth for x in r1]
plt.bar(r2, bars, color='black', width=barWidth, edgecolor='white')
plt.xlabel('Top-N Recommendation', fontweight='bold')
plt.ylabel('Precision', fontweight='bold')
plt.title('Three Amigos', fontweight='bold')
plt.xticks([r + barWidth for r in range(len(bars))],['N 3', 'N 5', 'N 10', 'N 20', 'N 30','N 50'], rotation='vertical')
plt.subplots_adjust(bottom=0.23, top=0.92, left=0.15)
plt.legend()
plt.show()

# Movie 6: Things I Hate About You
# N = 3 0.048
# N = 5 0.044
# N = 10 0.028
# N = 20 0.022
# N = 30 0.02
# N = 50 0.016

bars = [0.048, 0.044, 0.028, 0.022, 0.02, 0.016]
# r1 = [1, 2, 3, 4, 5]
barWidth = 0.4
r1 = np.arange(len(bars))
r2 = [x + barWidth for x in r1]
plt.bar(r2, bars, color='black', width=barWidth, edgecolor='white')
plt.xlabel('Top-N Recommendation', fontweight='bold')
plt.ylabel('Precision', fontweight='bold')
plt.title('Things I Hate About You', fontweight='bold')
plt.xticks([r + barWidth for r in range(len(bars))],['N 3', 'N 5', 'N 10', 'N 20', 'N 30','N 50'], rotation='vertical')
plt.subplots_adjust(bottom=0.23, top=0.92, left=0.15)
plt.legend()
plt.show()




import numpy as np
from scipy.optimize import root

def IRR(x):
    k = 800*math.pow((x/100)+1,-1) + 1300*math.pow((x/100)+1,-2) - 200*math.pow((x/100)+1,-3) - 1200
    return k
root(IRR, x0 = [0])

def PV(x,r):
    k = x*math.pow(1.14,-r)
    return k

def PVsum(num_per, cf, WACC):
    sum = 0
    for i in range(num_per):
        sum+= cf*math.pow(1+WACC, -(i+1))
    return(sum)

print(PVsum(8, 200000, 0.1083)-1000000-1000000*0.75*0.06)
