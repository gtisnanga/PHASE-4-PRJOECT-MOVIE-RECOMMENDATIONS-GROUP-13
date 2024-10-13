# Movie Recommendation System for Netflix
Phase: 4 Group: 13

Group Members:

- Sylvia Manono
- Amos Kipngetich
- Angela Maina
- Charles Ndegwa
- Sandra Koech
- Gloria Tisnanga
- Alex Miningwa

Student Pace: Part time

Scheduled Project Review Date/Time: October 14, 2024

Instructor Name: Samuel G. Mwangi

# Executive Summary
This project aims to improve Netflix’s recommendation system to solve the issue of users endlessly scrolling to find content they enjoy. By enhancing the recommendation engine, Netflix seeks to provide more personalized and relevant suggestions, increasing user engagement and viewing time, while reducing decision fatigue.

## Business and Data Understanding

Netflix, a global streaming platform with a vast and diverse movie catalog, aims to provide personalized content to ensure user engagement and satisfaction. With thousands of options available, users often face difficulty in finding content that matches their preferences, leading to decision fatigue and lower engagement. This results in users endlessly scrolling without finding something to watch quickly.

To address this, Netflix is focused on enhancing its recommendation engine by suggesting films similar to those users have enjoyed, based on the content and genre of the movies. The goal is to build a content-based recommendation system that provides relevant movie suggestions, helping users discover new content easily, stay engaged, and ultimately increase viewing time. This system will enhance Netflix’s ability to deliver a personalized and enjoyable experience for its users.

# Dataset Overview
The dataset includes:

## Movie Titles, Genres
User IDs, Movie IDs, Ratings This data allows for both content-based and collaborative filtering approaches.
EDA Highlights
Exploratory data analysis revealed key patterns in user ratings, popular genres, and user-movie interactions. Visualization techniques were used to identify trends and correlations in the data.

## Modeling Techniques
Content-Based Filtering: Recommends movies similar to what users have liked using cosine similarity based on genres.
Collaborative Filtering:
KNNBasic: Predicts ratings based on similar users.
KNNWithMeans: Adjusts for user biases by incorporating mean ratings.
SVD: Matrix factorization to uncover hidden patterns in user-item interactions.
Hybrid Model: Combines content-based and collaborative filtering techniques for enhanced performance.

# Objectives

Here are three refined objectives based on your project description:

1. **Develop a Personalized Movie Recommendation System**
   Build a recommendation model using collaborative filtering (e.g., matrix factorization or deep learning techniques like neural collaborative filtering) to predict user ratings for movies. Recommend the top 5 most relevant movies for each user, enhancing the personalization experience on the platform.

2. **Address Cold Start for New Users**
   Tackle the cold start issue by implementing content-based filtering and recommending movies based on user preferences (genres, actors). Complement this with trending or popular movies to ensure engagement while collecting more personalized data for future recommendations.

3. **Enhance System Precision and User Feedback Integration**
   Improve recommendation accuracy and relevance through a hybrid approach that combines collaborative filtering and content-based filtering. Implement a feedback mechanism to refine the model continuously, using metrics such as RMSE, Precision@K, and F1 Score for evaluation, and leverage user ratings to enhance future recommendations.

### 1. Importing Libraries

Import necessary libraries for data manipulation, modeling, and evaluation.

```
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import numpy as np
from surprise import SVD
from surprise.prediction_algorithms import KNNWithMeans, KNNBasic, KNNBaseline
from surprise import Dataset, Reader
from surprise import accuracy
from surprise.model_selection import cross_validate, train_test_split, GridSearchCV
import warnings
warnings.filterwarnings('ignore')

```

### 2. Loading and Inspecting the datasets

```
# Load the CSV files
movies_df = pd.read_csv('ml-latest-small/movies.csv')
ratings_df = pd.read_csv('ml-latest-small/ratings.csv')
tags_df = pd.read_csv('ml-latest-small/tags.csv')
links_df = pd.read_csv('ml-latest-small/links.csv')

```

```
# Display the first few rows of each file to understand their structure
print(movies_df.head())
print(ratings_df.head())
print(tags_df.head())
print(links_df.head())

```

```
 movieId                               title  \
0        1                    Toy Story (1995)   
1        2                      Jumanji (1995)   
2        3             Grumpier Old Men (1995)   
3        4            Waiting to Exhale (1995)   
4        5  Father of the Bride Part II (1995)   

                                        genres  
0  Adventure|Animation|Children|Comedy|Fantasy  
1                   Adventure|Children|Fantasy  
2                               Comedy|Romance  
3                         Comedy|Drama|Romance  
4                                       Comedy  
   userId  movieId  rating  timestamp
0       1        1     4.0  964982703
1       1        3     4.0  964981247
2       1        6     4.0  964982224
3       1       47     5.0  964983815
4       1       50     5.0  964982931
   userId  movieId              tag   timestamp
0       2    60756            funny  1445714994
1       2    60756  Highly quotable  1445714996
2       2    60756     will ferrell  1445714992
3       2    89774     Boxing story  1445715207
4       2    89774              MMA  1445715200
   movieId  imdbId   tmdbId
0        1  114709    862.0
1        2  113497   8844.0
2        3  113228  15602.0
3        4  114885  31357.0
4        5  113041  11862.0

```

### 4. Data Merging

```
# Merge ratings with movies to associate ratings with movie titles
ratings_movies_df = pd.merge(ratings_df, movies_df, on='movieId', how='inner')

# Merge with tags to include movie tags for content-based filtering
ratings_movies_tags_df = pd.merge(ratings_movies_df, tags_df, on=['userId', 'movieId'], how='left')

# Merge with links to associate external database IDs (if needed)
final_df = pd.merge(ratings_movies_tags_df, links_df, on='movieId', how='left')

# Inspect the final dataset
print(final_df.head())

```
```
userId  movieId  rating  timestamp_x             title  \
0       1        1     4.0    964982703  Toy Story (1995)   
1       5        1     4.0    847434962  Toy Story (1995)   
2       7        1     4.5   1106635946  Toy Story (1995)   
3      15        1     2.5   1510577970  Toy Story (1995)   
4      17        1     4.5   1305696483  Toy Story (1995)   

                                        genres  tag  timestamp_y  imdbId  \
0  Adventure|Animation|Children|Comedy|Fantasy  NaN          NaN  114709   
1  Adventure|Animation|Children|Comedy|Fantasy  NaN          NaN  114709   
2  Adventure|Animation|Children|Comedy|Fantasy  NaN          NaN  114709   
3  Adventure|Animation|Children|Comedy|Fantasy  NaN          NaN  114709   
4  Adventure|Animation|Children|Comedy|Fantasy  NaN          NaN  114709   

   tmdbId  
0   862.0  
1   862.0  
2   862.0  
3   862.0  
4   862.0  

```
# DATA DESCRIPTION
There are a number of csv files available with different columns in the Data file. 


movies.csv

movieId - Unique identifier for each movie.

title - The movie titles.

genre - The various genres a movie falls into.


ratings.csv

userId - Unique identifier for each user

movieId - Unique identifier for each movie.

rating - A value between 0 to 5 that a user rates a movie on. 5 is the highest while 0 is the lowest rating.

timestamp - This are the seconds that have passed since Midnight January 1, 1970(UTC)


tags.csv

userId - Unique identifier for each user

movieId - Unique identifier for each movie.

tag - A phrase determined by the user.

timestamp - This are the seconds that have passed since Midnight January 1, 1970(UTC)


links.csv

movieId - It's an identifier for movies used by https://movielens.org and has link to each movie.

imdbId - It's an identifier for movies used by http://www.imdb.com and has link to each movie.

tmdbId - is an identifier for movies used by https://www.themoviedb.org and has link to each movie.

## Data Understanding

```
# checking rows and coulumns in merged data set
print(f'Shape for the merged dataset, {final_df.shape}')

```
```
Shape for the merged dataset, (102677, 10)
```

```

# checking data types
final_df.info()

```
```
<class 'pandas.core.frame.DataFrame'>
Int64Index: 102677 entries, 0 to 102676
Data columns (total 10 columns):
 #   Column       Non-Null Count   Dtype  
---  ------       --------------   -----  
 0   userId       102677 non-null  int64  
 1   movieId      102677 non-null  int64  
 2   rating       102677 non-null  float64
 3   timestamp_x  102677 non-null  int64  
 4   title        102677 non-null  object 
 5   genres       102677 non-null  object 
 6   tag          3476 non-null    object 
 7   timestamp_y  3476 non-null    float64
 8   imdbId       102677 non-null  int64  
 9   tmdbId       102664 non-null  float64
dtypes: float64(3), int64(4), object(3)
memory usage: 8.6+ MB

```
```

# checking columns
final_df.columns

```

```
Index(['userId', 'movieId', 'rating', 'timestamp_x', 'title', 'genres', 'tag',
       'timestamp_y', 'imdbId', 'tmdbId'],
      dtype='object')

```

```
#checking for missing values
final_df.isna().sum()

```

```
userId             0
movieId            0
rating             0
timestamp_x        0
title              0
genres             0
tag            99201
timestamp_y    99201
imdbId             0
tmdbId            13
dtype: int64

```

```
# Dealing with the null values
# Drop 'tag' and 'timestamp_y' columns, as they have many null values
final_df.drop(columns=['tag', 'timestamp_y'], inplace=True)

# Fill NaNs in 'tmdbId' column with 0
final_df['tmdbId'].fillna(0, inplace=True)

# Drop any remaining rows with null values in the dataset
final_df.dropna(inplace=True)

# Check for NaN values again to confirm
print(final_df.isna().sum())

```

```
userId         0
movieId        0
rating         0
timestamp_x    0
title          0
genres         0
imdbId         0
tmdbId         0
dtype: int64

```

 Rationale for dropping values
 Columns that contain a significant number of missing values (like 'tag' and 'timestamp_y') are dropped. The 'tmdbId' is filled with zeros
 to ensure completeness.
 
 ```
 # checking shape after dropping null values
final_df.shape

```

```
(102677, 8)

```

```
#checking for duplicate values
final_df.duplicated().sum()

```
```
1841
```

```
# Check for duplicated rows
duplicates = final_df[final_df.duplicated()]

```

```
# Drop duplicate rows
final_df.drop_duplicates(inplace=True)

# Verify if duplicates are removed
print(final_df.duplicated().sum())

```

```
0
```

```
# Check for duplicated rows
duplicates = final_df[final_df.duplicated()]

```

```
# rechecking shape
final_df.shape

```

```
(100836, 8)

```

# EDA

## Leading Questions
1. User Activity Levels: Which users are the most active, and how does their activity compare to less active users?
2. Item Popularity: Which movies are rated most frequently, and are there patterns in terms of genre or release year?
3. Rating Trends Over Time: How do average ratings and the number of ratings submitted change over time, and are there any seasonal or genre-specific trends?

```
# Univariated Analysis
# Count the number of ratings per user
user_activity = final_df['userId'].value_counts()

# Plot the user activity distribution
plt.figure(figsize=(10,6))
sns.histplot(user_activity, bins=30, kde=True)
plt.title('User Activity Levels (Number of Ratings per User)')
plt.xlabel('Number of Ratings')
plt.ylabel('Count of Users')
plt.show()

# Count the number of ratings per movie
movie_popularity = final_df['movieId'].value_counts()

# Plot the movie popularity distribution
plt.figure(figsize=(10,6))
sns.histplot(movie_popularity, bins=30, kde=True)
plt.title('Movie Popularity (Number of Ratings per Movie)')
plt.xlabel('Number of Ratings')
plt.ylabel('Count of Movies')
plt.show()

# Plot the distribution of ratings
plt.figure(figsize=(10,6))
sns.countplot(x='rating', data=final_df)
plt.title('Rating Distribution')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.show()

```
