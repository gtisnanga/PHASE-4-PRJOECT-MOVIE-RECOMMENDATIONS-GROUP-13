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


# EDA

## Leading Questions
1. User Activity Levels: Which users are the most active, and how does their activity compare to less active users?
2. Item Popularity: Which movies are rated most frequently, and are there patterns in terms of genre or release year?
3. Rating Trends Over Time: How do average ratings and the number of ratings submitted change over time, and are there any seasonal or genre-specific trends?


![image](https://github.com/user-attachments/assets/4212da0d-8974-4c45-b5af-cb75d50970cd)

![image](https://github.com/user-attachments/assets/50b0c221-eeb9-4ff6-9d2a-f6fd2f14f68b)


![image](https://github.com/user-attachments/assets/839144a3-fd17-4da5-a24d-7b7913ab6bd5)

![image](https://github.com/user-attachments/assets/0752aa65-8b3b-4a0e-ba6b-745d91c5a878)

![image](https://github.com/user-attachments/assets/2c568f8e-7f3c-456f-a0e7-d76972fb8e96)

# Modelling
![image](https://github.com/user-attachments/assets/a4b19ff2-785e-41e9-835c-a23daefdb3c6)

Value of 1.1689: This specific RMSE value means that, on average, the estimated ratings from your deployed model deviate from the actual ratings by approximately 1.17 units on the rating scale.
Scale Context: If your ratings are on a scale of 1 to 5, an RMSE of about 1.17 suggests that the predictions are relatively close to the actual ratings. This is a better performance compared to the RMSE of 2.5133 from the hybrid model you evaluated earlier. It implies that the deployed model provides more accurate predictions than the hybrid model.

Analyzing the Updated Hexbin Plot
Key Observations:

Strong Positive Correlation: The plot clearly shows a strong positive correlation between actual and estimated ratings, as evidenced by the diagonal line of dense hexagons. This indicates that the model's predictions generally align with the actual ratings.

Clustering: The data points cluster tightly along the diagonal line, suggesting that the model's predictions are consistently accurate.

Density: The color intensity of the hexagons reveals the density of data points. The darker blue hexagons indicate areas with a higher concentration of data points, suggesting that the model's predictions are more frequent in those ranges of actual and estimated ratings.

Sparse Areas: There are some sparse areas, especially in the lower-left and upper-right corners, indicating fewer data points in those regions. This might be due to a lack of training data or inherent limitations in the model's ability to predict ratings in those areas.

## Conclusion
The evaluation of your movie recommendation system reveals promising results. The RMSE of **1.1689** indicates that the deployed model's predictions are closely aligned with actual ratings, enhancing the overall accuracy compared to the previous hybrid model's RMSE of **2.5133**. This suggests that the refined approach effectively captures user preferences, contributing to a more personalized experience. The hexbin plot analysis further demonstrates a strong positive correlation between actual and estimated ratings, highlighting the model's reliability and consistency in predicting user ratings.

## Recommendations
1. **Refine the Collaborative Filtering Approach**:
   - Continue enhancing the collaborative filtering model by experimenting with different matrix factorization techniques and tuning hyperparameters to further reduce RMSE and improve accuracy.

2. **Leverage Content-Based Filtering**:
   - Implement content-based filtering techniques for new users to alleviate the cold start problem. Use features such as genres, directors, and actors to recommend movies aligned with their preferences.

3. **User Feedback Mechanism**:
   - Introduce a feedback loop that allows users to rate recommended movies. Use this data to refine the model continually, adapting to changing preferences and improving recommendation relevance over time.

4. **Evaluate Additional Metrics**:
   - In addition to RMSE, consider using metrics like **Precision@K**, **Recall**, and **F1 Score** to assess the recommendation system’s effectiveness from different perspectives. These metrics will help ensure that the model not only predicts accurately but also recommends relevant movies.

5. **Continuous Learning**:
   - Explore advanced techniques like reinforcement learning or deep learning approaches to enhance the model’s ability to learn and adapt over time based on user interactions and feedback.

## Way Forward
1. **Conduct A/B Testing**: Implement A/B testing for different recommendation strategies to determine the most effective model. This will provide insights into user engagement and satisfaction levels.

2. **Regular Model Updates**: Schedule regular updates to the model, incorporating new data and user feedback to maintain the system's relevance and accuracy.

3. **Expand Data Sources**: Consider integrating external data sources (e.g., social media trends, movie reviews) to enrich the recommendation process and improve user engagement.

4. **User Segmentation**: Analyze user data to create segments based on viewing habits and preferences. Tailor recommendations for different segments to enhance personalization.

5. **User Interface Enhancement**: Improve the user interface to make the recommendations more visible and accessible. Clear communication of why specific movies are recommended can enhance user trust and satisfaction.

By implementing these recommendations and strategies, you can further enhance your movie recommendation system, ensuring it meets user needs and remains competitive in the evolving landscape of digital streaming platforms.
