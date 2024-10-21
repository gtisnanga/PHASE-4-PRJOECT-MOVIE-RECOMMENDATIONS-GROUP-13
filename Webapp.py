import gradio as gr
import pandas as pd
import joblib
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split

# Load the pre-trained model
model = joblib.load('optimized_svd_model.pkl')

# Load movie data
file_path = r"C:\Users\PC\Documents\Flatiron\dsc-data-science-env-config\Project_phase_4\movies.csv"
movies_df = pd.read_csv(file_path)

# Function to generate recommendations
def get_recommendations(user_id, top_n=5):
    file_path = r"C:\Users\PC\Documents\Flatiron\dsc-data-science-env-config\Project_phase_4\ratings.csv"
    ratings_df = pd.read_csv(file_path)
    combined_ratings_df = pd.concat([ratings_df, pd.DataFrame({'userId': [user_id]*len(movies_df), 'movieId': movies_df['movieId'], 'rating': [0]*len(movies_df)})], axis=0)
    combined_ratings_df = combined_ratings_df[['userId', 'movieId', 'rating']]
    combined_ratings_df['rating'] = combined_ratings_df['rating'].astype(float)

    reader = Reader(rating_scale=(0.5, 5.0))
    new_data = Dataset.load_from_df(combined_ratings_df, reader)

    trainset, testset = train_test_split(new_data, test_size=0.2, random_state=42)
    model.fit(trainset)
    predictions = model.test(testset)

    list_of_movies = [(m_id, model.predict(user_id, m_id).est) for m_id in movies_df['movieId']]
    ranked_movies = sorted(list_of_movies, key=lambda x: x[1], reverse=True)

    top_movies = ranked_movies[:top_n]
    top_movie_titles = [(movies_df.loc[movies_df['movieId'] == rec[0], 'title'].values[0], rec[1]) for rec in top_movies]
    
    return [f'Recommendation #{i+1}: {title} (Predicted Rating: {rating:.2f})' for i, (title, rating) in enumerate(top_movie_titles)]

# Gradio interface
def recommend(user_id, top_n):
    recommendations = get_recommendations(user_id, top_n)
    return "\n".join(recommendations)

# Creating the Gradio interface
interface = gr.Interface(
    fn=recommend,
    inputs=[
        gr.Number(label="Enter your user ID", value=1000),
        gr.Slider(label="Number of recommendations", minimum=1, maximum=10, value=5)
    ],
    outputs=gr.Textbox(label="Top Recommendations"),
    title="Movie Recommendation System"
)

# Launch the Gradio app
interface.launch()
