# Movie Recommendation System - Flask Web Application
# This application provides movie recommendations using content-based filtering and sentiment analysis

# Import necessary libraries for web framework, data processing, ML, and web scraping
import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import bs4 as bs
import urllib.request
import pickle
import requests
from IPython.display import HTML

# Load pre-trained machine learning models from disk
# These models are used for sentiment analysis of movie reviews
filename = 'nlp_model.pkl'
clf = pickle.load(open(filename, 'rb'))  # Pre-trained sentiment classification model
vectorizer = pickle.load(open('tranform.pkl','rb'))  # TF-IDF vectorizer for text preprocessing

def create_similarity():
    """
    Creates a content-based similarity matrix for movie recommendations
    Uses cosine similarity on movie features to find similar movies
    """
    # Load the main dataset containing movie information and combined features
    data = pd.read_csv('main_data.csv')
    
    # Create a count matrix from the combined features (genres, cast, keywords, etc.)
    cv = CountVectorizer()
    count_matrix = cv.fit_transform(data['comb'])  # 'comb' contains concatenated movie features
    
    # Calculate cosine similarity between all movies based on their features
    similarity = cosine_similarity(count_matrix)
    return data,similarity

def rcmd(m):
    """
    Main recommendation function that finds similar movies based on content
    Args: m (str) - movie title to get recommendations for
    Returns: List of 10 similar movie titles or error message
    """
    m = m.lower()  # Convert input to lowercase for case-insensitive matching
    
    # Check if similarity matrix exists, create if not
    try:
        data.head()
        similarity.shape
    except:
        data, similarity = create_similarity()
    
    # Check if the requested movie exists in our database
    if m not in data['movie_title'].unique():
        return('Sorry! The movie you requested is not in our database. Please check the spelling or try with some other movies')
    else:
        # Find the index of the requested movie
        i = data.loc[data['movie_title']==m].index[0]
        
        # Get similarity scores for this movie with all other movies
        lst = list(enumerate(similarity[i]))
        
        # Sort movies by similarity score in descending order
        lst = sorted(lst, key = lambda x:x[1] ,reverse=True)
        lst = lst[1:11] # excluding first item since it is the requested movie itself
        
        # Extract movie titles of top 10 similar movies
        l = []
        for i in range(len(lst)):
            a = lst[i][0]
            l.append(data['movie_title'][a])
        return l
    
def convert_to_list(my_list):
    """
    Utility function to convert string representation of list to actual list
    Handles the format: '["item1","item2"]' -> ["item1","item2"]
    """
    my_list = my_list.split('","')
    my_list[0] = my_list[0].replace('["','')
    my_list[-1] = my_list[-1].replace('"]','')
    return my_list

def get_suggestions():
    """
    Gets all movie titles from database for autocomplete functionality
    Returns: List of capitalized movie titles
    """
    data = pd.read_csv('main_data.csv')
    return list(data['movie_title'].str.capitalize())

# Alternative function for genre-based recommendations (commented out)
'''
def best_movies_by_genre(genre,top_n):
    movie_score = pd.read_csv('movie_score.csv')
    return pd.DataFrame(movie_score.loc[(movie_score[genre]==1)].sort_values(['weighted_score'],ascending=False)[['title']][:top_n])
'''

def best_movies_by_genre(genre,top_n,year = 1920):
    """
    Finds best movies by genre and year filtering
    Args:
        genre (str) - Genre to filter by
        top_n (int) - Number of top movies to return
        year (int) - Minimum year filter (default: 1920)
    Returns: DataFrame with top movies sorted by weighted score
    """
    # Load movie scoring dataset
    movie_score = pd.read_csv('movie_score.csv')
    
    # Extract year from movie title (assumes format: "Movie Title (YYYY)")
    movie_score['year'] = movie_score['title'].apply(lambda _ : int(_[-5:-1]))
    
    # Handle empty year input
    if year == '':
        year = 1920
    
    # Filter movies by minimum year
    movie_score = movie_score[movie_score['year'] >= int(year)]
    
    # Filter by genre and sort by weighted score, return top N movies
    return pd.DataFrame(movie_score.loc[(movie_score[genre]==1)].sort_values(['weighted_score'],ascending=False)[['title','count','mean','weighted_score']][:top_n])

# Initialize Flask web application
app = Flask(__name__)

# Route 1: Home page - displays main interface with movie search
@app.route("/")
@app.route("/home")
def home():
    """
    Renders the home page with autocomplete suggestions
    """
    suggestions = get_suggestions()  # Get all movie titles for autocomplete
    return render_template('home.html',suggestions=suggestions)

# Route 2: Genres page - displays genre selection interface
@app.route("/genres")
def genres():
    """
    Renders the genres selection page
    """
    return render_template('genres.html')

# Route 3: Genre-based movie recommendations
@app.route("/genre", methods = ['GET','POST'])
def genre():
    """
    Handles genre-based movie recommendations
    GET: Shows the genre selection form
    POST: Processes genre selection and returns top movies
    """
    if request.method == 'POST':
        # Get form data from user selection
        result = request.form
        print(result['Genre'])  # Debug: print selected genre
        print(type(result['Genre']))  # Debug: print data type
        
        # Get top 10 movies for the selected genre and year
        df = best_movies_by_genre(result['Genre'],10,result['Year'])
        
        # Reset dataframe index and remove the old index column
        df.reset_index(inplace=True)
        df = df.drop(labels='index', axis=1)
        
        # Convert dataframe to HTML table for display
        html = HTML(df.to_html(classes='table table-striped'))
        
        # Create dictionary for template rendering (movie index: movie title)
        dummy = {}
        for i in range(len(df['title'].tolist())):
            dummy[i] = df['title'].tolist()[i]
        
        # Render results page with movie list and genre name
        return render_template('genre.html',result = dummy, gename = {1:result['Genre']})
    else:
        # If GET request, show the index page
        return render_template('index.html')

# Route 4: AJAX endpoint for similarity-based recommendations
@app.route("/similarity",methods=["POST"])
def similarity():
    """
    AJAX endpoint that returns similar movies based on content similarity
    Used for quick movie-to-movie recommendations
    """
    movie = request.form['name']  # Get movie name from AJAX request
    rc = rcmd(movie)  # Get recommendations using the rcmd function
    
    if type(rc)==type('string'):
        # If error message returned (movie not found)
        return rc
    else:
        # Join movie recommendations with separator for easy parsing on frontend
        m_str="---".join(rc)
        return m_str

# Route 5: Main recommendation page with detailed movie information
@app.route("/recommend",methods=["POST"])
def recommend():
    """
    Main recommendation endpoint that processes detailed movie information
    Performs web scraping for reviews and sentiment analysis
    Returns comprehensive movie details with recommendations
    """
    # Extract all movie data from the AJAX request
    # Basic movie information
    title = request.form['title']
    imdb_id = request.form['imdb_id']
    poster = request.form['poster']
    genres = request.form['genres']
    overview = request.form['overview']
    vote_average = request.form['rating']
    vote_count = request.form['vote_count']
    release_date = request.form['release_date']
    runtime = request.form['runtime']
    status = request.form['status']
    
    # Cast information (received as string representations of lists)
    cast_ids = request.form['cast_ids']
    cast_names = request.form['cast_names']
    cast_chars = request.form['cast_chars']
    cast_bdays = request.form['cast_bdays']
    cast_bios = request.form['cast_bios']
    cast_places = request.form['cast_places']
    cast_profiles = request.form['cast_profiles']
    
    # Recommendation data
    rec_movies = request.form['rec_movies']
    rec_posters = request.form['rec_posters']

    # Get movie suggestions for autocomplete functionality
    suggestions = get_suggestions()

    # Convert string representations to actual lists for processing
    # These come from frontend as JSON-like strings
    rec_movies = convert_to_list(rec_movies)
    rec_posters = convert_to_list(rec_posters)
    cast_names = convert_to_list(cast_names)
    cast_chars = convert_to_list(cast_chars)
    cast_profiles = convert_to_list(cast_profiles)
    cast_bdays = convert_to_list(cast_bdays)
    cast_bios = convert_to_list(cast_bios)
    cast_places = convert_to_list(cast_places)
    
    # Process cast IDs (convert from "[1,2,3]" format to [1,2,3])
    cast_ids = cast_ids.split(',')
    cast_ids[0] = cast_ids[0].replace("[","")
    cast_ids[-1] = cast_ids[-1].replace("]","")
    
    # Clean up cast biographies by handling escaped characters
    for i in range(len(cast_bios)):
        cast_bios[i] = cast_bios[i].replace(r'\n', '\n').replace(r'\"','\"')
    
    # Create data structures for template rendering
    # Dictionary mapping posters to movie titles for recommendation cards
    movie_cards = {rec_posters[i]: rec_movies[i] for i in range(len(rec_posters))}
    
    # Dictionary for cast information display (name -> [id, character, profile_pic])
    casts = {cast_names[i]:[cast_ids[i], cast_chars[i], cast_profiles[i]] for i in range(len(cast_profiles))}

    # Dictionary for detailed cast information (name -> [id, profile_pic, birthday, birthplace, bio])
    cast_details = {cast_names[i]:[cast_ids[i], cast_profiles[i], cast_bdays[i], cast_places[i], cast_bios[i]] for i in range(len(cast_places))}

    # Web scraping section: Extract user reviews from IMDB
    # Construct IMDB reviews URL and fetch the page
    sauce = urllib.request.urlopen('https://www.imdb.com/title/{}/reviews?ref_=tt_ov_rt'.format(imdb_id)).read()
    soup = bs.BeautifulSoup(sauce,'lxml')  # Parse HTML content
    
    # Find all review text elements on the page
    soup_result = soup.find_all("div",{"class":"text show-more__control"})

    # Initialize lists for storing reviews and their sentiment predictions
    reviews_list = [] # list of reviews
    reviews_status = [] # list of comments (good or bad)
    
    # Process each review found on the page
    for reviews in soup_result:
        if reviews.string:  # Check if review text exists
            reviews_list.append(reviews.string)  # Store the review text
            
            # Sentiment Analysis: Pass review through our trained ML model
            movie_review_list = np.array([reviews.string])  # Convert to numpy array
            movie_vector = vectorizer.transform(movie_review_list)  # Vectorize the text
            pred = clf.predict(movie_vector)  # Get sentiment prediction
            
            # Store sentiment result (Good for positive, Bad for negative)
            reviews_status.append('Good' if pred else 'Bad')

    # Create dictionary mapping reviews to their sentiment classifications
    movie_reviews = {reviews_list[i]: reviews_status[i] for i in range(len(reviews_list))}     

    # Render the final recommendation page with all processed data
    # Pass all movie details, cast information, recommendations, and analyzed reviews
    return render_template('recommend.html',title=title,poster=poster,overview=overview,vote_average=vote_average,
        vote_count=vote_count,release_date=release_date,runtime=runtime,status=status,genres=genres,
        movie_cards=movie_cards,reviews=movie_reviews,casts=casts,cast_details=cast_details)

# Application entry point - runs the Flask development server
if __name__ == '__main__':
    app.run(debug=True)  # Enable debug mode for development
