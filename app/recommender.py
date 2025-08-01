import pandas as pd
import string
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import difflib

BASE_DIR = Path(__file__).resolve().parent

# Globals (only loaded once)
books_cb = None
cosine_sim = None
indices = None
model = None
books = None
ratings = None

# Load content-based recommendation data
def load_content_data():
    global books_cb, cosine_sim, indices
    if books_cb is not None:
        return  # Already loaded

    books_raw = pd.read_csv(BASE_DIR / "data/books_cleaned.csv")
    books_cb = books_raw.drop_duplicates(subset='Book-Title').reset_index(drop=True)
    books_cb = books_cb.head(100)  # TEMP: limit for performance during development
    books_cb['Book-Title'] = books_cb['Book-Title'].fillna('')

    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(books_cb['Book-Title'])

    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    normalized_titles = books_cb['Book-Title'].str.lower().str.strip()
    normalized_titles = normalized_titles.apply(
        lambda x: x.translate(str.maketrans('', '', string.punctuation))
    )
    indices = pd.Series(books_cb.index, index=normalized_titles).drop_duplicates()

# Get content-based recommendations
def get_content_recommendations(title: str, n: int = 5):
    load_content_data()

    title = title.strip().lower()
    title = title.translate(str.maketrans('', '', string.punctuation))

    if title not in indices:
        close_matches = difflib.get_close_matches(title, indices.index, n=1, cutoff=0.6)
        if not close_matches:
            return ["❌ Book not found."]
        title = close_matches[0]

    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:n+1]
    book_indices = [i[0] for i in sim_scores]

    return books_cb['Book-Title'].iloc[book_indices].tolist()

# Load collaborative filtering model and data
def load_collab_model():
    global model, books, ratings
    if model is not None:
        return

    import joblib
    books = pd.read_csv(BASE_DIR / "data/books_cleaned.csv")
    ratings = pd.read_csv(BASE_DIR / "data/ratings_cleaned.csv")
    model = joblib.load(BASE_DIR / "data/svd_model.pkl")

# Get collaborative filtering recommendations
def get_collab_recommendations(user_id: int, n: int = 5):
    try:
        load_collab_model()
        user_rated_books = ratings[ratings['User-ID'] == user_id]['ISBN'].tolist()
        all_books = books['ISBN'].unique()
        books_to_predict = [isbn for isbn in all_books if isbn not in user_rated_books]

        predictions = [(isbn, model.predict(user_id, isbn).est) for isbn in books_to_predict]
        predictions.sort(key=lambda x: x[1], reverse=True)
        top_isbns = [isbn for isbn, _ in predictions[:n]]

        return books[books['ISBN'].isin(top_isbns)]['Book-Title'].tolist()
    except Exception as e:
        return [f"❌ Error during prediction: {e}"]