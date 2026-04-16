import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import os
import re
import spacy

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")


def preprocess_text(text):
    """Clean and lemmatize text using spaCy."""
    if not isinstance(text, str) or text.strip() == "":
        return ""
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    # Lowercase
    text = text.lower().strip()
    # spaCy lemmatization + stopword removal
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and len(token.lemma_) > 2]
    return " ".join(tokens)


class TfidfRecommender:
    def __init__(self, csv_path="data/courses.csv", model_path="models/tfidf_vectorizer.joblib"):
        self.csv_path = csv_path
        self.model_path = model_path
        self.vectors_path = "models/tfidf_course_vectors.joblib"
        self.processed_path = "models/processed_df.joblib"

        # Ensure models folder exists
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        # Load courses dataset
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"{self.csv_path} not found.")

        df_raw = pd.read_csv(self.csv_path)

        # ── Column Mapping ──────────────────────────────────────────
        # Map your actual CSV columns to internal standard names
        self.df = pd.DataFrame()
        self.df['course_id']    = range(1, len(df_raw) + 1)  # auto-generate IDs
        self.df['title']        = df_raw.get('Title', df_raw.get('Course Title', 'Unknown')).fillna('Unknown')
        self.df['department']   = df_raw.get('Category', df_raw.get('COURSE CATEGORIES', 'General')).fillna('General')
        self.df['description']  = df_raw.get('Short Intro', df_raw.get('Course Short Intro', '')).fillna('')
        self.df['university']   = df_raw.get('School', df_raw.get('Site', 'Unknown')).fillna('Unknown')
        self.df['level']        = df_raw.get('Level', 'Unknown').fillna('Unknown')
        self.df['rating']       = pd.to_numeric(df_raw.get('Rating', 0), errors='coerce').fillna(0)
        self.df['skills']       = df_raw.get('Skills', '').fillna('')
        self.df['what_you_learn'] = df_raw.get('What you learn', '').fillna('')
        self.df['sub_category'] = df_raw.get('Sub-Category', '').fillna('')

        # Remove duplicates
        self.df.drop_duplicates(subset=['title', 'description'], inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        self.df['course_id'] = range(1, len(self.df) + 1)

        # ── Combined text for vectorization ─────────────────────────
        # Combine multiple fields for richer matching
        self.df['combined_text'] = (
            self.df['title'] + " " +
            self.df['department'] + " " +
            self.df['sub_category'] + " " +
            self.df['skills'] + " " +
            self.df['what_you_learn'] + " " +
            self.df['description']
        )

        # ── Load or Train Model ──────────────────────────────────────
        if (os.path.exists(self.model_path) and
                os.path.exists(self.vectors_path) and
                os.path.exists(self.processed_path)):
            print("[TF-IDF] Loading saved model...")
            self.vectorizer = joblib.load(self.model_path)
            self.course_vectors = joblib.load(self.vectors_path)
            self.df = joblib.load(self.processed_path)
        else:
            print("[TF-IDF] Preprocessing text with spaCy (first run, may take a few minutes)...")
            self.df['processed_text'] = self.df['combined_text'].apply(preprocess_text)

            print("[TF-IDF] Training TF-IDF vectorizer...")
            self.vectorizer = TfidfVectorizer(
                stop_words='english',
                ngram_range=(1, 2),
                max_features=50000,
                sublinear_tf=True
            )
            self.course_vectors = self.vectorizer.fit_transform(self.df['processed_text'].tolist())

            # Save everything
            joblib.dump(self.vectorizer, self.model_path)
            joblib.dump(self.course_vectors, self.vectors_path)
            joblib.dump(self.df, self.processed_path)
            print("[TF-IDF] Model trained and saved!")

    def recommend(self, query, top_k=10):
        """Generate top_k course recommendations based on TF-IDF keyword similarity."""
        # Preprocess the query same way as training data
        processed_query = preprocess_text(query)
        query_vector = self.vectorizer.transform([processed_query])
        similarities = cosine_similarity(query_vector, self.course_vectors)[0]

        # Get top_k indices
        top_indices = similarities.argsort()[::-1][:top_k]

        recommendations = []
        for idx in top_indices:
            score = float(similarities[idx]) * 100
            recommendations.append({
                "course_id":   int(self.df.iloc[idx]['course_id']),
                "title":       str(self.df.iloc[idx]['title']),
                "department":  str(self.df.iloc[idx]['department']),
                "description": str(self.df.iloc[idx]['description']),
                "university":  str(self.df.iloc[idx]['university']),
                "level":       str(self.df.iloc[idx]['level']),
                "rating":      float(self.df.iloc[idx]['rating']),
                "score":       round(score, 2)
            })

        return recommendations


# Example usage
if __name__ == "__main__":
    recommender = TfidfRecommender()
    results = recommender.recommend("I want to learn Python for data science")
    for r in results:
        print(r)