import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import os

class TfidfRecommender:
    def __init__(self, csv_path="data/courses.csv", model_path="models/tfidf_vectorizer.joblib"):
        self.csv_path = csv_path
        self.model_path = model_path

        # Ensure models folder exists
        if not os.path.exists(os.path.dirname(model_path)):
            os.makedirs(os.path.dirname(model_path))

        # Load courses dataset
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"{self.csv_path} not found. Please provide your courses CSV.")
        
        self.df = pd.read_csv(self.csv_path)

        # Ensure required columns exist
        required_cols = ['course_id', 'title', 'department', 'description']
        for col in required_cols:
            if col not in self.df.columns:
                raise ValueError(f"CSV is missing required column: {col}")

        # Fill missing descriptions
        self.df['description'] = self.df['description'].fillna('')

        # Load or train TF-IDF vectorizer
        vectors_path = "models/tfidf_course_vectors.joblib"
        if os.path.exists(self.model_path) and os.path.exists(vectors_path):
            self.vectorizer = joblib.load(self.model_path)
            self.course_vectors = joblib.load(vectors_path)
        else:
            self.vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,2))
            self.course_vectors = self.vectorizer.fit_transform(self.df['description'].tolist())
            joblib.dump(self.vectorizer, self.model_path)
            joblib.dump(self.course_vectors, vectors_path)

    def recommend(self, query, top_k=10):
        """Generate top_k course recommendations based on TF-IDF keyword similarity."""
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.course_vectors)[0]

        # Get top_k indices
        top_indices = similarities.argsort()[::-1][:top_k]

        recommendations = []
        for idx in top_indices:
            score = float(similarities[idx]) * 100  # convert to 0-100%
            recommendations.append({
                "course_id": int(self.df.iloc[idx]['course_id']),
                "title": self.df.iloc[idx]['title'],
                "department": self.df.iloc[idx]['department'],
                "description": self.df.iloc[idx]['description'],
                "score": round(score, 2)
            })

        return recommendations

# Example usage
if __name__ == "__main__":
    recommender = TfidfRecommender()
    query = "I want to learn Python for data science"
    results = recommender.recommend(query)
    for r in results:
        print(r)
