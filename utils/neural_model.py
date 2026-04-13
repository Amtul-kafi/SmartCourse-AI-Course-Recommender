# utils/neural_model.py

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import os

class NeuralRecommender:
    def __init__(self, csv_path="data/courses.csv", model_path="models/neural_model_embeddings.joblib"):
        self.csv_path = csv_path
        self.model_path = model_path

        # Load courses dataset
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"{self.csv_path} not found. Please provide your courses CSV.")
        self.df = pd.read_csv(self.csv_path)

        # Initialize sentence transformer model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

        # Load or compute embeddings
        if os.path.exists(self.model_path):
            self.course_embeddings = joblib.load(self.model_path)
        else:
            self.course_embeddings = self.model.encode(self.df['description'].tolist(), show_progress_bar=True)
            joblib.dump(self.course_embeddings, self.model_path)

    def recommend(self, query, top_k=10):
        """
        Generate top_k course recommendations based on semantic similarity.

        Args:
            query (str): User's natural language input
            top_k (int): Number of recommendations to return

        Returns:
            List[dict]: Top courses with title, department, description, and relevance score
        """
        query_embedding = self.model.encode([query])
        similarities = cosine_similarity(query_embedding, self.course_embeddings)[0]

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
    recommender = NeuralRecommender()
    query = "I want to learn Python for data science"
    results = recommender.recommend(query)
    for r in results:
        print(r)
