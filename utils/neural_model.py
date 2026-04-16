# utils/neural_model.py

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
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
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = text.lower().strip()
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and len(token.lemma_) > 2]
    return " ".join(tokens)


class NeuralRecommender:
    def __init__(self, csv_path="data/courses.csv", model_path="models/neural_model_embeddings.joblib"):
        self.csv_path = csv_path
        self.model_path = model_path
        self.processed_path = "models/neural_processed_df.joblib"

        os.makedirs("models", exist_ok=True)

        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"{self.csv_path} not found.")

        # ── Load and map columns ─────────────────────────────────────
        if os.path.exists(self.processed_path):
            print("[Neural] Loading saved processed dataframe...")
            self.df = joblib.load(self.processed_path)
        else:
            df_raw = pd.read_csv(self.csv_path)

            self.df = pd.DataFrame()
            self.df['course_id']    = range(1, len(df_raw) + 1)
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

            # Combined text for embeddings
            self.df['combined_text'] = (
                self.df['title'] + " " +
                self.df['department'] + " " +
                self.df['sub_category'] + " " +
                self.df['skills'] + " " +
                self.df['what_you_learn'] + " " +
                self.df['description']
            )

            print("[Neural] Preprocessing text with spaCy...")
            self.df['processed_text'] = self.df['combined_text'].apply(preprocess_text)

            joblib.dump(self.df, self.processed_path)
            print("[Neural] Processed dataframe saved!")

        # ── Load sentence transformer ────────────────────────────────
        print("[Neural] Loading SentenceTransformer model...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

        # ── Load or compute embeddings ───────────────────────────────
        if os.path.exists(self.model_path):
            print("[Neural] Loading saved embeddings...")
            self.course_embeddings = joblib.load(self.model_path)
        else:
            print("[Neural] Computing embeddings (first run, may take a few minutes)...")
            self.course_embeddings = self.model.encode(
                self.df['processed_text'].tolist(),
                show_progress_bar=True,
                batch_size=64
            )
            joblib.dump(self.course_embeddings, self.model_path)
            print("[Neural] Embeddings saved!")

    def recommend(self, query, top_k=10):
        """Generate top_k course recommendations based on semantic similarity."""
        processed_query = preprocess_text(query)
        query_embedding = self.model.encode([processed_query])
        similarities = cosine_similarity(query_embedding, self.course_embeddings)[0]

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


if __name__ == "__main__":
    recommender = NeuralRecommender()
    results = recommender.recommend("I want to learn Python for data science")
    for r in results:
        print(r)