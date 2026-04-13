from flask import Flask, render_template, request, jsonify
from utils.database import Database
from utils.tfidf_model import TfidfRecommender
from utils.neural_model import NeuralRecommender
import os
from datetime import datetime

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'

# ------------------------------
# Paths
# ------------------------------
DATA_PATH = os.path.join("data", "courses.csv")

# ------------------------------
# Load Models
# ------------------------------
tfidf_model = TfidfRecommender(csv_path=DATA_PATH)
neural_model = NeuralRecommender(csv_path=DATA_PATH)

# ------------------------------
# Database
# ------------------------------
db = Database()

# ------------------------------
# Routes
# ------------------------------

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/recommend')
def recommend():
    return render_template('recommend.html')

@app.route('/dashboard')
def dashboard():
    search_history = db.get_history()
    saved_sessions = db.get_saved_recommendations()
    model_comparison = db.get_model_comparisons()
    return render_template(
        'dashboard.html',
        search_history=search_history,
        saved_sessions=saved_sessions,
        model_comparison=model_comparison
    )

# ------------------------------
# API Endpoints
# ------------------------------

@app.route('/api/recommend', methods=['POST'])
def api_recommend():
    data = request.json
    query = data.get("query", "").strip()
    model_type = data.get("model", "tfidf")

    if not query:
        return jsonify({"error": "Query text is required."}), 400

    results = []

    if model_type == "tfidf":
        results = tfidf_model.recommend(query, top_k=10)
    elif model_type == "neural":
        results = neural_model.recommend(query, top_k=10)
    else:
        return jsonify({"error": "Invalid model type."}), 400

    formatted_results = []
    for idx, course in enumerate(results):
        formatted_results.append({
            "course_id": course.get("id", idx),
            "title": course.get("title", "No Title"),
            "department": course.get("department", "No Department"),
            "description": course.get("description", "No Description"),
            "relevance": round(course.get("relevance", 0), 2)
        })

    # Save search + recommendations
    session_id = datetime.now().strftime("%Y%m%d%H%M%S")
    db.save_query_results(session_id, query, model_type, formatted_results)

    return jsonify({"session_id": session_id, "recommendations": formatted_results})


@app.route('/api/history', methods=['GET'])
def api_history():
    history = db.get_history()
    return jsonify(history)


@app.route('/api/save', methods=['POST'])
def api_save():
    data = request.json
    course_id = data.get("course_id")
    query = data.get("query")
    model_type = data.get("model")

    if not all([course_id, query, model_type]):
        return jsonify({"error": "Course ID, query, and model type are required"}), 400

    db.save_single_recommendation(course_id, query, model_type)
    return jsonify({"message": "Recommendation saved successfully."})

# ------------------------------
# Run App
# ------------------------------
if __name__ == "__main__":
    app.run(debug=True)
