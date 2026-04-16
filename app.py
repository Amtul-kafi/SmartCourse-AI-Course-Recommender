from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from utils.database import Database
from utils.tfidf_model import TfidfRecommender
from utils.neural_model import NeuralRecommender
import os
from datetime import datetime

app = Flask(__name__)
app.secret_key = "smartcourse_secret"

DATA_PATH = os.path.join("data", "courses.csv")

tfidf_model = TfidfRecommender(csv_path=DATA_PATH)
neural_model = NeuralRecommender(csv_path=DATA_PATH)

db = Database()


# =========================
# LOGIN PAGE (DEFAULT PAGE)
# =========================
@app.route('/')
def login():
    return render_template("login.html")


# =========================
# HOME PAGE
# =========================
@app.route('/home')
def home():
    return render_template('home.html')


# =========================
# ABOUT
# =========================
@app.route('/about')
def about():
    return render_template('about.html')


# =========================
# RECOMMEND
# =========================
@app.route('/recommend')
def recommend():
    return render_template('recommend.html')


# =========================
# DASHBOARD
# =========================
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


# =========================
# API: RECOMMEND
# =========================
@app.route('/api/recommend', methods=['POST'])
def api_recommend():
    data = request.json
    query = data.get("query", "").strip()
    model_type = data.get("model", "tfidf")

    if not query:
        return jsonify({"error": "Query required"}), 400

    if model_type == "tfidf":
        results = tfidf_model.recommend(query, top_k=10)
    else:
        results = neural_model.recommend(query, top_k=10)

    
    formatted = []
    for c in results:
        formatted.append({
            "course_id":  c.get("course_id", 0),
            "title":      c.get("title", ""),
            "department": c.get("department", ""),
            "description":c.get("description", ""),
            "university": c.get("university", ""),
            "level":      c.get("level", ""),
            "rating":     c.get("rating", 0),
            "relevance":  round(c.get("score", 0), 2)   # model returns 'score'
        })

    session_id = datetime.now().strftime("%Y%m%d%H%M%S")
    db.save_query_results(session_id, query, model_type, formatted)

    return jsonify({
        "session_id": session_id,
        "recommendations": formatted
    })


# =========================
# API: HISTORY (GET)
# =========================
@app.route('/api/history')
def api_history():
    history = db.get_history()
    return jsonify(history)


# =========================
# API: COMPARE (GET)
# =========================
@app.route('/api/compare', methods=['POST'])
def api_compare():
    """Run same query through both models and return side-by-side results."""
    data = request.json
    query = data.get("query", "").strip()

    if not query:
        return jsonify({"error": "Query required"}), 400

    tfidf_results  = tfidf_model.recommend(query, top_k=10)
    neural_results = neural_model.recommend(query, top_k=10)

    def fmt(results):
        return [{
            "course_id":  c.get("course_id", 0),
            "title":      c.get("title", ""),
            "department": c.get("department", ""),
            "description":c.get("description", ""),
            "university": c.get("university", ""),
            "level":      c.get("level", ""),
            "rating":     c.get("rating", 0),
            "relevance":  round(c.get("score", 0), 2)
        } for c in results]

    session_id = datetime.now().strftime("%Y%m%d%H%M%S")
    db.save_query_results(session_id, query, "tfidf",  fmt(tfidf_results))
    db.save_query_results(session_id, query, "neural", fmt(neural_results))

    return jsonify({
        "query":   query,
        "tfidf":   fmt(tfidf_results),
        "neural":  fmt(neural_results)
    })


# =========================
# API: SAVE
# =========================
@app.route('/api/save', methods=['POST'])
def api_save():
    data = request.json
    db.save_single_recommendation(
        data.get("course_id"),
        data.get("query"),
        data.get("model")
    )
    return jsonify({"message": "saved"})


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)