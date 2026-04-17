# SmartCourse – AI Powered Course Recommendation System (Prototype)

---

##  Overview
SmartCourse is a Flask-based AI prototype that recommends courses based on user input in natural language. It uses Natural Language Processing (NLP) and Machine Learning techniques to match user queries with relevant courses from a dataset.

The system demonstrates how recommendation engines can be built using both traditional and semantic similarity approaches.

---

##  Core Idea
Users enter learning preferences such as:
- "I want to learn Python for cybersecurity"
- "AI for beginners"
- "Data science for machine learning"

The system processes the input and returns relevant course recommendations.

---

##  Models Used

### 1. TF-IDF Model
- Uses keyword-based matching
- Compares user query with course titles and descriptions
- Works best for direct keyword similarity

### 2. Neural Model
- Uses sentence embeddings
- Understands meaning of the query
- Finds conceptually similar courses even if wording is different

---

##  Features
- Course recommendations using NLP
- Two AI models (TF-IDF + Neural Embeddings)
- Save favorite recommendations
- View search history
- Compare both models
- Interactive web interface (Flask frontend)

---

##  Project Structure

- **app.py** → Main Flask application (routes + APIs)

### utils/
- **database.py** → Handles SQLite database operations
- **tfidf_model.py** → TF-IDF recommendation logic
- **neural_model.py** → Neural embedding recommendation logic

### templates/
- **home.html** → Homepage
- **recommend.html** → Recommendation page
- **dashboard.html** → Saved results + history
- **about.html** → About page

### static/
- CSS and JavaScript files for frontend

### data/
- **courses.csv** → Dataset used for recommendations

### database/
- **smartcourse.db** → Stores user history and saved data

### models/
- Pre-trained ML models and embeddings

---

##  API Endpoints

### POST `/api/recommend`
- Input: `query`, `model (tfidf / neural)`
- Output: Top recommended courses
- Automatically saves search history

---

### GET `/api/history`
- Returns all previous searches and recommendations

---

### POST `/api/save`
- Input: `course_id`, `query`, `model`
- Saves selected recommendation manually

---

##  How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt


