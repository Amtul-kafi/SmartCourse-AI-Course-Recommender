SmartCourse – AI Powered Course Recommendation System (Prototype Phase)


SmartCourse - AI Powered Course Recommender

This project is a prototype of an AI-based course recommendation system called SmartCourse. It allows users to type their learning preferences in natural language and get recommended courses. The system uses two different recommendation models to give better results.

Project Overview

SmartCourse helps users find courses they want to learn based on what they type. Users can enter queries like "I want to learn Python for cybersecurity" or "AI for beginners" and get recommendations from two models:

TF-IDF Model: This model finds courses by matching keywords in the course title and description.

Neural Model: This model understands the meaning of the query using sentence embeddings and finds courses that are conceptually similar.

Users can also save their favorite recommendations, view their search history, and compare results from both models side by side.

Project Structure:

Here is how the project is organized:

app.py : This is the main Flask backend file. It handles page routes and API endpoints.

requirements.txt : Contains all the Python packages needed to run the project.

utils/ : This folder contains the helper scripts:

database.py : Handles saving and retrieving search history and recommendations from the database.

tfidf_model.py : The TF-IDF recommendation model.

neural_model.py : The neural recommendation model using sentence embeddings.

static/ : Contains CSS and JavaScript files:

styles.css : Custom styling for the website.

recommend.js : JavaScript for handling recommendation requests.

dashboard.js : JavaScript for dashboard interactions.

templates/ : Contains HTML templates for the web pages:

home.html : Homepage of the website.

recommend.html : Page where users enter their queries and get recommendations.

dashboard.html : Page to view saved recommendations, search history, and model comparison.

about.html : About page with project information.

database/ : Contains the SQLite database smartcourse.db which stores all search and recommendation data.

data/ : Contains the course dataset courses.csv used for generating recommendations.

APIs

The project backend provides REST APIs for the frontend to work:

POST /api/recommend

Accepts: query (text), model ("tfidf" or "neural")

Returns: Top recommended courses for the query, including title, department, description, and relevance score.

Saves the query and results to the database automatically.

GET /api/history

Returns all previous search queries and their recommendations.

POST /api/save

Accepts: course_id, query, model

Saves a single recommendation manually to the database.

How to Run

Clone the project folder.

Make sure Python 3.10 or above is installed.

Install the required packages using:

pip install -r requirements.txt


Make sure courses.csv is in the data folder.

Run the Flask app:

python3 app.py


Open the browser and go to http://127.0.0.1:5000/

Usage

Go to the homepage and click "Get Your Recommendations" to go to the recommendation page.

Enter your learning preference and select a model.

Click "Get Recommendations" to see the top courses.

Click "Save Recommendation" to save a course.

Go to the dashboard to view all your saved recommendations, history, and model comparisons.

This project is a working prototype and can be extended further to include more advanced models, better UI, and user authentication in the future.