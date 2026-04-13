# utils/database.py
import sqlite3
from datetime import datetime
from typing import List, Dict, Optional

DB_PATH = "database/smartcourse.db"

class Database:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self.init_db()

    def init_db(self):
        """Initialize database tables if they do not exist."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        # Table for storing search queries
        c.execute("""
        CREATE TABLE IF NOT EXISTS search_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT NOT NULL,
            timestamp TEXT NOT NULL
        )
        """)

        # Table for storing recommendations for each query
        c.execute("""
        CREATE TABLE IF NOT EXISTS recommendations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            search_id INTEGER NOT NULL,
            model TEXT NOT NULL,  -- 'tfidf' or 'neural'
            course_id INTEGER NOT NULL,
            course_title TEXT,
            department TEXT,
            description TEXT,
            relevance REAL,
            FOREIGN KEY(search_id) REFERENCES search_history(id)
        )
        """)

        conn.commit()
        conn.close()

    def save_search(self, query: str) -> int:
        """Save a search query with the current timestamp, returns search_id."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        c.execute("INSERT INTO search_history (query, timestamp) VALUES (?, ?)", (query, timestamp))
        search_id = c.lastrowid
        conn.commit()
        conn.close()
        return search_id

    def save_recommendations(self, search_id: int, model: str, courses: List[Dict]):
        """
        Save course recommendations for a given search.
        courses: list of dicts with keys: course_id, title, department, description, relevance
        """
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        for course in courses:
            c.execute("""
            INSERT INTO recommendations 
            (search_id, model, course_id, course_title, department, description, relevance)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                search_id,
                model,
                course['course_id'],
                course['title'],
                course['department'],
                course['description'],
                course['relevance']
            ))
        conn.commit()
        conn.close()

    def get_history(self) -> List[Dict]:
        """Return all search queries with timestamps and associated recommendations."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        # Get all search history in descending order
        c.execute("SELECT id, query, timestamp FROM search_history ORDER BY timestamp DESC")
        history_rows = c.fetchall()

        # Build full history with recommendations grouped by search_id
        history = []
        for search_id, query, timestamp in history_rows:
            c.execute("""
            SELECT model, course_id, course_title, department, description, relevance 
            FROM recommendations 
            WHERE search_id = ?
            """, (search_id,))
            recs = c.fetchall()
            rec_list = []
            for r in recs:
                rec_list.append({
                    'model': r[0],
                    'course_id': r[1],
                    'title': r[2],
                    'department': r[3],
                    'description': r[4],
                    'relevance': r[5]
                })
            history.append({
                'search_id': search_id,
                'query': query,
                'timestamp': timestamp,
                'recommendations': rec_list
            })

        conn.close()
        return history

    def get_recommendations_by_search(self, search_id: int, model: Optional[str] = None) -> List[Dict]:
        """Retrieve recommendations for a specific search, optionally filtered by model."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        if model:
            c.execute("""
            SELECT course_id, course_title, department, description, relevance
            FROM recommendations
            WHERE search_id = ? AND model = ?
            """, (search_id, model))
        else:
            c.execute("""
            SELECT course_id, course_title, department, description, relevance
            FROM recommendations
            WHERE search_id = ?
            """, (search_id,))
        rows = c.fetchall()
        conn.close()
        result = []
        for r in rows:
            result.append({
                'course_id': r[0],
                'title': r[1],
                'department': r[2],
                'description': r[3],
                'relevance': r[4]
            })
        return result
    # -----------------------------------------
    # Save query + recommendations together
    # -----------------------------------------
    def save_query_results(self, session_id: str, query: str, model: str, courses: List[Dict]):
        """
        This matches your app.py call.
        session_id is not used because your DB structure
        already links by search_id.
        """
        search_id = self.save_search(query)
        self.save_recommendations(search_id, model, courses)


    # -----------------------------------------
    # Save single recommendation manually
    # -----------------------------------------
    def save_single_recommendation(self, course_id: int, query: str, model: str):
        """
        Saves one course manually when user clicks Save button.
        """
        search_id = self.save_search(query)

        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        # You may want to fetch course info properly later.
        # For now we save minimal info.
        c.execute("""
            INSERT INTO recommendations 
            (search_id, model, course_id, course_title, department, description, relevance)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            search_id,
            model,
            course_id,
            "Saved Course",
            "Unknown",
            "Manually saved recommendation",
            0.0
        ))

        conn.commit()
        conn.close()


    # -----------------------------------------
    # Get saved recommendations (for dashboard)
    # -----------------------------------------
    def get_saved_recommendations(self):
        """
        Returns all recommendations grouped by search.
        """
        return self.get_history()


    # -----------------------------------------
    # Model comparison (optional safe version)
    # -----------------------------------------
    def get_model_comparisons(self):
        """
        Returns comparison structure grouped by search and model.
        """
        history = self.get_history()
        comparisons = []

        for item in history:
            tfidf = []
            neural = []

            for rec in item['recommendations']:
                if rec['model'] == 'tfidf':
                    tfidf.append(rec)
                elif rec['model'] == 'neural':
                    neural.append(rec)

            comparisons.append({
                "search_id": item["search_id"],
                "query": item["query"],
                "timestamp": item["timestamp"],
                "tfidf": tfidf,
                "neural": neural
            })

        return comparisons
