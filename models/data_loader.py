# models/dataloader.py
import pandas as pd
import os

class DataLoader:
    def __init__(self, csv_path="data/courses.csv"):
        """
        Load courses dataset from CSV
        csv_path: path to courses.csv file
        """
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found at {csv_path}")

        self.df = pd.read_csv(csv_path)

        # Ensure required columns exist
        required_columns = ['course_id', 'title', 'department', 'description']
        for col in required_columns:
            if col not in self.df.columns:
                raise ValueError(f"Missing required column: {col}")

        # Fill missing descriptions with empty string
        self.df['description'] = self.df['description'].fillna("")

    def get_all_courses(self):
        """Return all courses as list of dicts"""
        return self.df.to_dict(orient='records')

    def get_top_courses(self, top_n=10):
        """Return top N courses (for prototype purposes)"""
        # Here we just take the first top_n courses
        return self.df.head(top_n).to_dict(orient='records')

if __name__ == "__main__":
    loader = DataLoader()
    print(f"Total courses loaded: {len(loader.df)}")
    print("Top 10 courses preview:")
    print(loader.get_top_courses())
