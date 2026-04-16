# evaluate.py
# Run this once to evaluate both models: python evaluate.py

from utils.tfidf_model import TfidfRecommender
from utils.neural_model import NeuralRecommender
import json

# ── Sample test queries with relevant keywords ───────────────────
# Each query has a list of keywords we expect to see in good results
TEST_QUERIES = [
    {
        "query": "I want to learn Python for data science",
        "relevant_keywords": ["python", "data science", "data", "machine learning", "programming"]
    },
    {
        "query": "web development with JavaScript and React",
        "relevant_keywords": ["javascript", "react", "web", "frontend", "html", "css"]
    },
    {
        "query": "machine learning and deep learning with neural networks",
        "relevant_keywords": ["machine learning", "deep learning", "neural", "ai", "tensorflow"]
    },
    {
        "query": "business management and leadership skills",
        "relevant_keywords": ["business", "management", "leadership", "strategy", "mba"]
    },
    {
        "query": "photography and video editing",
        "relevant_keywords": ["photography", "photo", "video", "editing", "camera", "creative"]
    },
    {
        "query": "cybersecurity and ethical hacking",
        "relevant_keywords": ["security", "cyber", "hacking", "network", "ethical"]
    },
    {
        "query": "finance and investment banking",
        "relevant_keywords": ["finance", "investment", "banking", "financial", "accounting"]
    },
    {
        "query": "graphic design and UI UX",
        "relevant_keywords": ["design", "graphic", "ui", "ux", "visual", "figma"]
    },
]


def is_relevant(course, keywords):
    """Check if a course result is relevant based on keyword matching."""
    text = (
        course.get("title", "") + " " +
        course.get("department", "") + " " +
        course.get("description", "")
    ).lower()
    return any(kw.lower() in text for kw in keywords)


def precision_at_k(results, keywords, k):
    """Precision@K: how many of top-K results are relevant."""
    top_k = results[:k]
    relevant_count = sum(1 for c in top_k if is_relevant(c, keywords))
    return relevant_count / k if k > 0 else 0


def recall_at_k(results, keywords, k):
    """Recall@K: of all relevant results, how many did we find in top-K."""
    top_k = results[:k]
    total_relevant = sum(1 for c in results if is_relevant(c, keywords))
    found_relevant = sum(1 for c in top_k if is_relevant(c, keywords))
    return found_relevant / total_relevant if total_relevant > 0 else 0


def hit_rate_at_k(results, keywords, k):
    """Hit Rate@K: did we find at least one relevant result in top-K."""
    top_k = results[:k]
    return 1.0 if any(is_relevant(c, keywords) for c in top_k) else 0.0


def evaluate_model(model, model_name, k=10):
    """Run all test queries and compute average metrics."""
    print(f"\n{'='*50}")
    print(f"  Evaluating: {model_name}")
    print(f"{'='*50}")

    precision_scores = []
    recall_scores    = []
    hit_rate_scores  = []

    for item in TEST_QUERIES:
        query    = item["query"]
        keywords = item["relevant_keywords"]

        results = model.recommend(query, top_k=k)

        p = precision_at_k(results, keywords, k)
        r = recall_at_k(results, keywords, k)
        h = hit_rate_at_k(results, keywords, k)

        precision_scores.append(p)
        recall_scores.append(r)
        hit_rate_scores.append(h)

        print(f"\n  Query: \"{query}\"")
        print(f"    Precision@{k}: {p:.2f}  |  Recall@{k}: {r:.2f}  |  Hit Rate: {h:.2f}")
        for i, c in enumerate(results[:3], 1):
            print(f"      {i}. {c['title']} [{c['department']}] — score: {c.get('score', 0):.1f}%")

    avg_p = sum(precision_scores) / len(precision_scores)
    avg_r = sum(recall_scores)    / len(recall_scores)
    avg_h = sum(hit_rate_scores)  / len(hit_rate_scores)

    print(f"\n  {'─'*40}")
    print(f"  AVERAGE METRICS ({model_name})")
    print(f"  {'─'*40}")
    print(f"  Precision@{k} : {avg_p:.4f}  ({avg_p*100:.1f}%)")
    print(f"  Recall@{k}    : {avg_r:.4f}  ({avg_r*100:.1f}%)")
    print(f"  Hit Rate@{k}  : {avg_h:.4f}  ({avg_h*100:.1f}%)")

    return {
        "model": model_name,
        "precision": round(avg_p, 4),
        "recall":    round(avg_r, 4),
        "hit_rate":  round(avg_h, 4)
    }


if __name__ == "__main__":
    print("\n🔍 Loading models...")
    DATA_PATH = "data/courses.csv"
    tfidf_model  = TfidfRecommender(csv_path=DATA_PATH)
    neural_model = NeuralRecommender(csv_path=DATA_PATH)

    # Evaluate both models
    tfidf_metrics  = evaluate_model(tfidf_model,  "TF-IDF",  k=10)
    neural_metrics = evaluate_model(neural_model, "Neural",  k=10)

    # Side-by-side comparison
    print(f"\n{'='*50}")
    print("  FINAL COMPARISON")
    print(f"{'='*50}")
    print(f"  {'Metric':<15} {'TF-IDF':>10} {'Neural':>10}")
    print(f"  {'─'*35}")
    print(f"  {'Precision@10':<15} {tfidf_metrics['precision']:>10.4f} {neural_metrics['precision']:>10.4f}")
    print(f"  {'Recall@10':<15} {tfidf_metrics['recall']:>10.4f} {neural_metrics['recall']:>10.4f}")
    print(f"  {'Hit Rate@10':<15} {tfidf_metrics['hit_rate']:>10.4f} {neural_metrics['hit_rate']:>10.4f}")

    # Save results to file
    report = {"tfidf": tfidf_metrics, "neural": neural_metrics}
    with open("models/evaluation_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n✅ Report saved to models/evaluation_report.json")