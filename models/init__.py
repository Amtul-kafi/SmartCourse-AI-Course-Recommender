# models/__init__.py

# This file marks the models folder as a Python package.
# It also allows you to import models easily:
# from models import DataLoader, TFIDFRecommender, NeuralRecommender

from .dataloader import DataLoader
from .tfidf_model import TFIDFRecommender
from .neural_model import NeuralRecommender

__all__ = ["DataLoader", "TFIDFRecommender", "NeuralRecommender"]
