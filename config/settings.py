"""
Configuration settings for the Semantic Book Recommender.

This module contains all configuration parameters, constants, and settings
used throughout the application.
"""

import os
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration for ML models."""
    
    # OpenAI Configuration
    openai_model: str = "text-embedding-ada-002"
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    
    # Text Classification
    classification_model: str = "facebook/bart-large-mnli"
    fiction_categories: List[str] = None
    
    # Sentiment Analysis
    emotion_model: str = "j-hartmann/emotion-english-distilroberta-base"
    emotion_labels: List[str] = None
    
    def __post_init__(self):
        if self.fiction_categories is None:
            self.fiction_categories = ["Fiction", "Nonfiction"]
        
        if self.emotion_labels is None:
            self.emotion_labels = [
                "anger", "disgust", "fear", "joy", 
                "sadness", "surprise", "neutral"
            ]


@dataclass
class DataConfig:
    """Configuration for data files and paths."""
    
    # File paths
    books_csv_path: str = "data/books_with_emotions.csv"
    books_cleaned_path: str = "data/books_cleaned.csv"
    books_categories_path: str = "data/books_with_categories.csv"
    tagged_descriptions_path: str = "data/tagged_description.txt"
    cover_not_found_path: str = "data/cover-not-found.png"
    
    # Data processing
    min_description_words: int = 25
    thumbnail_size: str = "w800"
    
    # Vector search
    chunk_size: int = 0
    chunk_overlap: int = 0
    separator: str = "\n"


@dataclass
class UIConfig:
    """Configuration for the user interface."""
    
    # Streamlit settings
    page_title: str = "Semantic Book Recommender"
    page_icon: str = "📚"
    layout: str = "wide"
    
    # Display settings
    default_recommendations: int = 12
    min_recommendations: int = 4
    max_recommendations: int = 20
    cols_per_row: int = 4
    
    # Image settings
    image_width: int = 150
    description_truncate: int = 30
    title_truncate: int = 50


@dataclass
class SearchConfig:
    """Configuration for search parameters."""
    
    # Search parameters
    initial_top_k: int = 50
    final_top_k: int = 16
    
    # Tone mapping
    tone_mapping: Dict[str, str] = None
    
    def __post_init__(self):
        if self.tone_mapping is None:
            self.tone_mapping = {
                "Happy": "joy",
                "Surprising": "surprise",
                "Angry": "anger",
                "Suspenseful": "fear",
                "Sad": "sadness"
            }


# Global configuration instances
MODEL_CONFIG = ModelConfig()
DATA_CONFIG = DataConfig()
UI_CONFIG = UIConfig()
SEARCH_CONFIG = SearchConfig()


# Category mapping for book classification
CATEGORY_MAPPING = {
    'Fiction': "Fiction",
    'Juvenile Fiction': "Children's Fiction",
    'Biography & Autobiography': "Biography",
    'History': "History",
    'Literary Criticism': "Nonfiction",
    'Philosophy': "Philosophy",
    'Religion': "Religion",
    'Comics & Graphic Novels': "Comics",
    'Juvenile Nonfiction': "Children's Nonfiction",
    'Science': "Science",
    'Poetry': "Literary Collections",
    'Literary Collections': "Fiction",
    'Business & Economics': "Business & Economics",
    'Performing Arts': "Art",
    'Art': "Art",
    'Cooking': "Cooking",
    'Travel': "Travel",
    'Psychology': "Psychology",
    'Self-Help': "Self-Help",
    'Health & Fitness': "Health",
    'Body, Mind & Spirit': "Health",
    'Political Science': "Social and Political Science",
    'Social Science': "Social and Political Science",
    'Language Arts & Disciplines': "Art",
    'Drama': "Fiction"
}


def get_config() -> Dict:
    """
    Get all configuration settings as a dictionary.
    
    Returns:
        Dict: Dictionary containing all configuration settings
    """
    return {
        "model": MODEL_CONFIG,
        "data": DATA_CONFIG,
        "ui": UI_CONFIG,
        "search": SEARCH_CONFIG,
        "category_mapping": CATEGORY_MAPPING
    }


def validate_config() -> bool:
    """
    Validate that all required configuration is present.
    
    Returns:
        bool: True if configuration is valid, False otherwise
    """
    if not MODEL_CONFIG.openai_api_key:
        print("Warning: OPENAI_API_KEY not found in environment variables")
        return False
    
    # Check if required files exist
    required_files = [
        DATA_CONFIG.books_csv_path,
        DATA_CONFIG.tagged_descriptions_path
    ]
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"Warning: Required file not found: {file_path}")
            return False
    
    return True
