"""
Test suite for the BookRecommender class.

This module contains unit tests for the core book recommendation functionality.
"""

import unittest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from book_recommender import BookRecommender


class TestBookRecommender(unittest.TestCase):
    """Test cases for the BookRecommender class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create sample data for testing
        self.sample_books_data = pd.DataFrame({
            'isbn13': [9781234567890, 9781234567891, 9781234567892],
            'title': ['Test Book 1', 'Test Book 2', 'Test Book 3'],
            'authors': ['Author 1', 'Author 2; Author 3', 'Author 4'],
            'description': [
                'A wonderful story about friendship and adventure.',
                'A thrilling mystery novel with unexpected twists.',
                'A philosophical exploration of life and meaning.'
            ],
            'simple_categories': ['Fiction', 'Fiction', 'Philosophy'],
            'average_rating': [4.5, 4.2, 4.8],
            'num_pages': [300, 250, 400],
            'thumbnail': ['http://example.com/book1.jpg', 'http://example.com/book2.jpg', 'http://example.com/book3.jpg'],
            'joy': [0.8, 0.3, 0.6],
            'sadness': [0.2, 0.1, 0.4],
            'anger': [0.1, 0.2, 0.1],
            'fear': [0.2, 0.7, 0.1],
            'surprise': [0.6, 0.8, 0.3],
            'disgust': [0.1, 0.1, 0.1]
        })
        
        # Mock vector database
        self.mock_vector_db = Mock()
        self.mock_documents = [
            Mock(page_content='9781234567890 A wonderful story about friendship and adventure.'),
            Mock(page_content='9781234567891 A thrilling mystery novel with unexpected twists.'),
            Mock(page_content='9781234567892 A philosophical exploration of life and meaning.')
        ]
    
    @patch('src.book_recommender.pd.read_csv')
    @patch('src.book_recommender.TextLoader')
    @patch('src.book_recommender.CharacterTextSplitter')
    @patch('src.book_recommender.Chroma')
    @patch('src.book_recommender.OpenAIEmbeddings')
    def test_initialization(self, mock_embeddings, mock_chroma, mock_splitter, mock_loader, mock_read_csv):
        """Test BookRecommender initialization."""
        # Mock the data loading
        mock_read_csv.return_value = self.sample_books_data
        mock_loader.return_value.load.return_value = []
        mock_splitter.return_value.split_documents.return_value = []
        mock_chroma.from_documents.return_value = self.mock_vector_db
        
        # Initialize recommender
        recommender = BookRecommender()
        
        # Assertions
        self.assertIsNotNone(recommender.books_df)
        self.assertIsNotNone(recommender.vector_db)
        self.assertEqual(len(recommender.books_df), 3)
    
    @patch('src.book_recommender.pd.read_csv')
    @patch('src.book_recommender.TextLoader')
    @patch('src.book_recommender.CharacterTextSplitter')
    @patch('src.book_recommender.Chroma')
    @patch('src.book_recommender.OpenAIEmbeddings')
    def test_get_available_categories(self, mock_embeddings, mock_chroma, mock_splitter, mock_loader, mock_read_csv):
        """Test getting available categories."""
        # Mock the data loading
        mock_read_csv.return_value = self.sample_books_data
        mock_loader.return_value.load.return_value = []
        mock_splitter.return_value.split_documents.return_value = []
        mock_chroma.from_documents.return_value = self.mock_vector_db
        
        recommender = BookRecommender()
        categories = recommender.get_available_categories()
        
        # Assertions
        self.assertIn('All', categories)
        self.assertIn('Fiction', categories)
        self.assertIn('Philosophy', categories)
        self.assertEqual(len(categories), 4)  # All + 3 unique categories
    
    @patch('src.book_recommender.pd.read_csv')
    @patch('src.book_recommender.TextLoader')
    @patch('src.book_recommender.CharacterTextSplitter')
    @patch('src.book_recommender.Chroma')
    @patch('src.book_recommender.OpenAIEmbeddings')
    def test_get_available_tones(self, mock_embeddings, mock_chroma, mock_splitter, mock_loader, mock_read_csv):
        """Test getting available emotional tones."""
        # Mock the data loading
        mock_read_csv.return_value = self.sample_books_data
        mock_loader.return_value.load.return_value = []
        mock_splitter.return_value.split_documents.return_value = []
        mock_chroma.from_documents.return_value = self.mock_vector_db
        
        recommender = BookRecommender()
        tones = recommender.get_available_tones()
        
        # Assertions
        expected_tones = ["All", "Happy", "Surprising", "Angry", "Suspenseful", "Sad"]
        self.assertEqual(tones, expected_tones)
    
    @patch('src.book_recommender.pd.read_csv')
    @patch('src.book_recommender.TextLoader')
    @patch('src.book_recommender.CharacterTextSplitter')
    @patch('src.book_recommender.Chroma')
    @patch('src.book_recommender.OpenAIEmbeddings')
    def test_retrieve_semantic_recommendations(self, mock_embeddings, mock_chroma, mock_splitter, mock_loader, mock_read_csv):
        """Test semantic recommendation retrieval."""
        # Mock the data loading
        mock_read_csv.return_value = self.sample_books_data
        mock_loader.return_value.load.return_value = []
        mock_splitter.return_value.split_documents.return_value = []
        mock_chroma.from_documents.return_value = self.mock_vector_db
        
        # Mock similarity search
        self.mock_vector_db.similarity_search.return_value = self.mock_documents
        
        recommender = BookRecommender()
        recommendations = recommender.retrieve_semantic_recommendations(
            query="A story about friendship",
            final_top_k=2
        )
        
        # Assertions
        self.assertIsInstance(recommendations, pd.DataFrame)
        self.assertLessEqual(len(recommendations), 2)
        self.mock_vector_db.similarity_search.assert_called_once()
    
    @patch('src.book_recommender.pd.read_csv')
    @patch('src.book_recommender.TextLoader')
    @patch('src.book_recommender.CharacterTextSplitter')
    @patch('src.book_recommender.Chroma')
    @patch('src.book_recommender.OpenAIEmbeddings')
    def test_apply_tone_sorting(self, mock_embeddings, mock_chroma, mock_splitter, mock_loader, mock_read_csv):
        """Test emotional tone sorting."""
        # Mock the data loading
        mock_read_csv.return_value = self.sample_books_data
        mock_loader.return_value.load.return_value = []
        mock_splitter.return_value.split_documents.return_value = []
        mock_chroma.from_documents.return_value = self.mock_vector_db
        
        recommender = BookRecommender()
        
        # Test sorting by joy
        sorted_books = recommender._apply_tone_sorting(self.sample_books_data, "Happy")
        
        # Assertions
        self.assertIsInstance(sorted_books, pd.DataFrame)
        # Check if books are sorted by joy in descending order
        joy_scores = sorted_books['joy'].tolist()
        self.assertEqual(joy_scores, sorted(joy_scores, reverse=True))
    
    @patch('src.book_recommender.pd.read_csv')
    @patch('src.book_recommender.TextLoader')
    @patch('src.book_recommender.CharacterTextSplitter')
    @patch('src.book_recommender.Chroma')
    @patch('src.book_recommender.OpenAIEmbeddings')
    def test_format_book_recommendations(self, mock_embeddings, mock_chroma, mock_splitter, mock_loader, mock_read_csv):
        """Test formatting book recommendations."""
        # Mock the data loading
        mock_read_csv.return_value = self.sample_books_data
        mock_loader.return_value.load.return_value = []
        mock_splitter.return_value.split_documents.return_value = []
        mock_chroma.from_documents.return_value = self.mock_vector_db
        
        recommender = BookRecommender()
        formatted_recs = recommender.format_book_recommendations(self.sample_books_data)
        
        # Assertions
        self.assertIsInstance(formatted_recs, list)
        self.assertEqual(len(formatted_recs), 3)
        
        # Check structure of formatted recommendations
        for rec in formatted_recs:
            self.assertIn('image_url', rec)
            self.assertIn('caption', rec)
            self.assertIn('title', rec)
            self.assertIn('authors', rec)
            self.assertIn('description', rec)
    
    @patch('src.book_recommender.pd.read_csv')
    @patch('src.book_recommender.TextLoader')
    @patch('src.book_recommender.CharacterTextSplitter')
    @patch('src.book_recommender.Chroma')
    @patch('src.book_recommender.OpenAIEmbeddings')
    def test_format_authors(self, mock_embeddings, mock_chroma, mock_splitter, mock_loader, mock_read_csv):
        """Test author formatting."""
        # Mock the data loading
        mock_read_csv.return_value = self.sample_books_data
        mock_loader.return_value.load.return_value = []
        mock_splitter.return_value.split_documents.return_value = []
        mock_chroma.from_documents.return_value = self.mock_vector_db
        
        recommender = BookRecommender()
        
        # Test single author
        single_author = recommender._format_authors("Author 1")
        self.assertEqual(single_author, "Author 1")
        
        # Test two authors
        two_authors = recommender._format_authors("Author 1; Author 2")
        self.assertEqual(two_authors, "Author 1 and Author 2")
        
        # Test multiple authors
        multiple_authors = recommender._format_authors("Author 1; Author 2; Author 3")
        self.assertEqual(multiple_authors, "Author 1, Author 2, and Author 3")
        
        # Test NaN/None
        nan_author = recommender._format_authors(None)
        self.assertEqual(nan_author, "Unknown")


if __name__ == '__main__':
    unittest.main()
