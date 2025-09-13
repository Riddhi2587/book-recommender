"""
Core book recommendation system implementation.

This module contains the main BookRecommender class and related functionality
for semantic book recommendations using vector similarity search.
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional
from pathlib import Path

# LangChain imports
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma

# Local imports
from config.settings import MODEL_CONFIG, DATA_CONFIG, SEARCH_CONFIG, CATEGORY_MAPPING

# Configure logging
logger = logging.getLogger(__name__)


class BookRecommender:
    """
    A semantic book recommendation system that uses vector similarity search,
    text classification, and sentiment analysis to provide personalized book recommendations.
    
    This class handles the core functionality of loading data, setting up vector databases,
    and performing semantic searches to find relevant book recommendations.
    """
    
    def __init__(self, books_csv_path: Optional[str] = None, tagged_descriptions_path: Optional[str] = None):
        """
        Initialize the BookRecommender with data and vector database.
        
        Args:
            books_csv_path (str, optional): Path to the books CSV file with metadata.
                Defaults to DATA_CONFIG.books_csv_path.
            tagged_descriptions_path (str, optional): Path to the tagged descriptions text file.
                Defaults to DATA_CONFIG.tagged_descriptions_path.
        
        Raises:
            FileNotFoundError: If required data files are not found.
            ValueError: If OpenAI API key is not configured.
        """
        self.books_csv_path = books_csv_path or DATA_CONFIG.books_csv_path
        self.tagged_descriptions_path = tagged_descriptions_path or DATA_CONFIG.tagged_descriptions_path
        
        # Validate configuration
        if not MODEL_CONFIG.openai_api_key:
            raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
        
        # Initialize attributes
        self.books_df: Optional[pd.DataFrame] = None
        self.vector_db: Optional[Chroma] = None
        self.embeddings: Optional[OpenAIEmbeddings] = None
        
        # Initialize the system
        self._load_data()
        self._setup_vector_database()
    
    def _load_data(self) -> None:
        """
        Load and preprocess the books dataset.
        
        Raises:
            FileNotFoundError: If the books CSV file is not found.
            pd.errors.EmptyDataError: If the CSV file is empty.
        """
        try:
            logger.info(f"Loading books dataset from {self.books_csv_path}...")
            self.books_df = pd.read_csv(self.books_csv_path)
            
            # Process thumbnail URLs for better image quality
            self._process_thumbnails()
            
            # Apply category mapping if simple_categories column exists
            self._apply_category_mapping()
            
            logger.info(f"Loaded {len(self.books_df)} books successfully")
            
        except FileNotFoundError:
            logger.error(f"Books CSV file not found: {self.books_csv_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading books data: {e}")
            raise
    
    def _process_thumbnails(self) -> None:
        """Process thumbnail URLs to use higher quality images."""
        if "thumbnail" in self.books_df.columns:
            self.books_df["large_thumbnail"] = self.books_df["thumbnail"] + f"&fife={DATA_CONFIG.thumbnail_size}"
            self.books_df["large_thumbnail"] = np.where(
                self.books_df["large_thumbnail"].isna(),
                DATA_CONFIG.cover_not_found_path,
                self.books_df["large_thumbnail"]
            )
        else:
            logger.warning("Thumbnail column not found in dataset")
    
    def _apply_category_mapping(self) -> None:
        """Apply category mapping to simplify book categories."""
        if "categories" in self.books_df.columns and "simple_categories" not in self.books_df.columns:
            self.books_df["simple_categories"] = self.books_df["categories"].map(CATEGORY_MAPPING)
            logger.info("Applied category mapping to books dataset")
    
    def _setup_vector_database(self) -> None:
        """
        Initialize the vector database for semantic search.
        
        Raises:
            FileNotFoundError: If the tagged descriptions file is not found.
            Exception: If there's an error setting up the vector database.
        """
        try:
            logger.info("Setting up vector database...")
            
            # Load and process documents
            raw_documents = TextLoader(self.tagged_descriptions_path).load()
            text_splitter = CharacterTextSplitter(
                separator=DATA_CONFIG.separator,
                chunk_size=DATA_CONFIG.chunk_size,
                chunk_overlap=DATA_CONFIG.chunk_overlap
            )
            documents = text_splitter.split_documents(raw_documents)
            
            # Create embeddings and vector database
            self.embeddings = OpenAIEmbeddings(
                model=MODEL_CONFIG.openai_model,
                openai_api_key=MODEL_CONFIG.openai_api_key
            )
            self.vector_db = Chroma.from_documents(documents, self.embeddings)
            
            logger.info("Vector database setup complete")
            
        except FileNotFoundError:
            logger.error(f"Tagged descriptions file not found: {self.tagged_descriptions_path}")
            raise
        except Exception as e:
            logger.error(f"Error setting up vector database: {e}")
            raise
    
    def get_available_categories(self) -> List[str]:
        """
        Get list of available book categories.
        
        Returns:
            List[str]: List of unique categories, with "All" as the first option.
        """
        if self.books_df is None or "simple_categories" not in self.books_df.columns:
            return ["All"]
        
        categories = ["All"] + sorted(self.books_df["simple_categories"].dropna().unique().tolist())
        return categories
    
    def get_available_tones(self) -> List[str]:
        """
        Get list of available emotional tones for filtering.
        
        Returns:
            List[str]: List of emotional tones, with "All" as the first option.
        """
        return ["All", "Happy", "Surprising", "Angry", "Suspenseful", "Sad"]
    
    def retrieve_semantic_recommendations(
        self,
        query: str,
        category: Optional[str] = None,
        tone: Optional[str] = None,
        initial_top_k: Optional[int] = None,
        final_top_k: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Retrieve semantically similar books based on query and filters.
        
        Args:
            query (str): Natural language query describing desired book.
            category (str, optional): Book category filter. Defaults to None.
            tone (str, optional): Emotional tone filter. Defaults to None.
            initial_top_k (int, optional): Number of initial results from vector search.
                Defaults to SEARCH_CONFIG.initial_top_k.
            final_top_k (int, optional): Final number of recommendations to return.
                Defaults to SEARCH_CONFIG.final_top_k.
        
        Returns:
            pd.DataFrame: Filtered and sorted book recommendations.
        
        Raises:
            ValueError: If query is empty or invalid.
            Exception: If there's an error during the search process.
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        initial_top_k = initial_top_k or SEARCH_CONFIG.initial_top_k
        final_top_k = final_top_k or SEARCH_CONFIG.final_top_k
        
        try:
            logger.info(f"Searching for recommendations with query: '{query[:50]}...'")
            
            # Perform semantic search
            recommendations = self.vector_db.similarity_search(query, k=initial_top_k)
            
            # Extract ISBNs from search results
            book_isbns = self._extract_isbns_from_recommendations(recommendations)
            
            # Filter books by ISBN
            book_recs = self.books_df[self.books_df["isbn13"].isin(book_isbns)].head(initial_top_k)
            
            # Apply category filter
            if category and category != "All":
                book_recs = self._apply_category_filter(book_recs, category, final_top_k)
            else:
                book_recs = book_recs.head(final_top_k)
            
            # Apply emotional tone sorting
            if tone and tone != "All":
                book_recs = self._apply_tone_sorting(book_recs, tone)
            
            logger.info(f"Found {len(book_recs)} recommendations")
            return book_recs
            
        except Exception as e:
            logger.error(f"Error retrieving recommendations: {e}")
            raise
    
    def _extract_isbns_from_recommendations(self, recommendations: List) -> List[int]:
        """
        Extract ISBNs from vector search recommendations.
        
        Args:
            recommendations (List): List of document recommendations from vector search.
        
        Returns:
            List[int]: List of extracted ISBNs.
        """
        book_isbns = []
        for rec in recommendations:
            try:
                isbn = int(rec.page_content.strip('"').split()[0])
                book_isbns.append(isbn)
            except (ValueError, IndexError) as e:
                logger.warning(f"Could not extract ISBN from recommendation: {e}")
                continue
        return book_isbns
    
    def _apply_category_filter(self, books_df: pd.DataFrame, category: str, final_top_k: int) -> pd.DataFrame:
        """
        Apply category filter to book recommendations.
        
        Args:
            books_df (pd.DataFrame): Books to filter.
            category (str): Category to filter by.
            final_top_k (int): Maximum number of results to return.
        
        Returns:
            pd.DataFrame: Filtered books.
        """
        if "simple_categories" not in books_df.columns:
            logger.warning("simple_categories column not found, skipping category filter")
            return books_df.head(final_top_k)
        
        filtered_books = books_df[books_df["simple_categories"] == category].head(final_top_k)
        logger.info(f"Applied category filter '{category}', found {len(filtered_books)} books")
        return filtered_books
    
    def _apply_tone_sorting(self, books_df: pd.DataFrame, tone: str) -> pd.DataFrame:
        """
        Sort books by emotional tone scores.
        
        Args:
            books_df (pd.DataFrame): Books to sort.
            tone (str): Emotional tone to sort by.
        
        Returns:
            pd.DataFrame: Sorted books.
        """
        if tone not in SEARCH_CONFIG.tone_mapping:
            logger.warning(f"Unknown tone: {tone}")
            return books_df
        
        emotion_column = SEARCH_CONFIG.tone_mapping[tone]
        
        if emotion_column not in books_df.columns:
            logger.warning(f"Emotion column '{emotion_column}' not found, skipping tone sorting")
            return books_df
        
        sorted_books = books_df.sort_values(by=emotion_column, ascending=False)
        logger.info(f"Applied tone sorting for '{tone}' (column: {emotion_column})")
        return sorted_books
    
    def format_book_recommendations(self, recommendations: pd.DataFrame) -> List[Dict]:
        """
        Format book recommendations for display.
        
        Args:
            recommendations (pd.DataFrame): Book recommendations.
        
        Returns:
            List[Dict]: Formatted recommendations with image URLs and descriptions.
        """
        formatted_recs = []
        
        for _, row in recommendations.iterrows():
            # Truncate description for display
            description = str(row.get("description", ""))
            truncated_desc = self._truncate_text(description, DATA_CONFIG.min_description_words)
            
            # Format authors list
            authors = self._format_authors(row.get("authors", ""))
            
            # Create formatted recommendation
            formatted_rec = {
                "image_url": row.get("large_thumbnail", DATA_CONFIG.cover_not_found_path),
                "caption": f"{row.get('title', 'Unknown Title')} by {authors}: {truncated_desc}",
                "title": row.get("title", "Unknown Title"),
                "authors": authors,
                "description": description,
                "rating": row.get("average_rating", "N/A"),
                "pages": row.get("num_pages", "N/A"),
                "category": row.get("simple_categories", "N/A"),
                "isbn": row.get("isbn13", "N/A")
            }
            
            formatted_recs.append(formatted_rec)
        
        return formatted_recs
    
    def _truncate_text(self, text: str, max_words: int) -> str:
        """
        Truncate text to a maximum number of words.
        
        Args:
            text (str): Text to truncate.
            max_words (int): Maximum number of words.
        
        Returns:
            str: Truncated text with ellipsis if needed.
        """
        if not text or pd.isna(text):
            return "No description available"
        
        words = str(text).split()
        if len(words) <= max_words:
            return text
        
        return " ".join(words[:max_words]) + "..."
    
    def _format_authors(self, authors_str: str) -> str:
        """
        Format authors string for better readability.
        
        Args:
            authors_str (str): Raw authors string.
        
        Returns:
            str: Formatted authors string.
        """
        if pd.isna(authors_str) or not authors_str:
            return "Unknown"
        
        authors_split = str(authors_str).split(";")
        
        if len(authors_split) == 1:
            return authors_split[0].strip()
        elif len(authors_split) == 2:
            return f"{authors_split[0].strip()} and {authors_split[1].strip()}"
        else:
            formatted_authors = [author.strip() for author in authors_split[:-1]]
            return f"{', '.join(formatted_authors)}, and {authors_split[-1].strip()}"
    
    def get_dataset_info(self) -> Dict:
        """
        Get information about the loaded dataset.
        
        Returns:
            Dict: Dictionary containing dataset statistics and information.
        """
        if self.books_df is None:
            return {"error": "No dataset loaded"}
        
        info = {
            "total_books": len(self.books_df),
            "columns": list(self.books_df.columns),
            "categories": self.get_available_categories()[1:],  # Exclude "All"
            "has_emotions": any(col in self.books_df.columns for col in SEARCH_CONFIG.tone_mapping.values()),
            "has_thumbnails": "large_thumbnail" in self.books_df.columns,
            "memory_usage": f"{self.books_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
        }
        
        return info
