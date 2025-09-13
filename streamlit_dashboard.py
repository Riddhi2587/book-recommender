"""
Semantic Book Recommender - Streamlit Dashboard

This application provides a semantic book recommendation system using:
- Vector similarity search with OpenAI embeddings
- Text classification for book categorization
- Sentiment analysis for emotional tone filtering
- Streamlit for the web interface

Author: [Your Name]
Date: 2024
"""

import streamlit as st
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os
from typing import List, Dict, Tuple, Optional
import logging

# LangChain imports for vector search
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class BookRecommender:
    """
    A semantic book recommendation system that uses vector similarity search,
    text classification, and sentiment analysis to provide personalized book recommendations.
    """
    
    def __init__(self, books_csv_path: str, tagged_descriptions_path: str):
        """
        Initialize the BookRecommender with data and vector database.
        
        Args:
            books_csv_path (str): Path to the books CSV file with metadata
            tagged_descriptions_path (str): Path to the tagged descriptions text file
        """
        self.books_csv_path = books_csv_path
        self.tagged_descriptions_path = tagged_descriptions_path
        self.books_df = None
        self.vector_db = None
        self.embeddings = None
        
        # Initialize the system
        self._load_data()
        self._setup_vector_database()
    
    def _load_data(self) -> None:
        """Load and preprocess the books dataset."""
        try:
            logger.info("Loading books dataset...")
            self.books_df = pd.read_csv(self.books_csv_path)
            
            # Process thumbnail URLs for better image quality
            self.books_df["large_thumbnail"] = self.books_df["thumbnail"] + "&fife=w800"
            self.books_df["large_thumbnail"] = np.where(
                self.books_df["large_thumbnail"].isna(),
                "cover-not-found.png",
                self.books_df["large_thumbnail"]
            )
            
            logger.info(f"Loaded {len(self.books_df)} books successfully")
            
        except Exception as e:
            logger.error(f"Error loading books data: {e}")
            raise
    
    def _setup_vector_database(self) -> None:
        """Initialize the vector database for semantic search."""
        try:
            logger.info("Setting up vector database...")
            
            # Load and process documents
            raw_documents = TextLoader(self.tagged_descriptions_path).load()
            text_splitter = CharacterTextSplitter(
                separator="\n", 
                chunk_size=1, 
                chunk_overlap=0
            )
            documents = text_splitter.split_documents(raw_documents)
            
            # Create embeddings and vector database
            self.embeddings = OpenAIEmbeddings()
            
            # Check if database exists, if not create it
            if os.path.exists("./chroma_db"):
                # Load existing database
                self.vector_db = Chroma(
                    persist_directory="./chroma_db",
                    embedding_function=self.embeddings
                )
            else:
                # Create new database
                self.vector_db = Chroma.from_documents(
                    documents, 
                    self.embeddings,
                    persist_directory="./chroma_db"
                )
            
            logger.info("Vector database setup complete")
            
        except Exception as e:
            logger.error(f"Error setting up vector database: {e}")
            raise
    
    def get_available_categories(self) -> List[str]:
        """
        Get list of available book categories.
        
        Returns:
            List[str]: List of unique categories
        """
        if self.books_df is None:
            return ["All"]
        
        categories = ["All"] + sorted(self.books_df["simple_categories"].dropna().unique().tolist())
        return categories
    
    def get_available_tones(self) -> List[str]:
        """
        Get list of available emotional tones for filtering.
        
        Returns:
            List[str]: List of emotional tones
        """
        return ["All", "Happy", "Surprising", "Angry", "Suspenseful", "Sad"]
    
    def retrieve_semantic_recommendations(
        self,
        query: str,
        category: Optional[str] = None,
        tone: Optional[str] = None,
        initial_top_k: int = 50,
        final_top_k: int = 16
    ) -> pd.DataFrame:
        """
        Retrieve semantically similar books based on query and filters.
        
        Args:
            query (str): Natural language query describing desired book
            category (str, optional): Book category filter
            tone (str, optional): Emotional tone filter
            initial_top_k (int): Number of initial results from vector search
            final_top_k (int): Final number of recommendations to return
            
        Returns:
            pd.DataFrame: Filtered and sorted book recommendations
        """
        try:
            # Perform semantic search
            recommendations = self.vector_db.similarity_search(query, k=initial_top_k)
            logger.info(f"Found {len(recommendations)} recommendations for query: {query}")
            
            # Extract ISBNs from search results
            book_isbns = [int(rec.page_content.strip('"').split()[0]) for rec in recommendations]
            logger.info(f"Extracted {len(book_isbns)} ISBNs from search results")
            logger.info(f"Book ISBNs: {book_isbns}")
            
            # Filter books by ISBN
            book_recs = self.books_df[self.books_df["isbn13"].isin(book_isbns)].head(initial_top_k)
            logger.info(f"Filtered {len(book_recs)} books by ISBN")
            logger.info(f"Book recommendations: {book_recs}")
            
            # Apply category filter
            if category and category != "All":
                book_recs = book_recs[book_recs["simple_categories"] == category].head(final_top_k)
            else:
                book_recs = book_recs.head(final_top_k)
            
            # Apply emotional tone sorting
            if tone and tone != "All":
                book_recs = self._apply_tone_sorting(book_recs, tone)
            
            return book_recs
            
        except Exception as e:
            logger.error(f"Error retrieving recommendations: {e}")
            return pd.DataFrame()
    
    def _apply_tone_sorting(self, books_df: pd.DataFrame, tone: str) -> pd.DataFrame:
        """
        Sort books by emotional tone scores.
        
        Args:
            books_df (pd.DataFrame): Books to sort
            tone (str): Emotional tone to sort by
            
        Returns:
            pd.DataFrame: Sorted books
        """
        tone_mapping = {
            "Happy": "joy",
            "Surprising": "surprise", 
            "Angry": "anger",
            "Suspenseful": "fear",
            "Sad": "sadness"
        }
        
        if tone in tone_mapping and tone_mapping[tone] in books_df.columns:
            books_df = books_df.sort_values(by=tone_mapping[tone], ascending=False)
        
        return books_df
    
    def format_book_recommendations(self, recommendations: pd.DataFrame) -> List[Dict]:
        """
        Format book recommendations for display.
        
        Args:
            recommendations (pd.DataFrame): Book recommendations
            
        Returns:
            List[Dict]: Formatted recommendations with image URLs and descriptions
        """
        formatted_recs = []
        
        for _, row in recommendations.iterrows():
            # Truncate description for display
            description = row["description"]
            truncated_desc = " ".join(description.split()[:30]) + "..."
            
            # Format authors list
            authors = self._format_authors(row["authors"])
            
            # Create caption
            caption = f"{row['title']} by {authors}: {truncated_desc}"
            
            formatted_recs.append({
                "image_url": row["large_thumbnail"],
                "caption": caption,
                "title": row["title"],
                "authors": authors,
                "description": description,
                "rating": row.get("average_rating", "N/A"),
                "pages": row.get("num_pages", "N/A"),
                "category": row.get("simple_categories", "N/A")
            })
        
        return formatted_recs
    
    def _format_authors(self, authors_str: str) -> str:
        """
        Format authors string for better readability.
        
        Args:
            authors_str (str): Raw authors string
            
        Returns:
            str: Formatted authors string
        """
        if pd.isna(authors_str):
            return "Unknown"
        
        authors_split = authors_str.split(";")
        
        if len(authors_split) == 1:
            return authors_split[0]
        elif len(authors_split) == 2:
            return f"{authors_split[0]} and {authors_split[1]}"
        else:
            return f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'recommender' not in st.session_state:
        st.session_state.recommender = None
    if 'recommendations' not in st.session_state:
        st.session_state.recommendations = []


def load_recommender():
    """Load the book recommender system."""
    try:
        if st.session_state.recommender is None:
            with st.spinner("Loading book recommendation system..."):
                st.session_state.recommender = BookRecommender(
                    books_csv_path="data/books_with_emotions.csv",
                    tagged_descriptions_path="data/tagged_description.txt"
                )
        return st.session_state.recommender
    except Exception as e:
        st.error(f"Error loading recommender: {e}")
        return None


def display_book_card(book: Dict, col):
    """
    Display a single book recommendation card.
    
    Args:
        book (Dict): Book information dictionary
        col: Streamlit column object
    """
    with col:
        st.image(
            book["image_url"], 
            width=150,
            caption=book["title"][:50] + "..." if len(book["title"]) > 50 else book["title"]
        )
        
        # Book details in expandable section
        with st.expander(f"📖 {book['title'][:30]}..."):
            st.write(f"**Authors:** {book['authors']}")
            st.write(f"**Category:** {book['category']}")
            st.write(f"**Rating:** {book['rating']}")
            st.write(f"**Pages:** {book['pages']}")
            st.write(f"**Description:** {book['description']}")


def main():
    """Main Streamlit application."""
    # Page configuration
    st.set_page_config(
        page_title="Semantic Book Recommender",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.title("�� Semantic Book Recommender")
    st.markdown("""
    Discover your next favorite book using AI-powered semantic search! 
    Describe what you're looking for in natural language, and our system will find 
    books that match your interests using advanced machine learning techniques.
    """)
    
    # Load recommender system
    recommender = load_recommender()
    
    if recommender is None:
        st.error("Failed to load the recommendation system. Please check your data files and API keys.")
        return
    
    # Sidebar for filters
    st.sidebar.header("🔍 Search Filters")
    
    # Search query
    query = st.sidebar.text_area(
        "Describe the book you're looking for:",
        placeholder="e.g., A story about forgiveness and redemption",
        height=100
    )
    
    # Category filter
    categories = recommender.get_available_categories()
    selected_category = st.sidebar.selectbox(
        "📂 Category:",
        categories,
        index=0
    )
    
    # Tone filter
    tones = recommender.get_available_tones()
    selected_tone = st.sidebar.selectbox(
        "🎭 Emotional Tone:",
        tones,
        index=0
    )
    
    # Number of recommendations
    num_recommendations = st.sidebar.slider(
        "📊 Number of recommendations:",
        min_value=4,
        max_value=20,
        value=12,
        step=2
    )
    
    # Search button
    search_button = st.sidebar.button("🔍 Find Recommendations", type="primary")
    
    # Main content area
    if search_button and query.strip():
        with st.spinner("Searching for recommendations..."):
            try:
                # Get recommendations
                recommendations_df = recommender.retrieve_semantic_recommendations(
                    query=query,
                    category=selected_category,
                    tone=selected_tone,
                    final_top_k=num_recommendations
                )
                
                if recommendations_df.empty:
                    st.warning("No books found matching your criteria. Try adjusting your search terms or filters.")
                else:
                    # Format recommendations
                    formatted_recs = recommender.format_book_recommendations(recommendations_df)
                    st.session_state.recommendations = formatted_recs
                    
                    # Display results
                    st.success(f"Found {len(formatted_recs)} book recommendations!")
                    
                    # Display books in a grid
                    cols_per_row = 4
                    for i in range(0, len(formatted_recs), cols_per_row):
                        cols = st.columns(cols_per_row)
                        for j, col in enumerate(cols):
                            if i + j < len(formatted_recs):
                                display_book_card(formatted_recs[i + j], col)
                    
            except Exception as e:
                st.error(f"Error getting recommendations: {e}")
                logger.error(f"Recommendation error: {e}")
    
    elif search_button and not query.strip():
        st.warning("Please enter a search query to find book recommendations.")
    
    # Display previous recommendations if available
    elif st.session_state.recommendations:
        st.subheader("📖 Previous Recommendations")
        formatted_recs = st.session_state.recommendations
        
        # Display books in a grid
        cols_per_row = 4
        for i in range(0, len(formatted_recs), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, col in enumerate(cols):
                if i + j < len(formatted_recs):
                    display_book_card(formatted_recs[i + j], col)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    ### 🛠️ How it works:
    1. **Semantic Search**: Uses OpenAI embeddings to understand the meaning of your query
    2. **Text Classification**: Categorizes books using zero-shot classification
    3. **Sentiment Analysis**: Analyzes emotional tone using transformer models
    4. **Vector Database**: ChromaDB stores and searches book embeddings efficiently
    
    ### 📊 Features:
    - Natural language book search
    - Category-based filtering
    - Emotional tone analysis
    - High-quality book cover images
    - Detailed book information
    """)


if __name__ == "__main__":
    main()
