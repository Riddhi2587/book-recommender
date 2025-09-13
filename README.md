# 📚 Semantic Book Recommender

An intelligent book recommendation system that uses advanced machine learning techniques to provide personalized book suggestions based on natural language queries.

## 🌟 Features

- **Semantic Search**: Uses OpenAI embeddings to understand the meaning of your queries
- **Text Classification**: Automatically categorizes books using zero-shot classification
- **Sentiment Analysis**: Analyzes emotional tone of books for mood-based filtering
- **Vector Database**: Efficient similarity search using ChromaDB
- **Modern Web Interface**: Beautiful Streamlit dashboard with responsive design
- **Comprehensive Filtering**: Filter by category, emotional tone, and more

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **ML Framework**: Transformers, LangChain
- **Vector Database**: ChromaDB
- **Embeddings**: OpenAI GPT embeddings
- **Text Classification**: Facebook BART-large-MNLI
- **Sentiment Analysis**: DistilRoBERTa emotion classifier
- **Data Processing**: Pandas, NumPy

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/llm-semantic-book-recommender.git
   cd llm-semantic-book-recommender
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   Create a `.env` file in the root directory:
   ```bash
   OPENAI_API_KEY=your_openai_api_key_here
   ```

5. **Download the dataset**
   The project uses the Goodreads dataset. You can download it from [Kaggle](https://www.kaggle.com/datasets/jealousleopard/goodreadsbooks) or use the provided sample data.

## 📊 Data Processing Pipeline

The project includes a comprehensive data processing pipeline:

### 1. Data Exploration (`data-exploration.ipynb`)
- Loads and explores the Goodreads dataset
- Handles missing values and data quality issues
- Creates derived features like book age and description length
- Generates correlation analysis and visualizations

### 2. Text Classification (`text-classification.ipynb`)
- Maps complex book categories to simplified categories
- Uses zero-shot classification with BART-large-MNLI
- Classifies books as Fiction/Non-fiction automatically

### 3. Sentiment Analysis (`sentiment-analysis.ipynb`)
- Analyzes emotional tone of book descriptions
- Extracts emotion scores (joy, sadness, anger, fear, surprise, disgust)
- Creates emotion-based filtering capabilities

### 4. Vector Search (`vector-search.ipynb`)
- Creates embeddings for book descriptions using OpenAI
- Builds a vector database with ChromaDB
- Implements semantic similarity search

## 🎯 Usage

### Running the Streamlit App

```bash
streamlit run streamlit_dashboard.py
```

The app will open in your browser at `http://localhost:8501`

### Using the Recommendation System

1. **Enter a natural language query** describing the type of book you're looking for
2. **Select filters** (optional):
   - Category (Fiction, Non-fiction, etc.)
   - Emotional tone (Happy, Sad, Suspenseful, etc.)
   - Number of recommendations
3. **Click "Find Recommendations"** to get personalized suggestions

### Example Queries

- "A story about forgiveness and redemption"
- "Books about artificial intelligence and the future"
- "Mystery novels with strong female protagonists"
- "Philosophical books about the meaning of life"

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   User Query    │───▶│  Streamlit UI    │───▶│ BookRecommender │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
                                                         ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Recommendations│◀───│  Vector Search   │◀───│   ChromaDB      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │  OpenAI Embeddings│
                       └──────────────────┘

## 🔧 Configuration

### Environment Variables

Create a `.env` file with the following variables:

```bash
OPENAI_API_KEY=your_openai_api_key_here
```

### Customization

You can customize the recommendation system by:

- **Adjusting similarity thresholds** in the `BookRecommender` class
- **Adding new emotion categories** in the sentiment analysis
- **Modifying category mappings** in the text classification
- **Changing the UI layout** in the Streamlit app

