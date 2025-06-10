---
title: Semantic Book Recommender
emoji: 📚
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 5.33.1
app_file: app.py
pinned: false
license: mit
---

# Semantic Book Recommender 📚

A smart book recommendation system that uses semantic search and emotional tone analysis to help users discover their next favorite read. Built with LangChain, ChromaDB, and Gradio for an intuitive web interface.

## Features

- **Semantic Search**: Uses advanced sentence transformers to understand the meaning behind your book preferences
- **Category Filtering**: Browse recommendations by specific book categories
- **Emotional Tone Matching**: Find books that match your desired emotional experience (Happy, Surprising, Angry, Suspenseful, Sad)
- **Visual Gallery**: Browse recommendations with book covers and detailed descriptions
- **Fast Performance**: Optimized vector database for quick retrieval

## Demo

![Book Recommender Interface](demo.png)

Simply describe what you're looking for, select your preferred category and emotional tone, and get personalized book recommendations!

## Installation

### Prerequisites

- Python 3.10+
- pip package manager

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/nonsodev/semantic-book-recommender.git)
   cd semantic-book-recommender
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure data files are present**
   - `final_book_df.csv`: Main book dataset with metadata
   - `chroma_books/`: ChromaDB vector database directory
   - `cover-not-found.jpg`: Placeholder image for missing book covers

## Usage

### Running the Application

```bash
python gradio_dashboard.py
```

The application will launch a web interface (typically at `http://localhost:7860`) where you can:

1. Enter a description of your ideal book
2. Select a category (optional)
3. Choose an emotional tone (optional)
4. Click "Submit" to get recommendations

### Example Queries

- "A thrilling mystery set in Victorian London"
- "Romantic comedy with strong female protagonist"
- "Science fiction about artificial intelligence"
- "Historical fiction during World War II"

## Project Structure

```
semantic-book-recommender/
├── gradio_dashboard.py          # Main application file
├── requirements.txt             # Python dependencies
├── final_book_df.csv           # Book dataset
├── cover-not-found.jpg         # Placeholder image
├── chroma_books/               # Vector database
├── notebooks/
│   ├── data-exploration.ipynb  # Data analysis
│   ├── download_url.ipynb      # Data download utilities
│   ├── final_df.ipynb          # Data processing
│   ├── sentiment_analysis.ipynb # Emotion analysis
│   ├── supervised_clean.py     # Data cleaning
│   └── test_classification.ipynb # Model testing
└── data/
    ├── books_cleaned.csv       # Processed book data
    ├── books_with_categories.csv
    ├── books_with_sentiment.csv
    ├── books_with_urls.csv
    ├── search_progress.csv     # Processing logs
    ├── tagged_description.txt  # Tagged descriptions
    └── to_drop.txt            # Items to exclude
```

## How It Works

### 1. Semantic Search
- Uses `sentence-transformers/all-MiniLM-L6-v2` for fast, high-quality embeddings
- ChromaDB stores and retrieves similar books based on vector similarity
- Initial retrieval of top 50 matches, refined to top 16 recommendations

### 2. Filtering & Ranking
- **Category Filter**: Narrows results to specific genres
- **Emotional Tone**: Ranks books by emotion scores (joy, surprise, anger, fear, sadness)
- **Relevance**: Maintains semantic relevance while applying filters

### 3. User Interface
- Clean, modern design using Gradio's Glass theme
- Gallery view with book covers and descriptions
- Responsive layout for different screen sizes

## Data Sources

The book dataset includes:
- **Metadata**: Title, authors, ISBN, categories, publication info
- **Content**: Descriptions, summaries
- **Visual**: Thumbnail images, large cover images
- **Emotional Scores**: Joy, surprise, anger, fear, sadness ratings

## Configuration

### Embedding Models
You can switch between different embedding models in `gradio_dashboard.py`:

```python
# Fast and good quality (default)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Higher quality, slower
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
```

### Search Parameters
Adjust recommendation parameters:

```python
def retrieve_semantic_recommendations(
    query: str,
    category: str = None,
    tone: str = None,
    initial_top_k: int = 50,    # Initial retrieval count
    final_top_k: int = 16,      # Final recommendation count
)
```

## Development

### Adding New Features

1. **New Emotional Tones**: Add emotion columns to your dataset and update the `tones` list
2. **Additional Filters**: Extend the filtering logic in `retrieve_semantic_recommendations()`
3. **UI Improvements**: Modify the Gradio interface in the `dashboard` block

### Data Processing Pipeline

The project includes several notebooks for data processing:
- Data exploration and cleaning
- Sentiment analysis for emotional scoring
- URL processing for book covers
- Model testing and validation

## Dependencies

Key libraries used:
- **LangChain**: Vector database integration
- **ChromaDB**: Vector storage and similarity search
- **Gradio**: Web interface
- **HuggingFace Transformers**: Sentence embeddings
- **Pandas**: Data manipulation
- **NumPy**: Numerical operations

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Sentence Transformers for powerful embedding models
- ChromaDB for efficient vector storage
- Gradio for making ML interfaces accessible
- The open-source community for book metadata

---

**Happy Reading! 📖✨**
