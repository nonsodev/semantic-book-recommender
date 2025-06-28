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
# Smart Book Recommender 📚

An intelligent book recommendation system with dual search modes: semantic understanding and flexible literal matching. Features emotional tone analysis, category filtering, and a responsive web interface built with LangChain, ChromaDB, and Gradio.

## 🚀 [Try the Live Demo](https://huggingface.co/spaces/nonsodev/semantic-book-recommender)

![Book Recommender Interface](demo.png)

## ✨ Key Features

### 🔍 **Dual Search Modes**
- **Semantic Search**: AI-powered understanding of natural language queries (e.g., "fantasy adventure with magic")
- **Literal Search**: Flexible keyword matching with partial word support (e.g., "harry" → Harry Potter books)

### 🎯 **Smart Filtering**
- **Category Filtering**: Browse by specific book genres
- **Emotional Tone Matching**: Find books by emotional experience (Happy, Surprising, Angry, Suspenseful, Sad)
- **Intelligent Sorting**: Results ranked by relevance and emotional scores

### 🎨 **Modern Interface**
- Responsive card-based design with book covers
- Star ratings and reader statistics
- Direct download links when available
- Dark theme optimized for reading

### ⚡ **Performance Optimized**
- Cached embedding models for fast startup
- Efficient ChromaDB vector database
- Fallback image handling for missing covers
- Robust error handling and regex search

## Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Quick Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/nonsodev/semantic-book-recommender.git
   cd semantic-book-recommender
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure required data files**
   ```
   ├── final_book_df.csv          # Main book dataset
   ├── tagged_description.txt     # Book descriptions for embedding
   └── chroma_books/             # Vector database (auto-created)
   ```

4. **Run the application**
   ```bash
   python app.py
   ```

## Usage Guide

### Search Modes

#### 🧠 **Semantic Search**
Perfect for describing what you want in natural language:
- "Dark fantasy with dragons and magic"
- "Romantic comedy set in Paris"
- "Thrilling mystery in Victorian London"
- "Science fiction about artificial intelligence"

#### 🔤 **Literal Search**
Best for finding specific titles or authors:
- "harry" → finds Harry Potter books
- "tolkien" → finds J.R.R. Tolkien works
- "game thrones" → finds Game of Thrones
- "stephen king" → finds Stephen King novels

### Advanced Features

#### **Category Filtering**
Narrow results by genre:
- Fiction, Non-fiction, Fantasy, Romance, Mystery, etc.

#### **Emotional Tone Matching**
Find books by mood:
- **Happy**: High joy scores
- **Surprising**: High surprise scores  
- **Angry**: High anger scores
- **Suspenseful**: High fear scores
- **Sad**: High sadness scores

## How It Works

### 🔬 **Semantic Search Engine**
```python
# Uses sentence-transformers for embedding generation
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# ChromaDB for efficient similarity search
db_books = Chroma.from_documents(
    documents, embedding=embeddings,
    collection_name="books", persist_directory="chroma_books"
)
```

### 🔍 **Flexible Literal Search**
```python
# Intelligent regex pattern matching
def retrieve_literal_recommendations(query, category=None, tone=None):
    # Creates flexible patterns for partial word matching
    # Handles special characters and multiple word combinations
    # Falls back to simple string matching if regex fails
```

### 🎭 **Emotional Intelligence**
Books are analyzed and scored across five emotional dimensions:
- **Joy**: Happiness, humor, uplifting content
- **Surprise**: Plot twists, unexpected elements
- **Anger**: Conflict, tension, dramatic intensity  
- **Fear**: Suspense, thriller elements, mystery
- **Sadness**: Emotional depth, tragic elements

### 🎨 **Smart UI Components**
```python
def create_book_card_html(row):
    # Responsive card design with:
    # - Book cover with fallback handling
    # - Star ratings visualization  
    # - Author formatting (handles multiple authors)
    # - Truncated descriptions with full content
    # - Download links when available
```

## Project Structure

```
semantic-book-recommender/
├── app.py                      # Main application (your updated file)
├── requirements.txt            # Python dependencies
├── final_book_df.csv          # Book dataset with metadata
├── tagged_description.txt     # Book descriptions for embedding
├── chroma_books/              # ChromaDB vector database
├── demo.png                   # Interface screenshot
└── README.md                  # This file
```

## Configuration

### **Embedding Models**
Switch between models for different performance profiles:

```python
# Fast and efficient (default)
"sentence-transformers/all-MiniLM-L6-v2"

# Higher quality, slower
"sentence-transformers/all-mpnet-base-v2"  

# Multilingual support
"sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
```

### **Search Parameters**
Customize recommendation behavior:

```python
def retrieve_semantic_recommendations(
    query: str,
    initial_top_k: int = 50,    # Initial retrieval size
    final_top_k: int = 8,       # Final recommendations shown
    category: str = None,       # Category filter
    tone: str = None           # Emotional tone filter
)
```

### **UI Customization**
Modify card display and styling:

```python
# Book card dimensions
style="width: 80px; height: 120px"

# Description truncation
-webkit-line-clamp: 4

# Rating display
create_star_rating(rating)  # ★★★★☆ format
```

## Data Schema

### Book Dataset Columns
```python
# Core metadata
'isbn13', 'title_and_subtitle', 'authors', 'categories'

# Visual elements  
'thumbnail', 'large_thumbnail'

# Ratings and metrics
'average_rating', 'ratings_count'

# Content
'description'

# Emotional scores
'joy', 'surprise', 'anger', 'fear', 'sadness'

# Access
'url'  # Download/purchase links
```

## API Reference

### **Main Functions**

```python
# Semantic search with AI understanding
retrieve_semantic_recommendations(query, category, tone, initial_top_k, final_top_k)

# Literal search with flexible matching  
retrieve_literal_recommendations(query, category, tone, final_top_k)

# HTML card generation
create_book_card_html(row)

# Main Gradio interface function
recommend_books(query, category, tone, search_type)
```

## Dependencies

```python
# Core ML and Vector Database
langchain-chroma>=0.1.0
langchain-huggingface>=0.0.3  
langchain-community>=0.2.0
sentence-transformers>=2.2.0

# Data Processing
pandas>=1.5.0
numpy>=1.21.0

# Web Interface
gradio>=4.0.0

# Text Processing  
regex>=2022.0.0
```

## Performance Tips

### **Startup Optimization**
```python
# Model caching for faster restarts
os.environ["HF_HOME"] = "/tmp/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/tmp/hf_cache"
```

### **Search Optimization**
- Use semantic search for exploratory queries
- Use literal search for known titles/authors
- Combine category and tone filters for precision
- Try variations if initial results aren't satisfactory

### **Memory Management**
- ChromaDB persists to disk automatically
- Embeddings cached after first load
- Efficient pandas operations for filtering

## Contributing

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** changes (`git commit -m 'Add amazing feature'`)
4. **Push** to branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Development Areas
- [ ] Additional emotional dimensions
- [ ] Multi-language support
- [ ] User preference learning
- [ ] Social features (reviews, ratings)
- [ ] Advanced filtering (publication year, page count)

## Troubleshooting

### **Common Issues**

**ChromaDB not found:**
```bash
# The app will auto-create from tagged_description.txt
# Ensure this file exists in the project root
```

**Model download slow:**
```bash
# Models cache automatically after first download
# Subsequent starts will be much faster
```

**No search results:**
```bash
# Try switching between search modes
# Reduce filter constraints (category/tone)
# Use broader search terms
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **Sentence Transformers** for powerful embedding models
- **ChromaDB** for efficient vector storage and retrieval
- **Gradio** for creating accessible ML interfaces
- **LangChain** for seamless AI integration
- **HuggingFace** for model hosting and ecosystem

---

## 🎯 Example Queries to Try

### Semantic Search
- "Epic fantasy with complex magic systems"
- "Cozy mystery in a small town setting"  
- "Hard science fiction about space exploration"
- "Historical romance during the Regency era"

### Literal Search
- "agatha christie" (find Agatha Christie novels)
- "dune" (find Dune series books)
- "pride prejudice" (find Pride and Prejudice)
- "lord rings" (find Lord of the Rings)

**Happy Reading! 📖✨**