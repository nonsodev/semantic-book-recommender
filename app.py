import pandas as pd
import gradio as gr
import numpy as np
import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter

# Ensure model caching
os.environ["HF_HOME"] = "/tmp/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/tmp/hf_cache"

# Initialize embeddings with caching
print("Loading embeddings model...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# Initialize ChromaDB
print("Initializing ChromaDB...")
if not os.path.exists("chroma_books"):
    print("Creating new ChromaDB from tagged_description.txt...")
    try:
        raw_docs = TextLoader("tagged_description.txt", encoding="utf-8").load()
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=0,
            chunk_overlap=0,
            length_function=len,
        )
        documents = text_splitter.split_documents(raw_docs)
        print(f"Loaded {len(documents)} documents")
        
        db_books = Chroma.from_documents(
            documents,
            embedding=embeddings,
            collection_name="books",
            persist_directory="chroma_books",
        )
        print("ChromaDB created successfully!")
    except FileNotFoundError:
        print("ERROR: tagged_description.txt not found!")
        raise
else:
    print("Loading existing ChromaDB...")
    db_books = Chroma(
        persist_directory="chroma_books", 
        embedding_function=embeddings, 
        collection_name="books"
    )

# Load books data
print("Loading books data...")
try:
    books = pd.read_csv("final_book_df.csv")
    books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
    books["large_thumbnail"] = np.where(
        books["large_thumbnail"].isna(), 
        "cover-not-found.jpg", 
        books["large_thumbnail"]
    )
    print(f"Loaded {len(books)} books")
except FileNotFoundError:
    print("ERROR: final_book_df.csv not found!")
    raise

def retrieve_semantic_recommendations(
        query: str,
        category: str = None,
        tone: str = None,
        initial_top_k: int = 50,
        final_top_k: int = 8,  # Reduced for better display
) -> pd.DataFrame:
    """Retrieve semantic recommendations based on query, category, and tone."""
    
    recs = db_books.similarity_search(query, k=initial_top_k)
    books_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs]
    book_recs = books[books["isbn13"].isin(books_list)].head(initial_top_k)

    # Filter by category
    if category and category != "All":
        book_recs = book_recs[book_recs["categories"] == category].head(final_top_k)
    else:
        book_recs = book_recs.head(final_top_k)

    # Sort by emotional tone
    if tone == "Happy":
        book_recs = book_recs.sort_values(by="joy", ascending=False)
    elif tone == "Surprising":
        book_recs = book_recs.sort_values(by="surprise", ascending=False)
    elif tone == "Angry":
        book_recs = book_recs.sort_values(by="anger", ascending=False)
    elif tone == "Suspenseful":
        book_recs = book_recs.sort_values(by="fear", ascending=False)
    elif tone == "Sad":
        book_recs = book_recs.sort_values(by="sadness", ascending=False)

    return book_recs

def create_book_card_html(row):
    """Create an HTML card for a single book with full description, ratings, and download link."""
    
    # Handle missing description
    description = row.get("description", "No description available")
    if pd.isna(description):
        description = "No description available"
    
    # Format authors
    authors = row.get("authors", "Unknown Author")
    if pd.isna(authors):
        authors_str = "Unknown Author"
    else:
        authors_split = str(authors).split(";")
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = authors
    
    # Get other info
    title = row.get("title_and_subtitle", "Unknown Title")
    thumbnail = row.get("large_thumbnail", "cover-not-found.jpg")
    download_url = row.get("url", "")
    category = row.get("categories", "Unknown")
    
    # Handle ratings
    average_rating = row.get("average_rating", 0)
    ratings_count = row.get("ratings_count", 0)
    
    # Convert to proper numeric values
    try:
        avg_rating = float(average_rating) if not pd.isna(average_rating) else 0
        rating_count = int(ratings_count) if not pd.isna(ratings_count) else 0
    except (ValueError, TypeError):
        avg_rating = 0
        rating_count = 0
    
    # Create star rating display
    def create_star_rating(rating):
        """Create HTML for star rating display."""
        full_stars = int(rating)
        half_star = 1 if (rating - full_stars) >= 0.5 else 0
        empty_stars = 5 - full_stars - half_star
        
        stars_html = ""
        # Full stars
        stars_html += "★" * full_stars
        # Half star
        if half_star:
            stars_html += "☆"
        # Empty stars
        stars_html += "☆" * empty_stars
        
        return stars_html
    
    # Format rating display
    if avg_rating > 0:
        stars = create_star_rating(avg_rating)
        rating_display = f"""
        <div style="margin: 2px 0; display: flex; align-items: center; gap: 6px; flex-wrap: wrap;">
            <span style="color: #ffd700; font-size: 12px; letter-spacing: 1px;">{stars}</span>
            <span style="color: #cccccc; font-size: 10px;">
                {avg_rating:.1f} ({rating_count:,})
            </span>
        </div>
        """
    else:
        rating_display = """
        <div style="margin: 2px 0;">
            <span style="color: #888888; font-size: 10px;">No ratings</span>
        </div>
        """
    
    # Create download button if URL exists
    download_button = ""
    if download_url and not pd.isna(download_url) and str(download_url).strip():
        download_button = f"""
        <div style="margin-top: 6px;">
            <a href="{download_url}" target="_blank" 
               style="background-color: #4CAF50; color: white; padding: 6px 12px; 
                      text-decoration: none; border-radius: 4px; font-size: 10px; 
                      display: inline-block; text-align: center;">
                📖 Get Book
            </a>
        </div>
        """
    
    # Create the card HTML with responsive design
    card_html = f"""
    <div style="border: 1px solid #444; border-radius: 8px; padding: 12px; margin: 10px 0; 
                background-color: #2b2b2b; box-shadow: 0 2px 4px rgba(0,0,0,0.3);">
        
        <div style="display: flex; gap: 12px; flex-direction: row;">
            <div style="flex-shrink: 0;">
                <img src="{thumbnail}" alt="Book cover" 
                     style="width: 80px; height: 120px; object-fit: cover; border-radius: 4px;">
            </div>
            
            <div style="flex-grow: 1; min-width: 0; display: flex; flex-direction: column;">
                <h3 style="margin: 0 0 6px 0; color: #ffffff; font-size: 14px; line-height: 1.2; 
                           word-wrap: break-word; overflow-wrap: break-word;">
                    {title}
                </h3>
                
                <p style="margin: 0 0 4px 0; color: #cccccc; font-size: 11px; font-style: italic;">
                    {authors_str}
                </p>
                
                <p style="margin: 0 0 4px 0; color: #aaaaaa; font-size: 10px;">
                    {category}
                </p>
                
                {rating_display}
                
                <div style="flex-grow: 1; margin: 6px 0;">
                    <p style="margin: 0; color: #dddddd; font-size: 11px; line-height: 1.3; 
                              display: -webkit-box; -webkit-line-clamp: 4; -webkit-box-orient: vertical; 
                              overflow: hidden; text-overflow: ellipsis;">
                        {description}
                    </p>
                </div>
                
                {download_button}
            </div>
        </div>
    </div>
    """
    return card_html

def recommend_books(query: str, category: str, tone: str):
    """Main recommendation function for Gradio interface."""
    
    if not query.strip():
        return "<p>Please enter a search query to get book recommendations.</p>"
    
    try:
        recommendations = retrieve_semantic_recommendations(query, category, tone)
        
        if recommendations.empty:
            return "<p>No books found matching your criteria. Try adjusting your search terms or filters.</p>"
        
        # Create HTML for all book cards
        html_cards = []
        for _, row in recommendations.iterrows():
            card_html = create_book_card_html(row)
            html_cards.append(card_html)
        
        # Combine all cards with a header
        full_html = f"""
        <div style="font-family: Arial, sans-serif; background-color: #1a1a1a; padding: 20px; border-radius: 8px;">
            <h2 style="color: #ffffff; margin-bottom: 20px;">
                📚 Found {len(recommendations)} recommendations for: "{query}"
            </h2>
            {''.join(html_cards)}
        </div>
        """
        
        return full_html
    
    except Exception as e:
        print(f"Error in recommend_books: {e}")
        return f"<p>An error occurred while searching for books: {str(e)}</p>"

# Prepare dropdown options
categories = ["All"] + sorted(books["categories"].unique().tolist())
tones = ["All", "Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

# Create Gradio interface
with gr.Blocks(theme=gr.themes.Soft()) as dashboard:
    gr.Markdown("""
    # 📚 Semantic Book Recommender
    ## Find your next favorite book using AI-powered semantic search!
    
    Enter a description of what you're looking for, and our AI will find books that match your interests using semantic understanding.
    """)

    with gr.Row():
        with gr.Column(scale=2):
            user_query = gr.Textbox(
                label="Describe your ideal book:",
                placeholder="e.g., 'A thrilling mystery set in Victorian London with complex characters'",
                lines=3,
                max_lines=5,
            )

        with gr.Column(scale=1):
            category_dropdown = gr.Dropdown(
                label="Filter by category (optional)",
                choices=categories,
                value="All",
            )
            tone_dropdown = gr.Dropdown(
                label="Filter by emotional tone (optional)",
                choices=tones,
                value="All",
            )
            submit_button = gr.Button("🔍 Find Books", variant="primary", size="lg")

    gr.Markdown("---")
    
    # Use HTML component instead of Gallery for better text display
    output = gr.HTML(
        label="Book Recommendations",
        value="<p>Enter a search query above to get personalized book recommendations!</p>"
    )

    # Event handlers
    submit_button.click(
        fn=recommend_books,
        inputs=[user_query, category_dropdown, tone_dropdown],
        outputs=output,
    )
    
    # Allow Enter key to submit
    user_query.submit(
        fn=recommend_books,
        inputs=[user_query, category_dropdown, tone_dropdown],
        outputs=output,
    )

    # Add some usage tips at the bottom
    gr.Markdown("""
    ### 💡 Tips for better results:
    - Be specific about what you're looking for (genre, setting, themes, etc.)
    - Try different emotional tones to discover books with specific moods
    - Use category filters to narrow down results
    - Experiment with different search terms if you don't find what you're looking for
    """)

print("Enhanced app initialized successfully! 🚀")

if __name__ == "__main__":
    dashboard.launch()