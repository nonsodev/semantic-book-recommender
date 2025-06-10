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
        final_top_k: int = 16,
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

def recommend_books(query: str, category: str, tone: str):
    """Main recommendation function for Gradio interface."""
    
    if not query.strip():
        return []
    
    try:
        recommendations = retrieve_semantic_recommendations(query, category, tone)
        results = []

        for _, row in recommendations.iterrows():
            # Handle missing description
            description = row.get("description", "No description available")
            if pd.isna(description):
                description = "No description available"
            
            # Truncate description
            truncated_desc_split = str(description).split()
            truncated_description = " ".join(truncated_desc_split[:30]) + "..."

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

            # Create caption
            title = row.get("title_and_subtitle", "Unknown Title")
            caption = f"{title} by {authors_str}: {truncated_description}"
            results.append((row["large_thumbnail"], caption))
            
        return results
    
    except Exception as e:
        print(f"Error in recommend_books: {e}")
        return []

# Prepare dropdown options
categories = ["All"] + sorted(books["categories"].unique().tolist())
tones = ["All", "Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

# Create Gradio interface
with gr.Blocks(theme=gr.themes.Glass()) as dashboard:
    gr.Markdown("# 📚 Semantic Book Recommender")
    gr.Markdown("## Find your next favorite book using AI-powered semantic search!")

    with gr.Row():
        user_query = gr.Textbox(
            label="Describe your ideal book:",
            placeholder="e.g., 'A thrilling mystery set in Victorian London'",
            lines=2,
            max_lines=3,
        )

        with gr.Column():
            category_dropdown = gr.Dropdown(
                label="Select a category (optional)",
                choices=categories,
                value="All",
            )
            tone_dropdown = gr.Dropdown(
                label="Select an emotional tone (optional)",
                choices=tones,
                value="All",
            )
            submit_button = gr.Button("🔍 Find Books", variant="primary")

    gr.Markdown("## 📖 Recommendations")
    output = gr.Gallery(
        label="Recommended Books",
        columns=4,  # Reduced for better mobile experience
        rows=4,
        height="auto",
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

print("App initialized successfully! 🚀")

if __name__ == "__main__":
    dashboard.launch()