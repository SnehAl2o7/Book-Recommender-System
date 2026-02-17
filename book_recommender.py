#!/usr/bin/env python3
"""
Book Recommender System using Google Gemini API
A semantic book recommendation system powered by Google's Generative AI
"""

import os
import pandas as pd
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Load environment variables
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("❌ GOOGLE_API_KEY not found in .env file")

print("✅ Gemini API Key Loaded")

# Load books data
print("📚 Loading books dataset...")
books = pd.read_csv("books_cleaned.csv")
print(f"✅ Loaded {len(books)} books from dataset")

# Check if tagged_description.txt exists, if not create it
if not os.path.exists("tagged_description.txt"):
    print("📝 Creating tagged_description.txt...")
    books["tagged_description"].to_csv("tagged_description.txt", sep="\n", index=False, header=False)
    print("✅ Created tagged_description.txt")

# Initialize embeddings with the correct model name
print("🔧 Initializing Gemini embeddings...")
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",  # Updated to correct model name
    google_api_key=api_key,
)
print("✅ Embeddings initialized")

# Load and split text documents
print("📄 Loading and splitting documents...")
raw_documents = TextLoader("tagged_description.txt").load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0, separator="\n")
documents = text_splitter.split_documents(raw_documents)
print(f"✅ Split into {len(documents)} document chunks")

# Create or load vector database
print("🔍 Creating vector database (this may take a while)...")
db_books = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    persist_directory="./chroma_db"  # Persist to avoid recreating every time
)
print("✅ Vector database created successfully!")


def retrieve_semantic_recommendations(
        query: str,
        top_k: int = 10,
) -> pd.DataFrame:
    """
    Retrieve book recommendations based on semantic similarity
    
    Args:
        query: Natural language query describing desired book
        top_k: Number of recommendations to return
    
    Returns:
        DataFrame containing recommended books
    """
    print(f"🔎 Searching for: '{query}'...")
    
    # Similarity search using Gemini embeddings
    recs = db_books.similarity_search(query, k=50)
    
    books_list = []
    for i in range(len(recs)):
        # Extracting the ISBN from the page content
        try:
            isbn = int(recs[i].page_content.strip('"').split()[0])
            books_list.append(isbn)
        except (ValueError, IndexError):
            continue
    
    recommendations = books[books["isbn13"].isin(books_list)].head(top_k)
    print(f"✅ Found {len(recommendations)} recommendations\n")
    
    return recommendations


if __name__ == "__main__":
    # Example queries
    queries = [
        "A book to teach children about nature",
        "Mystery thriller with suspense",
        "Science fiction about space exploration",
        "Romance novel with happy ending",
        "Self-help book for personal growth"
    ]
    
    print("\n" + "="*80)
    print("📖 BOOK RECOMMENDER SYSTEM - DEMO")
    print("="*80 + "\n")
    
    # Run first query as demo
    query = queries[0]
    results = retrieve_semantic_recommendations(query, top_k=5)
    
    print(f"Query: '{query}'\n")
    print("Top 5 Recommendations:")
    print("-" * 80)
    
    for idx, (_, row) in enumerate(results.iterrows(), 1):
        print(f"\n{idx}. {row['title']}")
        print(f"   Author(s): {row['authors']}")
        print(f"   Category: {row['categories']}")
        print(f"   Rating: {row['average_rating']:.2f} ({int(row['ratings_count'])} ratings)")
        if pd.notna(row['description']) and len(str(row['description'])) > 0:
            desc = str(row['description'])[:200] + "..." if len(str(row['description'])) > 200 else str(row['description'])
            print(f"   Description: {desc}")
    
    print("\n" + "="*80)
    print("\n💡 You can modify the queries list in the code to try different searches!")
    print("   Or use the retrieve_semantic_recommendations() function directly.\n")
