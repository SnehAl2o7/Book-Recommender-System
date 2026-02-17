#!/usr/bin/env python3
"""
FastAPI Backend for Book Recommender System
Uses Google Gemini API for semantic search
"""

import os
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Load environment variables
load_dotenv()

app = FastAPI(title="Book Recommender API", version="1.0.0")

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
books_df = None
db_books = None
embeddings = None

class RecommendationRequest(BaseModel):
    query: str
    top_k: int = 10
    min_rating: Optional[float] = None
    category: Optional[str] = None

class BookResponse(BaseModel):
    isbn13: str
    title: str
    authors: str
    categories: str
    average_rating: float
    ratings_count: int
    description: Optional[str]
    thumbnail: Optional[str]
    published_year: Optional[float]

@app.on_event("startup")
async def startup_event():
    """Initialize the recommendation system on startup"""
    global books_df, db_books, embeddings
    
    print("🚀 Starting Book Recommender System...")
    
    # Load API key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("❌ GOOGLE_API_KEY not found in .env file")
    
    print("✅ API Key loaded")
    
    # Load books data
    print("📚 Loading books dataset...")
    books_df = pd.read_csv("books_cleaned.csv")
    print(f"✅ Loaded {len(books_df)} books")
    
    # Create tagged_description.txt if it doesn't exist
    if not os.path.exists("tagged_description.txt"):
        print("📝 Creating tagged_description.txt...")
        books_df["tagged_description"].to_csv("tagged_description.txt", sep="\n", index=False, header=False)
    
    # Initialize embeddings
    print("🔧 Initializing Gemini embeddings...")
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=api_key,
    )
    print("✅ Embeddings initialized")
    
    # Check if vector database already exists
    if os.path.exists("./chroma_db"):
        print("📂 Loading existing vector database...")
        db_books = Chroma(
            persist_directory="./chroma_db",
            embedding_function=embeddings
        )
        print("✅ Vector database loaded!")
    else:
        print("📄 Creating vector database (this will take a few minutes)...")
        raw_documents = TextLoader("tagged_description.txt").load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0, separator="\n")
        documents = text_splitter.split_documents(raw_documents)
        print(f"✅ Split into {len(documents)} document chunks")
        
        print("🔍 Creating vector database...")
        db_books = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory="./chroma_db"
        )
        print("✅ Vector database created and persisted!")
    
    print("🎉 Book Recommender System is ready!")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "message": "Book Recommender API is running",
        "total_books": len(books_df) if books_df is not None else 0
    }

@app.get("/categories")
async def get_categories():
    """Get all unique book categories"""
    if books_df is None:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    categories = sorted(books_df["categories"].dropna().unique().tolist())
    return {"categories": categories}

@app.post("/recommend", response_model=List[BookResponse])
async def get_recommendations(request: RecommendationRequest):
    """Get book recommendations based on semantic search"""
    if db_books is None or books_df is None:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        print(f"🔎 Searching for: '{request.query}'")
        
        # Perform semantic search
        recs = db_books.similarity_search(request.query, k=100)
        
        # Extract ISBNs from results
        books_list = []
        for rec in recs:
            try:
                isbn = int(rec.page_content.strip('"').split()[0])
                books_list.append(isbn)
            except (ValueError, IndexError):
                continue
        
        # Filter books based on ISBNs
        filtered_books = books_df[books_df["isbn13"].isin(books_list)].copy()
        
        # Apply additional filters
        if request.min_rating is not None:
            filtered_books = filtered_books[filtered_books["average_rating"] >= request.min_rating]
        
        if request.category is not None:
            filtered_books = filtered_books[
                filtered_books["categories"].str.contains(request.category, case=False, na=False)
            ]
        
        # Get top K results
        results = filtered_books.head(request.top_k)
        
        # Convert to response format
        recommendations = []
        for _, row in results.iterrows():
            recommendations.append(BookResponse(
                isbn13=str(row["isbn13"]),
                title=row["title"],
                authors=row["authors"],
                categories=row["categories"],
                average_rating=float(row["average_rating"]),
                ratings_count=int(row["ratings_count"]),
                description=row["description"] if pd.notna(row["description"]) else None,
                thumbnail=row["thumbnail"] if pd.notna(row["thumbnail"]) else None,
                published_year=float(row["published_year"]) if pd.notna(row["published_year"]) else None
            ))
        
        print(f"✅ Found {len(recommendations)} recommendations")
        return recommendations
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_stats():
    """Get database statistics"""
    if books_df is None:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    return {
        "total_books": len(books_df),
        "avg_rating": float(books_df["average_rating"].mean()),
        "categories_count": books_df["categories"].nunique(),
        "authors_count": books_df["authors"].nunique(),
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
