# Interactive Book Recommender System

An AI-powered book recommendation system using Google Gemini AI for semantic search.

## Features

- 🤖 **AI-Powered Search**: Uses Google Gemini embeddings for intelligent semantic matching
- 🔍 **Advanced Filters**: Filter by rating, category, and number of results
- 🎨 **Modern UI**: Beautiful gradient design with smooth animations
- ⚡ **Fast API Backend**: RESTful API built with FastAPI
- 📊 **Database Statistics**: View collection stats at a glance

## Setup

### Prerequisites

- Python 3.10+
- Conda environment manager
- Google Gemini API key

### Installation

1. **Activate the conda environment:**
   ```bash
   conda activate genai
   ```

2. **Install required packages:**
   ```bash
   pip install fastapi uvicorn pandas python-dotenv langchain langchain-google-genai langchain-community langchain-chroma chromadb
   ```

3. **Set up your API key:**
   
   Make sure your `.env` file contains:
   ```
   GOOGLE_API_KEY=your_api_key_here
   ```

## Usage

### Starting the Backend

Run the FastAPI server:

```bash
~/.conda/envs/genai/bin/python backend.py
```

The API will be available at `http://localhost:8000`

**Note**: The first time you run the backend, it will create a vector database from the book dataset. This may take a few minutes.

### Opening the Frontend

Simply open `index.html` in your web browser:

```bash
xdg-open index.html  # Linux
# or just double-click the file
```

## API Endpoints

### `GET /`
Health check endpoint

### `GET /categories`
Get all available book categories

### `POST /recommend`
Get book recommendations

**Request body:**
```json
{
  "query": "mystery thriller with suspense",
  "top_k": 10,
  "min_rating": 4.0,
  "category": "Fiction"
}
```

**Response:**
```json
[
  {
    "isbn13": "9780...",
    "title": "Book Title",
    "authors": "Author Name",
    "categories": "Fiction",
    "average_rating": 4.5,
    "ratings_count": 1000,
    "description": "Book description...",
    "thumbnail": "http://...",
    "published_year": 2020
  }
]
```

### `GET /stats`
Get database statistics

## How It Works

1. **Data Processing**: The system loads the `books_cleaned.csv` dataset and creates tagged descriptions
2. **Vector Database**: Book descriptions are embedded using Google Gemini's text-embedding-004 model and stored in ChromaDB
3. **Semantic Search**: User queries are embedded and matched against the book database using cosine similarity
4. **Filtering**: Results are filtered based on user preferences (rating, category)
5. **API Response**: Top matching books are returned to the frontend

## Files

- `backend.py` - FastAPI backend server
- `index.html` - Interactive web frontend
- `book_recommender.py` - Standalone CLI version
- `books_cleaned.csv` - Processed book dataset
- `.env` - API key configuration

## Tips

- **First Run**: The vector database creation happens once on the first run and is saved to disk
- **Query Tips**: Use natural language like "books about space exploration for kids" or "romantic comedy novels"
- **Filters**: Combine filters for more specific results (e.g., Fiction with rating > 4.5)

## Troubleshooting

**Backend won't start:**
- Ensure your GOOGLE_API_KEY is set in `.env`
- Check that all dependencies are installed
- Verify you're using the genai conda environment

**No recommendations:**
- Make sure the backend is running
- Check browser console for errors
- Verify the API URL in `index.html` matches your backend

**CORS errors:**
- The backend has CORS enabled for all origins in development
- For production, update the `allow_origins` in `backend.py`

## License

Apache License 2.0
