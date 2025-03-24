# Orchara Backend

A Flask-based backend service that provides API endpoints for searching, analyzing, and processing academic papers. The service connects to the Orchara database and provides various ML/AI capabilities.

## Overview

The backend service provides:
- Semantic search using vector embeddings
- Text summarization
- Paper analysis and insights
- RESTful API endpoints for all functionality

## Project Structure

```
orchara-backend/
├── api.py              # Main Flask application and API routes
├── docker-compose.yml  # Docker services configuration
├── Dockerfile         # Backend service container definition
├── search.py         # Search implementation and query processing
├── summarize.py      # Text summarization using ML models
└── util.py           # Shared utility functions
```

Key components:
- `api.py`: Main Flask application that exposes REST endpoints
- `search.py`: Implements search functionality and query processing
- `summarize.py`: Provides text summarization capabilities
- `util.py`: Shared utility functions used across the service

## API Endpoints

The service exposes the following endpoints:

### Search Papers
- `POST /api/search`
  - Request body:
    ```json
    {
      "query": "search query",
      "cluster_size": 10
    }
    ```
  - Returns: List of papers matching the query

### Summarize Text
- `POST /api/summarize`
  - Request body:
    ```json
    {
      "text": "text to summarize"
    }
    ```
  - Returns: Generated summary


## Running with Docker

1. **Start the service**:
```bash
docker-compose up -d
```

2. **View logs**:
```bash
docker-compose logs -f backend
```

3. **Stop the service**:
```bash
docker-compose down
```

### Environment Variables

Required environment variables:
- `DB_HOST`: Database host (default: db)
- `DB_PORT`: Database port (default: 5432)
- `DB_NAME`: Database name
- `DB_USER`: Database user
- `DB_PASSWORD`: Database password


## Usage

1. Start the service using Docker Compose
2. The API will be available at `http://localhost:5001`
3. Use the API endpoints to:
   - Search for papers using semantic similarity
   - Generate paper summaries