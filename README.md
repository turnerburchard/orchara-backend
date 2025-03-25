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
├── docker-compose.yml  # Production Docker services configuration
├── docker-compose.dev.yml  # Development Docker services configuration
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

### Development Mode
For development with hot-reloading and debug features:

1. **Start the service**:
```bash
docker compose -f docker-compose.dev.yml up
```

2. **View logs**:
```bash
docker compose -f docker-compose.dev.yml logs -f app
```

3. **Stop the service**:
```bash
docker compose -f docker-compose.dev.yml down
```

### Production Mode
For production deployment:

1. **Start the service**:
```bash
docker compose up
```

2. **View logs**:
```bash
docker compose logs -f app
```

3. **Stop the service**:
```bash
docker compose down
```

### Environment Variables

Required environment variables:
- `DB_HOST`: Database host (default: db)
- `DB_PORT`: Database port (default: 5432)
- `DB_NAME`: Database name
- `DB_USER`: Database user
- `DB_PASSWORD`: Database password

## Usage

1. Start the service using Docker Compose:
   - For development: `docker compose -f docker-compose.dev.yml up`
   - For production: `docker compose up`
2. The API will be available at `http://localhost:5001`
3. Use the API endpoints to:
   - Search for papers using semantic similarity
   - Generate paper summaries

## Future Improvements

The core system for KNN search, database indexing, and API/backend infrastructure is already in place. The following features are planned to significantly improve recommendation quality and system intelligence:

- **Claim Extraction from Abstracts**
  - Use a lightweight transformer model to extract key scientific claims or contributions from paper abstracts.
  - Store extracted claims in the database to enhance relevance scoring and downstream generation quality.

- **Graph-Based Influence Modeling**
  - Train a Graph Neural Network (GNN) on the citation graph to produce node embeddings and influence scores.
  - Integrate these influence metrics into ranking logic to prioritize high-impact or foundational papers, even if not textually similar.

- **Supervised Re-Ranking**
  - Apply a cross-encoder model to re-rank KNN search results based on semantic query-document relevance.
  - Improve final ranking precision beyond approximate vector similarity.

- **Diversity via Maximal Marginal Relevance (MMR)**
  - Introduce MMR to reduce redundancy in the final candidate set.
  - Ensure broader topical coverage in recommendations and generated summaries.

- **Asynchronous Retrieval-Augmented Generation (RAG)**
  - Generate in-depth literature reviews using RAG pipelines with LLMs.
  - Run generation as a background task, enabling users to see initial results immediately while detailed summaries load.

- **Precomputation and Distributed Workload**
  - Offload training and inference for heavier models (e.g., claim extractors, GNNs, cross-encoders) to a dedicated compute machine.
  - Keep all real-time components lightweight by using precomputed embeddings and rankings wherever possible.

These changes aim to combine modern ML techniques with efficient engineering practices to deliver higher-quality results without compromising responsiveness.
