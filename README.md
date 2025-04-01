# Orchara Backend

A FastAPI-based backend service for searching and analyzing academic papers. Built with Python and Docker.

## Quick Start

1. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your database and API credentials
   ```

2. **Start the service**
   ```bash
   # Development mode (with hot reload)
   docker compose -f docker-compose.dev.yml up

   # Production mode
   docker compose up
   ```

3. **Access the API**
   - The service runs on `http://localhost:5001`
   - API documentation available at `http://localhost:5001/docs`

## Key Features

- **Semantic Search**: Find papers using natural language queries
- **Paper Summarization**: Generate concise summaries of research papers
- **Citation Support**: Track and reference paper citations
- **Vector Search**: Efficient similarity search using embeddings

## Development

- Uses FastAPI for the web framework
- Docker for containerization
- Hot reload enabled in development mode
- Test data available for development

## Environment Variables

Required variables in `.env`:
```
DB_NAME=your_db_name
DB_USER=your_db_user
DB_PASSWORD=your_db_password
DB_HOST=db
DB_PORT=5432
OPENAI_API_KEY=your_openai_key
```

## Project Structure

```
orchara-backend/
├── app/                # Main application code
│   ├── api/           # API routes and models
│   ├── services/      # Core services (search, summarize)
│   └── utils/         # Utility functions
├── tests/             # Test files
├── docker-compose.yml # Production config
└── docker-compose.dev.yml # Development config
```
