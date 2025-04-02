# Orchara Backend

A FastAPI-based backend service for searching and analyzing academic papers.

## Quick Start

1. **Set up environment variables**
   ```bash
   cp .env.example .env
   ```

2. **Start the service**
   ```bash
   # Development
   make up
   make down

   # Production
   docker compose up -d
   docker compose down
   ```

## Features

- Semantic Search
- Paper Summarization
- Citation Support
- Vector Search
