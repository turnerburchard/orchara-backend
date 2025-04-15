import json
import numpy as np
import hnswlib
from typing import List, Dict, Any
from .config import SearchConfig, default_config
from app.models import SearchResult
from app.services.embedding import EmbeddingService
from .scoring import ScoringService
from app.services.database import DatabaseService
from app.utils.db import get_async_connection
import asyncio

class SearchService:
    def __init__(self, config: SearchConfig = default_config):
        self.config = config
        self.embedding_service = EmbeddingService()
        self.scoring_service = ScoringService(config)
        self.database_service = DatabaseService()
        self.database_service.require_abstract = config.REQUIRE_ABSTRACT
        
        with open(config.MAPPING_PATH, "r") as f:
            self.id_map = json.load(f)
        
        self.index = hnswlib.Index(space='cosine', dim=config.DIM)
        self.index.load_index(config.INDEX_PATH)
        self.index.set_ef(config.HNSW_EF)

    def validate_query(self, query: str) -> str:
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        return query.strip()

    async def search(self, query: str, cluster_size: int) -> List[SearchResult]:
        try:
            query = self.validate_query(query)
            print(f"Processing search query: '{query}' with cluster_size: {cluster_size}")

            query_embedding = await self.embedding_service.get_embedding_async(query, normalize=True)
            embedding_str = f"[{','.join(map(str, query_embedding))}]"
            
            conn = await get_async_connection()
            try:
                rows = await conn.fetch(
                    """
                    WITH similarity_scores AS (
                        SELECT 
                            id,
                            title,
                            abstract,
                            url,
                            doi,
                            authors,
                            published_date,
                            1 - (embedding <=> $1::vector) as semantic_score
                        FROM papers
                        WHERE embedding IS NOT NULL
                        AND abstract IS NOT NULL 
                        AND abstract != ''
                        AND title IS NOT NULL
                        AND title != ''
                        ORDER BY embedding <=> $1::vector
                        LIMIT $2
                    )
                    SELECT * FROM similarity_scores
                    """,
                    embedding_str,
                    cluster_size * self.config.SEARCH_MULTIPLIER
                )
                
                print(f"Found {len(rows)} papers using vector similarity")
                
                results = []
                for row in rows:
                    keyword_score = await self.scoring_service.calculate_keyword_relevance(
                        query, row['abstract']
                    )
                    
                    temp_result = SearchResult(
                        internal_id=row['id'],
                        paper_id=str(row['id']),
                        semantic_score=float(row['semantic_score']),
                        keyword_score=float(keyword_score),
                        title=row['title'],
                        abstract=row['abstract'],
                        url=row['url']
                    )
                    results.append(temp_result)
                
                if results:
                    diversity_scores = await self.scoring_service.calculate_diversity_score(results)
                    for i, result in enumerate(results):
                        result.diversity_score = float(diversity_scores[i] if i < len(diversity_scores) else 0.0)
                        result.final_score = (
                            self.config.SEMANTIC_WEIGHT * result.semantic_score +
                            self.config.KEYWORD_WEIGHT * result.keyword_score +
                            self.config.DIVERSITY_WEIGHT * result.diversity_score
                        )
                
                results.sort(key=lambda x: x.final_score, reverse=True)
                return results[:cluster_size]
                
            finally:
                await conn.close()
                
        except Exception as e:
            print(f"Error during search: {str(e)}")
            import traceback
            traceback.print_exc()
            return [] 