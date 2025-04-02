import json
import numpy as np
import hnswlib
from typing import List
from .config import SearchConfig, default_config
from app.api.models import SearchResult
from .embedding import EmbeddingService
from .scoring import ScoringService
from .database import DatabaseService
import asyncio

class SearchService:
    def __init__(self, config: SearchConfig = default_config):
        self.config = config
        self.embedding_service = EmbeddingService(config)
        self.scoring_service = ScoringService(config)
        self.database_service = DatabaseService(config)
        
        # Load the ID mapping from disk
        with open(config.MAPPING_PATH, "r") as f:
            self.id_map = json.load(f)
        
        # Initialize and load the HNSWlib index
        self.index = hnswlib.Index(space='cosine', dim=config.DIM)
        self.index.load_index(config.INDEX_PATH)
        self.index.set_ef(config.HNSW_EF)

    async def search(self, query: str, cluster_size: int) -> List[SearchResult]:
        """Returns up to `cluster_size` results, using semantic similarity,
        keyword matching, and diversity scoring."""
        try:
            print(f"Processing search query: '{query}' with cluster_size: {cluster_size}")
            
            # Get query embedding
            query_embedding = await self.embedding_service.get_query_embedding(query)
            query_embedding = np.array([query_embedding], dtype=np.float32)
            
            results = []
            attempt = 0
            
            while len(results) < cluster_size * self.config.RESULTS_MULTIPLIER and attempt < self.config.MAX_SEARCH_ATTEMPTS:
                extended_k = cluster_size * self.config.SEARCH_MULTIPLIER * (attempt + 1)
                max_elements = self.index.get_max_elements()
                k = min(extended_k, max_elements)
                
                print(f"Attempt {attempt + 1}: Querying index with k={k}")
                
                # Run HNSW search in thread pool since it's CPU-bound
                loop = asyncio.get_event_loop()
                labels, distances = await loop.run_in_executor(
                    None, 
                    lambda: self.index.knn_query(query_embedding, k=k)
                )
                
                # Convert internal IDs to paper IDs
                internal_ids = labels[0]
                paper_ids = [self.id_map.get(str(internal_id)) for internal_id in internal_ids]
                
                print(f"Found {len(paper_ids)} potential matches")
                
                # Fetch papers with optional abstract filtering
                details_dict = await self.database_service.get_paper_details(paper_ids)
                
                print(f"Retrieved {len(details_dict)} papers")
                
                # Reset results for new attempt
                results = []
                
                # Build results with semantic similarity scores
                for i, internal_id in enumerate(internal_ids):
                    paper_id = paper_ids[i]
                    distance = distances[0][i]
                    if paper_id in details_dict:
                        paper_details = details_dict[paper_id]
                        # Calculate keyword relevance score
                        keyword_score = await self.scoring_service.calculate_keyword_relevance(query, paper_details.abstract)
                        
                        results.append(SearchResult(
                            internal_id=int(internal_id),
                            paper_id=paper_id,
                            semantic_score=1 - float(distance),  # Convert distance to similarity score
                            keyword_score=keyword_score,
                            title=paper_details.title,
                            abstract=paper_details.abstract,
                            url=paper_details.url,
                        ))
                
                print(f"Processed {len(results)} valid results")
                attempt += 1
            
            if not results:
                print("Warning: No results found!")
                return []
            
            # Calculate final scores and sort
            results = await self.scoring_service.calculate_final_scores(results)
            results.sort(key=lambda x: x.final_score, reverse=True)
            return results[:cluster_size]
        
        except Exception as e:
            print(f"Search error: {str(e)}")
            return [] 