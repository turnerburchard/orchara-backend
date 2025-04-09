import json
import numpy as np
import hnswlib
from typing import List, Dict, Any
from .config import SearchConfig, default_config
from app.models import SearchResult
from app.services.embedding import EmbeddingService
from .scoring import ScoringService
from app.services.database import DatabaseService
import asyncio

class SearchService:
    def __init__(self, config: SearchConfig = default_config):
        self.config = config
        self.embedding_service = EmbeddingService()
        self.scoring_service = ScoringService(config)
        self.database_service = DatabaseService()
        self.database_service.require_abstract = config.REQUIRE_ABSTRACT
        
        # Load the ID mapping from disk - keep as raw dict without string conversion
        with open(config.MAPPING_PATH, "r") as f:
            self.id_map = json.load(f)
        
        # Initialize and load the HNSWlib index
        self.index = hnswlib.Index(space='cosine', dim=config.DIM)
        self.index.load_index(config.INDEX_PATH)
        self.index.set_ef(config.HNSW_EF)

    def validate_query(self, query: str) -> str:
        """Validate and clean the search query."""
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        return query.strip()

    async def search(self, query: str, cluster_size: int) -> List[SearchResult]:
        """Returns up to `cluster_size` results, using semantic similarity,
        keyword matching, and diversity scoring."""
        try:
            query = self.validate_query(query)
            
            print(f"Processing search query: '{query}' with cluster_size: {cluster_size}")

            query_embedding = await self.embedding_service.get_embedding_async(query, normalize=True)
            query_embedding = np.array([query_embedding], dtype=np.float32)

            cumulative_results = []
            processed_internal_ids = set()
            attempt = 0

            while len(cumulative_results) < cluster_size and attempt < self.config.MAX_SEARCH_ATTEMPTS:
                needed = cluster_size * self.config.RESULTS_MULTIPLIER
                k_multiplier = self.config.SEARCH_MULTIPLIER * (attempt + 1)
                k = min(max(needed, cluster_size * k_multiplier), self.index.get_max_elements())

                print(f"Attempt {attempt + 1}: Querying index with k={k}")

                loop = asyncio.get_event_loop()
                labels, distances = await loop.run_in_executor(
                    None,
                    lambda: self.index.knn_query(query_embedding, k=k, filter=lambda label: label not in processed_internal_ids)
                )

                internal_ids_this_attempt = labels[0]
                distances_this_attempt = distances[0]

                if not internal_ids_this_attempt.size:
                    print(f"Attempt {attempt + 1}: No new results found from HNSW.")
                    break

                new_internal_ids = set(internal_ids_this_attempt) - processed_internal_ids
                processed_internal_ids.update(new_internal_ids)

                paper_id_map_results_int = {}
                semantic_scores_map = {}
                for i, iid in enumerate(internal_ids_this_attempt):
                    iid_int = int(iid)
                    try:
                        paper_id_int = self.id_map.get(str(iid_int)) # id_map uses string keys
                        if paper_id_int is not None:
                             paper_id_map_results_int[iid_int] = int(paper_id_int) # Keep INT for DB query
                             semantic_scores_map[iid_int] = 1.0 - float(distances_this_attempt[i])
                    except Exception as e:
                        print(f"Warning: Error processing internal ID {iid_int}: {e}")

                paper_ids_to_fetch_int = list(paper_id_map_results_int.values())

                print(f"Attempt {attempt + 1}: Mapped {len(internal_ids_this_attempt)} internal IDs to {len(paper_ids_to_fetch_int)} unique INT paper IDs.")

                if not paper_ids_to_fetch_int:
                    print(f"Attempt {attempt + 1}: No valid paper IDs found.")
                    attempt += 1
                    continue

                papers = await self.database_service.get_papers(paper_ids_to_fetch_int)
                papers_dict = {paper['paper_id']: paper for paper in papers} # DB returns paper_id as string
                print(f"Attempt {attempt + 1}: Retrieved {len(papers)} papers from DB.")

                candidate_results = []
                for internal_id, paper_id_int_val in paper_id_map_results_int.items():
                    paper_id_str_val = str(paper_id_int_val)
                    if paper_id_str_val in papers_dict:
                        paper = papers_dict[paper_id_str_val]
                        abstract = str(paper.get('abstract', ''))
                        candidate_results.append({
                            "internal_id": internal_id,
                            "paper_id": paper_id_str_val,
                            "semantic_score": semantic_scores_map.get(internal_id, 0.0),
                            "title": str(paper.get('title', '')),
                            "abstract": abstract,
                            "url": str(paper.get('url', ''))
                        })
                print(f"Attempt {attempt + 1}: Created {len(candidate_results)} candidate result dicts.")

                if not candidate_results:
                    attempt += 1
                    continue

                keyword_scores = await asyncio.gather(
                    *[self.scoring_service.calculate_keyword_relevance(query, res['abstract']) for res in candidate_results]
                )
                for i, res in enumerate(candidate_results):
                    res['keyword_score'] = float(keyword_scores[i])

                # Create temporary SearchResult objects for diversity calculation
                temp_results_for_diversity = []
                for res_dict in candidate_results:
                    try:
                        temp_results_for_diversity.append(SearchResult(**res_dict))
                    except Exception as temp_val_error:
                        print(f"Warning: Failed temp SearchResult creation for diversity: {temp_val_error} - Dict: {res_dict}")

                diversity_scores = []
                if temp_results_for_diversity:
                    try:
                        diversity_scores = await self.scoring_service.calculate_diversity_score(temp_results_for_diversity)
                    except Exception as diversity_error:
                        print(f"Error calling calculate_diversity_score: {diversity_error}")
                        diversity_scores = [0.0] * len(candidate_results)
                else:
                     diversity_scores = [0.0] * len(candidate_results)

                if len(diversity_scores) != len(candidate_results):
                    print(f"Warning: Mismatch diversity scores ({len(diversity_scores)}) vs candidates ({len(candidate_results)}). Defaulting.")
                    diversity_scores = [0.0] * len(candidate_results)

                results_this_attempt = []
                for i, res in enumerate(candidate_results):
                    try:
                        sem_score = float(res.get('semantic_score', 0.0))
                        key_score = float(res.get('keyword_score', 0.0))
                        div_score = float(diversity_scores[i] if i < len(diversity_scores) else 0.0)

                        final_score = (
                            self.config.SEMANTIC_WEIGHT * sem_score +
                            self.config.KEYWORD_WEIGHT * key_score +
                            self.config.DIVERSITY_WEIGHT * div_score
                        )

                        res['diversity_score'] = div_score
                        res['final_score'] = float(final_score)

                        search_result_obj = SearchResult(**res)
                        results_this_attempt.append(search_result_obj)
                    except Exception as pydantic_error:
                        print(f"Pydantic validation failed for dict: {res} Error: {pydantic_error}")

                cumulative_results.extend(results_this_attempt)
                print(f"Attempt {attempt + 1}: Added {len(results_this_attempt)} valid results. Total results: {len(cumulative_results)}.")
                attempt += 1

            if not cumulative_results:
                print("Warning: No results found after all attempts!")
                return []

            cumulative_results.sort(key=lambda x: x.final_score, reverse=True)
            print(f"Returning {min(len(cumulative_results), cluster_size)} results after sorting.")
            return cumulative_results[:cluster_size]

        except ValueError as ve:
            raise ve
        except Exception as e:
            print(f"Error during search execution: {str(e)}")
            import traceback
            traceback.print_exc()
            return [] 