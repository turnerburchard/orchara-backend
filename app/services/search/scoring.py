import re
from collections import defaultdict
from typing import List, Set
from .config import SearchConfig
from app.models import SearchResult
import asyncio

class ScoringService:
    def __init__(self, config: SearchConfig):
        self.config = config

    def extract_keywords(self, text: str) -> List[str]:
        """Extract meaningful keywords from text, excluding common words"""
        if not text:
            return []
        words = re.findall(r'\w+', text.lower())
        return [word for word in words 
                if word not in self.config.STOP_WORDS 
                and len(word) > self.config.MIN_KEYWORD_LENGTH]

    async def calculate_keyword_relevance(self, query: str, text: str) -> float:
        """Calculate how relevant the text is to the search query based on keyword overlap"""
        if not text:
            return 0.0
            
        # Run CPU-bound operations in thread pool
        loop = asyncio.get_event_loop()
        query_keywords = await loop.run_in_executor(None, lambda: set(self.extract_keywords(query)))
        text_keywords = await loop.run_in_executor(None, lambda: set(self.extract_keywords(text)))
        
        if not query_keywords or not text_keywords:
            return 0.0
        
        overlap = len(query_keywords.intersection(text_keywords))
        total = len(query_keywords.union(text_keywords))
        
        return overlap / total if total > 0 else 0.0

    async def calculate_diversity_score(self, results: List[SearchResult]) -> List[float]:
        """Calculate diversity score based on keyword overlap"""
        if not results:
            return []
        
        # Run CPU-bound operations in thread pool
        loop = asyncio.get_event_loop()
        
        # Extract keywords from all abstracts
        all_keywords = []
        for result in results:
            keywords = await loop.run_in_executor(None, self.extract_keywords, result.abstract)
            all_keywords.extend(keywords)
        
        keyword_counts = defaultdict(int)
        for keyword in all_keywords:
            keyword_counts[keyword] += 1
        
        diversity_scores = []
        for result in results:
            keywords = await loop.run_in_executor(None, self.extract_keywords, result.abstract)
            avg_frequency = sum(keyword_counts[k] for k in keywords) / len(keywords) if keywords else 0
            diversity_score = 1 / (1 + avg_frequency)
            diversity_scores.append(diversity_score)
        
        return diversity_scores

    async def calculate_final_scores(self, results: List[SearchResult]) -> List[SearchResult]:
        """Calculate final scores combining semantic, keyword, and diversity scores"""
        diversity_scores = await self.calculate_diversity_score(results)
        
        for i, result in enumerate(results):
            result.diversity_score = diversity_scores[i]
            result.final_score = (
                self.config.SEMANTIC_WEIGHT * result.semantic_score +
                self.config.KEYWORD_WEIGHT * result.keyword_score +
                self.config.DIVERSITY_WEIGHT * diversity_scores[i]
            )
        
        return results 