import os
from dataclasses import dataclass, field
from typing import Set

@dataclass
class SearchConfig:
    # Paths
    INDEX_PATH: str = field(default_factory=lambda: os.environ.get('INDEX_PATH', '/data/hnsw_index.bin'))
    MAPPING_PATH: str = field(default_factory=lambda: os.environ.get('MAPPING_PATH', '/data/id_mapping.json'))
    
    # Model settings
    DIM: int = 384
    
    # Search settings
    SEARCH_MULTIPLIER: int = 5
    MAX_SEARCH_ATTEMPTS: int = 3
    HNSW_EF: int = 50
    REQUIRE_ABSTRACT: bool = True
    
    # Keyword settings
    MIN_KEYWORD_LENGTH: int = 3
    STOP_WORDS: Set[str] = field(default_factory=lambda: {
        'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i', 'it', 
        'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at'
    })
    
    # Scoring weights
    SEMANTIC_WEIGHT: float = 0.4
    KEYWORD_WEIGHT: float = 0.5
    DIVERSITY_WEIGHT: float = 0.1
    RESULTS_MULTIPLIER: int = 10

# Create a default config instance
default_config = SearchConfig() 