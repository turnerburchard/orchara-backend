from pydantic import BaseModel
from typing import List, Optional

class SearchRequest(BaseModel):
    query: str
    cluster_size: int

class SummarizeRequest(BaseModel):
    text: str

class Paper(BaseModel):
    paper_id: str
    title: str
    abstract: str
    url: str

class SummarizePapersRequest(BaseModel):
    papers: list[Paper]
    query: str | None = None

class SearchResult(BaseModel):
    internal_id: int
    paper_id: str
    semantic_score: float
    keyword_score: float
    diversity_score: float = 0.0  # Default value
    final_score: float = 0.0  # Default value
    title: str
    abstract: str
    url: str

class SearchResultDict(BaseModel):
    results: List[SearchResult]

class Citation(BaseModel):
    id: int
    paper_id: str
    title: str
    url: str
    context: str

class SummaryResult(BaseModel):
    summary: str
    citations: List[Citation]

class PDFUploadResult(BaseModel):
    success: bool
    paper: Paper
    missing_doi: bool = True 
    error: Optional[str] = None  