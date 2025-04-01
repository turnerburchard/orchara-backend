from pydantic import BaseModel
from typing import List

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