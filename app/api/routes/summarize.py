from fastapi import APIRouter, HTTPException
from app.api.models import SummarizeRequest, SummarizePapersRequest
from app.services.summarize import Summarizer
import logging
from typing import List

router = APIRouter()

@router.post("/summarize")
async def api_summarize(request: SummarizeRequest):
    summarizer = Summarizer()
    try:
        response = await summarizer.summarize(request.text)
        return {"summary": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/summarize_papers")
async def summarize_papers(request: SummarizePapersRequest):
    """
    Generate a summary with citations for a list of papers.
    """
    try:
        # Validate papers list
        if not request.papers:
            raise HTTPException(status_code=400, detail="No papers provided")
            
        # Validate each paper has required fields
        for paper in request.papers:
            if not all([paper.paper_id, paper.title, paper.abstract, paper.url]):
                raise HTTPException(
                    status_code=400,
                    detail=f"Paper {paper.paper_id} is missing required fields"
                )
        
        # Get summary with citations
        summarizer = Summarizer()
        result = await summarizer.summarize_with_citations(
            papers=request.papers,
            query=request.query
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error in summarize_papers: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 