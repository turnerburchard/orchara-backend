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
        response = summarizer.summarize(request.text)
        return {"summary": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/summarize_papers")
async def api_summarize_papers(request: SummarizePapersRequest):
    logging.info(f"Received request with {len(request.papers)} papers")
    logging.info(f"Request type: {type(request)}")
    logging.info(f"Papers type: {type(request.papers)}")
    
    # Validate papers list
    if not request.papers:
        raise HTTPException(status_code=400, detail="Papers list cannot be empty")
    
    # Log first paper details
    if len(request.papers) > 0:
        first_paper = request.papers[0]
        logging.info(f"First paper type: {type(first_paper)}")
        logging.info(f"First paper dict: {first_paper.dict()}")
    
    # Validate each paper has required fields
    for i, paper in enumerate(request.papers):
        if not paper.paper_id:
            raise HTTPException(status_code=400, detail=f"Paper {i} is missing paper_id")
        if not paper.title:
            raise HTTPException(status_code=400, detail=f"Paper {i} is missing title")
        if not paper.abstract:
            raise HTTPException(status_code=400, detail=f"Paper {i} is missing abstract")
        if not paper.url:
            raise HTTPException(status_code=400, detail=f"Paper {i} is missing url")
    
    # Log each paper's structure
    for i, paper in enumerate(request.papers):
        logging.info(f"Paper {i}: paper_id={paper.paper_id}, title={paper.title}, abstract_length={len(paper.abstract)}, url={paper.url}")
    
    summarizer = Summarizer()
    try:
        # Pass Pydantic models directly to summarizer
        result = summarizer.summarize_with_citations(request.papers)
        
        # Validate result structure
        if not isinstance(result, dict):
            raise ValueError("Invalid result format: expected dictionary")
        if "summary" not in result:
            raise ValueError("Invalid result format: missing summary")
        if "citations" not in result:
            raise ValueError("Invalid result format: missing citations")
            
        logging.info(f"Successfully generated summary with {len(result.get('citations', []))} citations")
        return result
        
    except ValueError as e:
        logging.error(f"Validation error in summarize_papers: {str(e)}")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logging.error(f"Error in summarize_papers: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error generating summary: {str(e)}") 