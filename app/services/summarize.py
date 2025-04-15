from openai import AsyncOpenAI
import json
import os
from dotenv import load_dotenv
from app.models import Paper, Citation, SummaryResult
from app.services.prompts import get_summary_prompt, get_summary_with_citations_prompt, get_summary_system_prompt
from typing import List

class Summarizer:
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Summarizer, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            load_dotenv()
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable is not set")
            self.client = AsyncOpenAI(api_key=api_key)
            self._initialized = True

    async def summarize(self, text):
        response = await self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": get_summary_prompt(text)}
            ]
        )
        return response.choices[0].message.content
        
    async def summarize_with_citations(self, papers: list[Paper], query: str = None) -> SummaryResult:
        formatted_papers = []
        for i, paper in enumerate(papers, 1):
            formatted_papers.append(f"Paper {i}: \"{paper.title}\"\nAbstract: {paper.abstract}\n")
        
        papers_text = "\n\n".join(formatted_papers)
        
        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": get_summary_system_prompt()},
                    {"role": "user", "content": get_summary_with_citations_prompt(papers_text, query)}
                ]
            )
            
            result = json.loads(response.choices[0].message.content)
            
            processed_citations: List[Citation] = []
            for citation in result.get("citations", []):
                id_num = citation["id"]
                if not isinstance(id_num, int):
                    continue
                    
                idx = id_num - 1
                if 0 <= idx < len(papers):
                    processed_citations.append(Citation(
                        id=id_num,
                        paper_id=papers[idx].paper_id,
                        title=papers[idx].title,
                        url=papers[idx].url,
                        context=citation.get("context", "")
                    ))
            
            summary_text = result.get("summary", "")
            summary_text = summary_text.replace(" .", ".")
            
            return SummaryResult(
                summary=summary_text,
                citations=processed_citations
            )
            
        except Exception as e:
            return SummaryResult(
                summary=f"Error generating summary: {str(e)}",
                citations=[]
            )

