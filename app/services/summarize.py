from openai import OpenAI
import json
import os
from dotenv import load_dotenv
from app.api.models import Paper
from pathlib import Path

class Summarizer:
    def __init__(self):
        load_dotenv()  # Load environment variables
        self.testing_mode = os.getenv('TESTING_MODE', 'false').lower() == 'true'
        
        if not self.testing_mode:
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable is not set")
            self.client = OpenAI(api_key=api_key)
            
        # Load test data
        test_data_path = Path(__file__).parent.parent / 'test_data' / 'summaries.json'
        if test_data_path.exists():
            with open(test_data_path) as f:
                self.test_data = json.load(f)
        else:
            self.test_data = {}

    def summarize(self, text):
        if self.testing_mode:
            return "This is a test summary of the provided text. It demonstrates the functionality of the summarization service without making actual API calls."
        else:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "user", "content": f"You are a researcher, accurately summarize the following set of research papers into a single paragraph: {text}"}
                ]
            )
            return response.choices[0].message.content
        
    def summarize_with_citations(self, papers: list[Paper]):
        """
        Generate a summary with citations for a list of papers.
        
        Args:
            papers: list of Pydantic Paper models, each with:
                - paper_id: unique identifier
                - title: paper title
                - abstract: paper abstract
                - url: link to the paper
                
        Returns:
            dict with:
                - summary: text with citations in {{cite:X}} format
                - citations: list of citation objects with paper details
        """
        if self.testing_mode:
            # Use test data based on number of papers
            if len(papers) == 1:
                test_result = self.test_data.get('single_paper', self._generate_test_result(papers))
            else:
                test_result = self.test_data.get('multiple_papers', self._generate_test_result(papers))
            
            # Update paper IDs and URLs to match actual papers
            for i, paper in enumerate(papers, 1):
                for citation in test_result['citations']:
                    if citation['id'] == i:
                        citation['paper_id'] = paper.paper_id
                        citation['title'] = paper.title
                        citation['url'] = paper.url
            
            return test_result
            
        # Format papers for the prompt
        formatted_papers = []
        for i, paper in enumerate(papers, 1):
            formatted_papers.append(f"Paper {i}: \"{paper.title}\"\nAbstract: {paper.abstract}\n")
        
        papers_text = "\n\n".join(formatted_papers)
        
        # Create the prompt
        prompt = f"""Analyze these research papers and provide:
        1. A 2-paragraph summary that synthesizes the main findings
        2. For each key point, cite exactly ONE paper number using {{{{cite:X}}}} format
           - Do not combine multiple papers in a single citation
           - Each citation should reference a single paper
        
        YOUR RESPONSE MUST BE IN THIS JSON FORMAT:
        {{
            "summary": "The synthesized text with citations marked as {{{{cite:X}}}}",
            "citations": [
                {{
                    "id": 1,  // Single paper number only
                    "context": "Brief description of what you're citing"
                }}
            ]
        }}

        Papers to analyze:
        {papers_text}"""

        # Make the API call
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                response_format={"type": "json_object"},  # Force JSON response
                messages=[
                    {"role": "system", "content": "You are a research assistant that produces structured summaries with precise citations. Each citation must reference exactly one paper."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Parse the response
            result = json.loads(response.choices[0].message.content)
            
            # Process citations to include full paper info
            processed_citations = []
            for citation in result.get("citations", []):
                # Ensure we have a single ID
                id_num = citation["id"]
                if not isinstance(id_num, int):
                    continue  # Skip invalid citations
                    
                idx = id_num - 1
                if 0 <= idx < len(papers):
                    processed_citations.append({
                        "id": id_num,
                        "paper_id": papers[idx].paper_id,
                        "title": papers[idx].title,
                        "url": papers[idx].url,
                        "context": citation.get("context", "")
                    })
            
            return {
                "summary": result.get("summary", ""),
                "citations": processed_citations
            }
            
        except Exception as e:
            return {
                "summary": f"Error generating summary: {str(e)}",
                "citations": []
            }
            
    def _generate_test_result(self, papers: list[Paper]):
        """Generate a test result if no matching test data is found"""
        citations = []
        summary_parts = []
        
        for i, paper in enumerate(papers, 1):
            citations.append({
                "id": i,
                "paper_id": paper.paper_id,
                "title": paper.title,
                "url": paper.url,
                "context": f"Key finding from paper {i}"
            })
            summary_parts.append(f"This paper discusses {paper.title} {{cite:{i}}}")
        
        return {
            "summary": " ".join(summary_parts),
            "citations": citations
        }

