from openai import OpenAI
import json
import os
from dotenv import load_dotenv

# TODO turn into common State class which passes to all instances?
testing_mode = False

class Summarizer:
    def __init__(self):
        load_dotenv()  # Load environment variables
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        self.client = OpenAI(api_key=api_key)

    def summarize(self, text):
        if testing_mode:
            summary = "The research papers explore machine learning algorithms, categorizing and comparing their performance in supervised settings. They survey applications of these algorithms, focusing on the fundamental components and principles of machine learning workflows that handle both numerical and categorical data. This summary highlights essential insights into algorithm effectiveness and data processing methodologies."
        else:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "user", "content": f"You are a researcher, accurately summarize the following set of research papers into a single paragraph: {text}"}
                ]
            )
            summary = response.choices[0].message.content

        return summary
        
    def summarize_with_citations(self, papers):
        """
        Generate a summary with citations for a list of papers.
        
        Args:
            papers: list of dicts, each with keys:
                - paper_id: unique identifier
                - title: paper title
                - abstract: paper abstract
                - url: link to the paper
                
        Returns:
            dict with:
                - summary: text with citations in {{cite:X}} format
                - citations: list of citation objects with paper details
        """
        if testing_mode:
            return {
                "summary": "Recent advances in machine learning models have shown significant improvements in natural language processing tasks {{cite:1}}. These models utilize transformer architectures to achieve state-of-the-art results on benchmark datasets {{cite:2}}. Despite these advances, challenges remain in computational efficiency {{cite:4}} and handling of long-form text {{cite:5}}.",
                "citations": [
                    {
                        "id": 1,
                        "paper_id": papers[0]['paper_id'],
                        "title": papers[0]['title'],
                        "url": papers[0]['url']
                    },
                    {
                        "id": 2,
                        "paper_id": papers[1]['paper_id'],
                        "title": papers[1]['title'],
                        "url": papers[1]['url']
                    },
                    {
                        "id": 4,
                        "paper_id": papers[3]['paper_id'],
                        "title": papers[3]['title'],
                        "url": papers[3]['url']
                    },
                    {
                        "id": 5,
                        "paper_id": papers[4]['paper_id'],
                        "title": papers[4]['title'],
                        "url": papers[4]['url']
                    }
                ]
            }
            
        # Format papers for the prompt
        formatted_papers = []
        for i, paper in enumerate(papers, 1):
            formatted_papers.append(f"Paper {i}: \"{paper['title']}\"\nAbstract: {paper['abstract']}\n")
        
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
                        "paper_id": papers[idx]['paper_id'],
                        "title": papers[idx]['title'],
                        "url": papers[idx]['url'],
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

