def get_summary_prompt(text: str) -> str:
    return f"""You are a researcher, accurately summarize the following set of research papers into a single paragraph: {text}"""

def get_summary_with_citations_prompt(papers_text: str, query: str | None = None) -> str:
    query_context = f"\nOriginal search query: \"{query}\"\n" if query else ""
    
    return f"""Analyze these research papers and provide:
    1. A 2-paragraph summary that synthesizes the main findings, focusing on relevance to the original search query
       - Write in a natural, flowing style
       - Avoid using phrases like "Paper X" or "the first paper"
       - Instead, introduce findings naturally and cite them using {{{{cite:X}}}} format
    2. For each key point, cite exactly ONE paper number using {{{{cite:X}}}} format
       - Do not combine multiple papers in a single citation
       - Each citation should reference a single paper
       - Prioritize citing papers that are most relevant to the search query
    
    YOUR RESPONSE MUST BE IN THIS JSON FORMAT:
    {{
        "summary": "The synthesized text with citations marked as {{{{cite:X}}}}",
        "citations": [
            {{
                "id": 1,  // Must be a number matching the citation in the text
                "paper_id": "1",  // Must be a string
                "title": "Paper Title",  // Must be a string
                "url": "http://example.com",  // Must be a string
                "context": "Brief description of what you're citing"  // Must be a string
            }}
        ]
    }}

    IMPORTANT:
    - Each citation in the summary must have a matching entry in the citations array
    - The citation ID in the text ({{{{cite:X}}}}) must match the id in the citations array
    - All fields in each citation object are required
    - Focus on synthesizing information that is most relevant to the search query
    - Write in a natural, flowing style without explicitly referencing paper numbers in the text

    Papers to analyze:{query_context}
    {papers_text}"""

def get_summary_system_prompt() -> str:
    return """You are a research assistant that produces structured summaries with precise citations. Each citation must reference exactly one paper. Focus on synthesizing information that is most relevant to the user's search query.""" 