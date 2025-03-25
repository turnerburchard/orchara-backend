import re
import asyncio
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
import aiohttp
import io
from PyPDF2 import PdfReader

def clean_text(text):
    """
    Clean extracted text by removing extra whitespace, LaTeX notation, etc.
    """
    if not text:
        return text
        
    # Remove LaTeX math notation
    text = re.sub(r'\\\(.*?\\\)', '', text)
    text = re.sub(r'\$.*?\$', '', text)
    
    # Remove citations
    text = re.sub(r'\[\d+(?:,\s*\d+)*\]', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Remove common artifacts
    text = text.replace('Abstract.', '').replace('Abstract:', '')
    text = text.strip()
    
    return text

async def download_pdf(url):
    """
    Downloads a PDF file from a URL and returns it as a bytes object.
    """
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    return await response.read()
                return None
    except Exception as e:
        print(f"Error downloading PDF: {str(e)}")
        return None

async def extract_abstract_from_pdf(pdf_bytes):
    """
    Extracts the abstract from a PDF file.
    Returns the abstract text or None if not found.
    """
    try:
        # Read PDF from bytes
        pdf = PdfReader(io.BytesIO(pdf_bytes))
        
        # Get text from first page
        first_page = pdf.pages[0]
        text = first_page.extract_text()
        
        # Try to find abstract section with various patterns
        patterns = [
            r'(?i)abstract[:\s]*(.*?)(?:\n\n|\n(?:keywords|introduction|background))',
            r'(?i)abstract[:\s]*(.*?)(?=\n\s*[1I]\.?\s+[A-Z])',  # Look for section I or 1
            r'(?i)abstract[:\s]*(.*?)(?=\n\s*keywords:)',
            r'(?i)abstract[:\s]*(.*?)(?=\n\s*\d+\.?\s+[A-Z])'  # Look for numbered sections
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                abstract = match.group(1).strip()
                return clean_text(abstract)
        return None
    except Exception as e:
        print(f"Error extracting PDF abstract: {str(e)}")
        return None

async def get_final_url(doi_url):
    """
    Follows redirects to get the final URL, handling DOI redirects properly.
    """
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(doi_url, allow_redirects=True) as response:
                return str(response.url)
    except Exception as e:
        print(f"Error following redirects: {str(e)}")
        return doi_url

async def scrape_abstract(doi_url):
    """
    Scrapes the abstract from a DOI URL using Playwright with Chromium in headless mode.
    
    Args:
        doi_url (str): The DOI URL to scrape
        
    Returns:
        str: The extracted abstract text, or None if not found
    """
    try:
        # Follow redirects to get final URL
        final_url = await get_final_url(doi_url)
        print(f"Final URL after redirects: {final_url}")
        
        # Check if URL points directly to a PDF
        if final_url.lower().endswith('.pdf'):
            print("Detected direct PDF URL, attempting to extract abstract...")
            pdf_bytes = await download_pdf(final_url)
            if pdf_bytes:
                return await extract_abstract_from_pdf(pdf_bytes)
            return None
            
        # Extract DOI from URL for special cases
        doi_match = re.search(r'10\.\d{4,}/[^/]+(?:/[^/]+)*', doi_url)
        if doi_match:
            doi = doi_match.group(0)
            # For MDPI, construct URL directly
            if '10.3390/' in doi:
                journal_id = doi.split('/')[1]
                article = doi.split('/')[-1]
                final_url = f"https://www.mdpi.com/{journal_id}/pdf/{article}.pdf"
                print("Detected MDPI article, using PDF URL:", final_url)
                pdf_bytes = await download_pdf(final_url)
                if pdf_bytes:
                    return await extract_abstract_from_pdf(pdf_bytes)
        
        print(f"Accessing URL: {final_url}")
        
        async with async_playwright() as p:
            # Launch browser
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(
                user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            )
            page = await context.new_page()
            
            try:
                # For Nature, try a different wait strategy
                if 'nature.com' in final_url:
                    await page.goto(final_url, timeout=45000)  # Don't wait for networkidle
                    await page.wait_for_selector('div#Abs1-content, div.c-article-section__content, #abstract-content', timeout=10000)
                else:
                    await page.goto(final_url, wait_until='networkidle', timeout=45000)
                
                # Check if we were redirected to a PDF
                response = await page.goto(final_url)
                if response and response.headers.get('content-type', '').lower().startswith('application/pdf'):
                    print("Detected PDF content, attempting to extract abstract...")
                    pdf_bytes = await download_pdf(response.url)
                    if pdf_bytes:
                        return await extract_abstract_from_pdf(pdf_bytes)
                    return None
                
                await page.wait_for_load_state('domcontentloaded')
                await asyncio.sleep(2)  # Additional wait for JS rendering
                
                # Try publisher-specific patterns first
                if '10.3390/' in doi_url:  # MDPI
                    try:
                        abstract_element = await page.wait_for_selector('.art-abstract', timeout=5000)
                        if abstract_element:
                            text = await abstract_element.text_content()
                            return clean_text(text)
                    except:
                        print("Could not find MDPI abstract with art-abstract class")
                
                elif 'cell.com' in final_url or '10.1016' in doi_url:  # Cell/Elsevier
                    try:
                        abstract_element = await page.wait_for_selector('div#abs0010, div.abstract.author, section.abstract, .abstract.graphical, .abstract.svAbstract', timeout=5000)
                        if abstract_element:
                            text = await abstract_element.text_content()
                            return clean_text(text)
                    except:
                        print("Could not find Cell/Elsevier abstract")
                
                elif 'nature.com' in final_url or '10.1038' in doi_url:  # Nature
                    try:
                        abstract_element = await page.wait_for_selector('div#Abs1-content, div.c-article-section__content, #abstract-content', timeout=5000)
                        if abstract_element:
                            text = await abstract_element.text_content()
                            return clean_text(text)
                    except:
                        print("Could not find Nature abstract")
                
                elif 'aip.org' in final_url or '10.1063' in doi_url:  # AIP
                    try:
                        # Extract article ID from DOI
                        article_id = doi_url.split('10.1063/')[-1]
                        api_url = f"https://pubs.aip.org/api/v1/articles/10.1063/{article_id}"
                        
                        # Try to get metadata from AIP API
                        async with aiohttp.ClientSession() as session:
                            async with session.get(api_url) as response:
                                if response.status == 200:
                                    data = await response.json()
                                    if 'abstract' in data:
                                        return clean_text(data['abstract'])
                                    elif 'description' in data:
                                        return clean_text(data['description'])
                        
                        print("Could not find AIP abstract via API, falling back to page scraping...")
                        
                        # Try specific AIP selectors as fallback
                        selectors = [
                            'div.article-text > div.section > p:first-of-type',  # Main abstract text
                            'div.hlFld-Abstract p',  # Alternative abstract container
                            'section.abstract p:first-of-type',  # Another common pattern
                            'meta[name="citation_abstract"]',  # Meta tag
                            'meta[name="dc.description"]',  # Dublin Core meta tag
                            'div.article-content p:first-of-type',  # First paragraph of content
                            'div[role="main"] p:first-of-type'  # First paragraph of main content
                        ]
                        
                        # First try to wait for the abstract section to load
                        try:
                            await page.wait_for_selector('div.article-text, div.hlFld-Abstract, section.abstract', timeout=5000)
                        except:
                            print("Could not find main abstract container")
                        
                        # Then try each selector
                        for selector in selectors:
                            try:
                                abstract_element = await page.wait_for_selector(selector, timeout=2000)
                                if abstract_element:
                                    if selector.startswith('meta'):
                                        text = await abstract_element.get_attribute('content')
                                    else:
                                        text = await abstract_element.text_content()
                                    text = clean_text(text)
                                    # Validate the content
                                    if (text and len(text) > 50 and  # Reasonable length
                                        not any(x in text.lower() for x in ['views icon', 'download citation', 'search site', 'verify you are human']) and  # No navigation text
                                        not text.startswith('You do not') and  # No access messages
                                        not text.startswith('This content')):  # No content messages
                                        return text
                            except:
                                continue
                        
                        print("Could not find AIP abstract")
                    except Exception as e:
                        print(f"Error processing AIP article: {str(e)}")
                
                elif 'springer.com' in final_url or '10.1134' in doi_url:  # Springer
                    try:
                        # Try Springer-specific selectors
                        selectors = [
                            'div[data-title="Abstract"] p',  # Main abstract container
                            'section.Abstract p',  # Alternative abstract section
                            'div.c-article-section__content p',  # Content section
                            'meta[name="citation_abstract"]',  # Meta tag
                            'meta[name="dc.description"]',  # Dublin Core meta tag
                            'div[class*="abstract"] p'  # Any div with abstract in class
                        ]
                        for selector in selectors:
                            try:
                                abstract_element = await page.wait_for_selector(selector, timeout=2000)
                                if abstract_element:
                                    if selector.startswith('meta'):
                                        text = await abstract_element.get_attribute('content')
                                    else:
                                        text = await abstract_element.text_content()
                                    text = clean_text(text)
                                    if text and len(text) > 100 and not text.startswith('To illustrate'):
                                        return text
                            except:
                                continue
                        print("Could not find Springer abstract")
                    except:
                        print("Error processing Springer article")
                
                # Get the page content and parse with BeautifulSoup as fallback
                content = await page.content()
                soup = BeautifulSoup(content, 'html.parser')
                
                print("\nSearching for abstract...")
                
                # Try common abstract patterns
                abstract_patterns = [
                    {'class': re.compile(r'abstract', re.I)},
                    {'id': re.compile(r'abstract', re.I)},
                    {'section': re.compile(r'abstract', re.I)},
                    {'name': re.compile(r'abstract', re.I)},
                    {'class': 'abstract-content'},
                    {'class': 'article-abstract'},
                    {'class': 'abstract-text'},
                    {'class': 'abstractSection'},
                    {'id': 'abstract-content'},
                    {'property': 'og:description'},
                    {'name': 'description'},
                    {'class': 'first-paragraph'},
                    {'class': 'article-text'},
                    {'class': 'content-paragraph'}
                ]
                
                for pattern in abstract_patterns:
                    print(f"Trying pattern: {pattern}")
                    elements = soup.find_all(attrs=pattern)
                    print(f"Found {len(elements)} matching elements")
                    
                    for element in elements:
                        # Remove any nested headings
                        for heading in element.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
                            heading.decompose()
                            
                        # Clean up the text
                        text = element.get_text(strip=True)
                        text = clean_text(text)
                        
                        if text and len(text) > 100:  # Only return if it looks like a real abstract
                            print(f"Found abstract with length {len(text)} characters")
                            return text
                
                print("\nTrying paragraph search...")
                # Try finding paragraphs that look like abstracts
                paragraphs = soup.find_all('p')
                print(f"Found {len(paragraphs)} paragraphs")
                
                for p in paragraphs:
                    text = p.get_text(strip=True)
                    if len(text) > 100 and ('abstract' in text.lower() or 'summary' in text.lower()):
                        text = clean_text(text)
                        print(f"Found abstract in paragraph with length {len(text)} characters")
                        return text
                
                return None
                
            finally:
                await context.close()
                await browser.close()
        
    except Exception as e:
        print(f"Error scraping abstract: {str(e)}")
        return None

async def main():
    test_urls = [
        "https://dx.doi.org/10.3390/land13040551",  # MDPI
        "https://doi.org/10.1016/j.heliyon.2024.e25471",  # Elsevier (Open Access)
        "https://doi.org/10.1038/s41598-024-54245-z",  # Nature Scientific Reports (Open Access)
        "https://doi.org/10.1126/sciadv.adi8338"  # Science Advances (Open Access)
    ]
    
    for url in test_urls:
        print(f"\nTesting URL: {url}")
        print("=" * 80)
        abstract = await scrape_abstract(url)
        if abstract:
            print("\nScraped Abstract:")
            print("-" * 80)
            print(abstract)
            print("-" * 80)
        else:
            print("No abstract found")
        print("=" * 80)
        await asyncio.sleep(2)  # Add a small delay between requests

# Test the function with the provided URLs
if __name__ == "__main__":
    asyncio.run(main()) 
    asyncio.run(main()) 