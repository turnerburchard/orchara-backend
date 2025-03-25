import asyncio
from abstract_scraper.abstract_scraper import scrape_abstract

async def test_urls():
    urls = [
        'http://dx.doi.org/10.22331/q-2024-04-30-1324',  # Quantum journal
        'http://dx.doi.org/10.1063/1.1491289',  # AIP
        'http://dx.doi.org/10.1038/s41598-024-60321-1',  # Nature Scientific Reports
        'http://dx.doi.org/10.1134/s0021364024600216'  # JETP Letters
    ]
    
    for url in urls:
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

if __name__ == "__main__":
    asyncio.run(test_urls()) 