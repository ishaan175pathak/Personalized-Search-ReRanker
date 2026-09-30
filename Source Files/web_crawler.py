import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import re

def fetch_page(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=5)
        if response.status_code == 200 and 'text/html' in response.headers['Content-Type']:
            return response.text
    except Exception as e:
        print(f"Failed to fetch {url}: {e}")
    return None

def extract_links(base_url, html):
    soup = BeautifulSoup(html, 'html.parser')
    links = set()
    for a_tag in soup.find_all('a', href=True):
        href = a_tag['href']
        full_url = urljoin(base_url, href)
        if urlparse(full_url).scheme in ('http', 'https'):
            links.add(full_url)
    return links

def extract_text(html):
    soup = BeautifulSoup(html, 'html.parser')
    for script in soup(["script", "style"]):
        script.decompose()
    return soup.get_text(separator=' ', strip=True)

def crawl(seed_url, max_pages=10):
    visited = set()
    to_visit = [seed_url]
    crawled_data = []

    while to_visit and len(visited) < max_pages:
        url = to_visit.pop(0)
        if url in visited:
            continue
        html = fetch_page(url)
        if html:
            visited.add(url)
            text = extract_text(html)
            crawled_data.append({"url": url, "content": text})
            links = extract_links(url, html)
            for link in links:
                if link not in visited and len(to_visit) + len(visited) < max_pages:
                    to_visit.append(link)
    return crawled_data

if __name__ == '__main__':
    # example query to check the functionality of the system 
    seed = "https://en.wikipedia.org/wiki/Natural_language_processing"
    results = crawl(seed, max_pages=5)
    for page in results:
        print(f"\nURL: {page['url']}\nContent Snippet: {page['content'][:300]}...")
