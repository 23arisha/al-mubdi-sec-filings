import requests
from bs4 import BeautifulSoup
import re

# Helper Functions
def clean_text(text):
    text = re.sub(r'\xa0', ' ', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'(?<=\w)-\n(?=\w)', '', text)
    text = re.sub(r'(?<=\S)\n(?=\S)', ' ', text)
    text = re.sub(r'[•#]', '', text)
    text = re.sub(r'\n{2,}', '\n', text)
    return "\n".join([line.strip() for line in text.splitlines() if line.strip()])

def extract_section(text, start_pattern, end_pattern):
    start = re.search(start_pattern, text, re.IGNORECASE)
    end = re.search(end_pattern, text, re.IGNORECASE)
    if start and end:
        return text[start.end():end.start()].strip()
    elif start:
        return text[start.end():].strip()
    return ""

def extract_sections_from_10k_html(url):
    headers = {'User-Agent': 'your-email@example.com'}
    response = requests.get(url, headers=headers, timeout=20)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')
    for tag in soup(['script', 'style', 'table']):
        tag.decompose()
    text = clean_text(soup.get_text(separator='\n'))
    sections = {
        "Business Overview": extract_section(text, r"item\s+1\.*\s*business", r"item\s+1a\.*"),
        "Risk Factors": extract_section(text, r"item\s+1a\.*\s*risk factors", r"item\s+1b\.*"),
        "Legal Proceedings": extract_section(text, r"item\s+3\.*\s*legal proceedings", r"item\s+4\.*"),
        "MD&A": extract_section(text, r"item\s+7\.*\s+management.*?operations", r"item\s+7a\.*"),
        "Market Risk": extract_section(text, r"item\s+7a\.*.*?market risk", r"item\s+8\.*"),
        "Financial Statements": extract_section(text, r"item\s+8\.*\s*financial statements", r"item\s+9\.*"),
        "Controls and Procedures": extract_section(text, r"item\s+9a\.*\s*controls and procedures", r"item\s+9b\.*"),
        "Executive Compensation": extract_section(text, r"item\s+11\.*\s*executive compensation", r"item\s+12\.*"),
    }
    return " ".join([f"{k}\n{v}" for k, v in sections.items() if v])

def extract_form4_text(url):
    """Fetch and clean Form 4 filing text from SEC URL."""
    headers = {'User-Agent': 'your-email@example.com'}
    response = requests.get(url, headers=headers, timeout=20)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')
    for tag in soup(['script', 'style']):
        tag.decompose()
    text = clean_text(soup.get_text(separator='\n'))
    return text