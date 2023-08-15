import requests
from bs4 import BeautifulSoup
import PyPDF2
from io import BytesIO
import torch
from transformers import BartForConditionalGeneration, BartTokenizer
import faiss
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

def get_pdf_urls(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    table = soup.find('table')
    pdf_urls = [a['href'] for a in table.find_all('a') if 'pdf' in a['href']]
    return pdf_urls

def read_pdf(pdf_url):
    response = requests.get(pdf_url)
    pdf_reader = PyPDF2.PdfReader(BytesIO(response.content))
    text = "".join(page.extract_text() for page in pdf_reader.pages)
    return text

def summarize_and_extract_embeddings(text):
    inputs = tokenizer([text], max_length=1024, return_tensors='pt', truncation=True)
    summary_ids = model.generate(inputs['input_ids'], num_beams=4, min_length=30, max_length=150, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    encoder_outputs = model.get_encoder()(inputs['input_ids'])
    embeddings = encoder_outputs.last_hidden_state.mean(dim=1)
    return summary, embeddings

def process_embeddings(embeddings, save_path='c:\cloudODC\PDPdata\processed_embeddings.npy'):
    processed_embeddings = embeddings.mean(dim=0).cpu().detach().numpy().reshape(1, -1)
    np.save(save_path, processed_embeddings)
    return processed_embeddings



def is_directed(text, threshold_X=2):
    keywords = ["directed graph", "directed edges", "directed nodes"]
    count = sum(text.lower().count(keyword) for keyword in keywords)
    return count > threshold_X

def is_heterogeneous(text, threshold_X=2):
    keywords = ["heterogeneous graph", "mixed types", "multiple types"]
    count = sum(text.lower().count(keyword) for keyword in keywords)
    return count > threshold_X

def is_continuous(text, threshold_X=2):
    keywords = ["continuous graph", "continuous data", "continuous nodes"]
    count = sum(text.lower().count(keyword) for keyword in keywords)
    return count > threshold_X

def get_documents_from_url(url):
    pdf_urls = get_pdf_urls(url)
    documents = [{'url': pdf_url, 'text': read_pdf(pdf_url)} for pdf_url in pdf_urls]
    return documents

tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

# FAISS index
d = 1024
index = faiss.IndexFlatL2(d)

# Gather based on URL
url = 'https://github.com/epowell101/graph-fraud-detection-papers'
pdf_urls = get_pdf_urls(url)

recent_summaries = []

for pdf_url in tqdm(pdf_urls[-5:]): # Limiting to last 5 for printing recent summaries
    # Read the PDF
    pdf_text = read_pdf(pdf_url)
    
    # Summarize and extract embeddings
    summary, embeddings = summarize_and_extract_embeddings(pdf_text)
    
    # Store the summary
    recent_summaries.append((pdf_url, summary))
    
    # Store embeddings in FAISS (after processing them if needed)
    processed_embeddings = process_embeddings(embeddings)
    processed_embeddings = embeddings.cpu().detach().numpy().reshape(1, -1)  # Detach from computation graph and reshape
  # Detach from computation graph and reshape
    index.add(processed_embeddings)

# Print recent summaries
for url, summary in recent_summaries:
    print(f"URL: {url}")
    print(f"Summary: {summary}\n")


# URL containing PDF links
url = 'https://github.com/epowell101/graph-fraud-detection-papers'

# Gather and process documents
documents = get_documents_from_url(url)
recent_summaries = []
for doc in documents[-5:]:
    summary, embeddings = summarize_and_extract_embeddings(doc['text'])
    recent_summaries.append((doc['url'], summary))
    processed_embeddings = process_embeddings(embeddings).reshape(1, -1)
    index.add(processed_embeddings)

# Analyze and print results
counts = defaultdict(int)
classified_docs = defaultdict(list)
for url, summary in recent_summaries:
    for func, criteria in [(is_directed, 'directed'), (is_heterogeneous, 'heterogeneous'), (is_continuous, 'continuous')]:
        if func(summary):
            counts[criteria] += 1
            classified_docs[criteria].append(url)
print("Criteria\tCount\tLinks")
for criteria, count in counts.items():
    links = ", ".join(classified_docs[criteria])
    print(f"{criteria}\t{count}\t{links}")


def count_keywords(text, keywords):
    counts = {keyword: text.lower().count(keyword) for keyword in keywords}
    return counts

# List of keywords to count
keywords = ["directed graph", "directed edges", "directed nodes", 
            "heterogeneous graph", "mixed types", "multiple types",
            "continuous graph", "continuous data", "continuous nodes"]

# Collect counts across all summaries
all_counts = defaultdict(int)
for url, summary in recent_summaries:
    counts = count_keywords(summary, keywords)
    for keyword, count in counts.items():
        all_counts[keyword] += count

# Plotting the histogram
plt.bar(all_counts.keys(), all_counts.values())
plt.xticks(rotation=90)
plt.xlabel('Keywords')
plt.ylabel('Counts')
plt.title('Keyword Counts in Summaries')
plt.show()
