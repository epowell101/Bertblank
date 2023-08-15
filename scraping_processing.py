import requests
from bs4 import BeautifulSoup
import PyPDF2
from io import BytesIO
import torch
import faiss
from tqdm import tqdm
import numpy as np
import logging
from transformers import BertTokenizer, BertModel, BartTokenizer, BartForConditionalGeneration

# Function to summarize using BART and extract embeddings
def summarize_and_extract_embeddings(text, bart_tokenizer, bart_model):
    inputs = bart_tokenizer([text], max_length=1024, return_tensors='pt', truncation=True)
    summary_ids = bart_model.generate(inputs['input_ids'], num_beams=4, min_length=30, max_length=150, early_stopping=True)
    summary = bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    encoder_outputs = bart_model.get_encoder()(inputs['input_ids'])
    embeddings = encoder_outputs.last_hidden_state.mean(dim=1)
    return summary, embeddings

# Function to create embeddings using BERT
def embed_search_terms(search_terms, bert_tokenizer, bert_model):
    search_embeddings = []
    for term in search_terms:
        inputs = bert_tokenizer(term, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = bert_model(**inputs)
        search_embeddings.append(outputs.last_hidden_state.mean(dim=1).cpu().numpy().reshape(1, -1))
    return np.vstack(search_embeddings)

def prepare_search_index(model_name='bert-base-uncased'):
    # Load Pre-trained BERT Model (or another model specified by model_name)
    bert_tokenizer = BertTokenizer.from_pretrained(model_name)
    bert_model = BertModel.from_pretrained(model_name)

    # Initialize FAISS Index
    dimension = bert_model.config.hidden_size  # Dimensionality of the embeddings
    index = faiss.IndexFlatIP(dimension)

    return bert_tokenizer, bert_model, index

def create_faiss_index(document_embeddings):
    # Normalize embeddings to enable cosine similarity with dot product
    faiss.normalize_L2(document_embeddings)

    # Create index with inner product similarity
    index = faiss.IndexFlatIP(document_embeddings.shape[1])
    index = faiss.IndexIDMap(index)
    index.add_with_ids(document_embeddings, np.array(range(len(document_embeddings))))
    return index

def search_embeddings(search_embeddings, index, k=5):
    D, I = index.search(search_embeddings, k)
    return D, I

def search_for_query(query, bert_tokenizer, bert_model, index, k=10):
    logging.debug('search for query function')
    inputs = bert_tokenizer(query, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    query_embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy().reshape(1, -1)
    
    # Search in the FAISS index
    D, I = index.search(query_embedding, k)
    
    # Return distances (likelihood scores) and indices of documents
    return D, I

def interpret_results(search_terms, I, pdf_urls):
    for term_idx, term in enumerate(search_terms):
        print(f"Search term: {term}")
        for neighbor_idx in I[term_idx]:
            print(f"  - Document {neighbor_idx}: {pdf_urls[neighbor_idx]}")

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

def process_embeddings(embeddings, save_path='c:\\cloudODC\\PDPdata\\processed_embeddings.npy'):
    processed_embeddings = embeddings.mean(dim=0).cpu().detach().numpy().reshape(1, -1)
    np.save(save_path, processed_embeddings)
    return processed_embeddings

def get_documents_from_url(url):
    pdf_urls = get_pdf_urls(url)
    documents = [{'url': pdf_url, 'text': read_pdf(pdf_url)} for pdf_url in pdf_urls]
    return documents

def add_to_faiss_index(index, embeddings):
    processed_embeddings = embeddings.cpu().detach().numpy().reshape(1, -1)  # Detach from computation graph and reshape
    index.add(processed_embeddings)

# Function to process PDFs
def process_pdfs(pdf_urls, bart_tokenizer, bart_model, bert_tokenizer, bert_model, index):
    recent_summaries = []
    bert_embeddings = []
    for pdf_url in tqdm(pdf_urls[-5:]): # Limiting to last 5 for printing recent summaries
        pdf_text = read_pdf(pdf_url)

        # Creating summary using BART
        summary, _ = summarize_and_extract_embeddings(pdf_text, bart_tokenizer, bart_model)
        recent_summaries.append((pdf_url, summary))
        
        # Creating embeddings using BERT
        embeddings = embed_search_terms([pdf_text], bert_tokenizer, bert_model)
        processed_embeddings = process_embeddings(embeddings)
        bert_embeddings.append(processed_embeddings)
        add_to_faiss_index(index, processed_embeddings)

    return recent_summaries, bert_embeddings

