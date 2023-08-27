import requests
from bs4 import BeautifulSoup
import PyPDF2
from io import BytesIO
import torch
import faiss
from tqdm import tqdm
import numpy as np
import logging
import time
from bertopic import BERTopic
from transformers import BertTokenizer, BertModel, BartTokenizer, BartForConditionalGeneration
import numpy as np

def ensure_dimension(embeddings, desired_dimension=3072):
    if len(embeddings.shape) == 1:  # If 1D array, expand dimensions
        embeddings = np.expand_dims(embeddings, axis=0)

    if embeddings.shape[1] == 1:  # Squeeze if the middle dimension is 1
        embeddings = np.squeeze(embeddings, axis=1)

    current_dimension = embeddings.shape[1]  # Capture the second dimension after potential squeezing

    if current_dimension != desired_dimension:
        if current_dimension == 768:  # if current is 768 and desired is 3072
            # Concatenating the same embeddings four times to get to 3072
            embeddings = np.concatenate([embeddings] * 4, axis=1)
        elif current_dimension == 3072 and desired_dimension == 768:  # if current is 3072 and desired is 768
            # Averaging the embeddings to reduce dimension to 768
            embeddings = embeddings.reshape(-1, 4, 768).mean(axis=1)
        else:
            raise ValueError(f"Unexpected dimensions: current {current_dimension}, desired {desired_dimension}")

    return embeddings

# Function to summarize using BART
def summarize(text, bart_tokenizer, bart_model):
    inputs = bart_tokenizer([text], max_length=1024, return_tensors='pt', truncation=True)
    summary_ids = bart_model.generate(inputs['input_ids'], num_beams=4, min_length=30, max_length=150, early_stopping=True)
    summary = bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def search_for_query_embeddings(query_embeddings, faissindex, k=5):
    # Ensure that query_embeddings is a NumPy array
    query_embeddings = np.array(query_embeddings)

    # Check the shape and type of query_embeddings
    print("Query embeddings shape:", query_embeddings.shape)
    print("Query embeddings type:", type(query_embeddings))
    query_embeddings= ensure_dimension(query_embeddings, 3072)
    result = faissindex.search(query_embeddings, k)
    print("Search result:", result) # Check what is actually returned

    D, I = result # Unpack the result
    return D, I

def search_for_query(query, bert_tokenizer, bert_model, faissindex, k=10):
    logging.debug('search for query function')
    inputs = bert_tokenizer(query, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    query_embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy().reshape(1, -1)
    query_embedding=ensure_dimension(query_embedding)
    # Search in the FAISS index
    D, I = faissindex.search(query_embedding, k)
    
    # Return distances (likelihood scores) and indices of documents
    return D, I

def get_pdf_urls(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    table = soup.find('table')
    pdf_urls = [a['href'] for a in table.find_all('a') if 'pdf' in a['href']]
    return pdf_urls

# def use_BERTopic(documents):
    # using BERTopic for visualization and as a comparison
    # topic_model = BERTopic(language="english", calculate_probabilities=True, verbose=True)
    # topics = topic_model.fit_transform(documents)
    # freq = topic_model.get_topic_info()
    # print(freq.head(10))
    # print(topic_model.topics_[:15])
    # topic_model.visualize_topics()
    # topic_model.visualize_hierarchy(top_n_topics=25)
    # topic_model.visualize_heatmap(n_clusters=5, width=1000, height=1000)
    # return topics

def read_pdf(pdf_url):
    response = requests.get(pdf_url)
    pdf_reader = PyPDF2.PdfReader(BytesIO(response.content))
    text = "".join(page.extract_text() for page in pdf_reader.pages)
    return text

def process_embeddings(embeddings, save_path='c:\\cloudODC\\PDPdata\\processed_embeddings.npy'):
    # processed_embeddings = np.mean(embeddings, axis=0).reshape(1, -1)
    np.save(save_path, embeddings)
    print("Shape of embeddings received:", embeddings.shape)
    print("Type of embeddings received:", type(embeddings))
    return 

def get_documents_from_url(url):
    pdf_urls = get_pdf_urls(url)
    documents = [{'url': pdf_url, 'text': read_pdf(pdf_url)} for pdf_url in pdf_urls]
    return documents

def add_to_faiss_index(faissindex, embeddings):
    print(f"Shape of embeddings array: {embeddings.shape}")
    print(type(faissindex), "within add_to_faiss")
    embeddings = ensure_dimension (embeddings, 3072)
    faissindex.add(embeddings)

# Function to process PDFs
def process_pdfs(pdf_urls, bart_tokenizer, bart_model, bert_tokenizer, bert_model, faissindex):
    recent_summaries = []
    documents = [] # List to collect the PDF texts
    gather_embeddings =[]
    for pdf_url in tqdm(pdf_urls[-15:]): # Limiting to x for testing
        pdf_text = read_pdf(pdf_url)
        documents.append(pdf_text) # Adding the text to each document

        # Creating summary using BART
        summary = summarize(pdf_text, bart_tokenizer, bart_model)
        recent_summaries.append((pdf_url, summary))
        
        # Creating embeddings using BERT
        embeddings = embed_text(pdf_text, bert_tokenizer, bert_model)
        embeddings = ensure_dimension(embeddings, 3072)
        gather_embeddings.append(embeddings)
        add_to_faiss_index(faissindex, embeddings)

    # Using BERTopic on the documents
    # topics = use_BERTopic(documents)
    return recent_summaries, gather_embeddings

# Function to create embeddings using BERT 
def embed_text(pdf_text, bert_tokenizer, bert_model):
    text_units = pdf_text.split('\n')
    text_units = [unit for unit in text_units if unit.strip() != '']
    concatenated_embeddings = []

    for unit in text_units:
        inputs = bert_tokenizer(unit, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = bert_model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            
            # Concatenating the last four layers for each token
            concatenated_layers = torch.cat((
                hidden_states[-4],
                hidden_states[-3],
                hidden_states[-2],
                hidden_states[-1]), dim=-1)
            
            # Averaging over the tokens to get a single vector representation for the entire unit
            mean_embedding = torch.mean(concatenated_layers, dim=1).cpu().numpy()
            concatenated_embeddings.append(mean_embedding)

    # Stack the embeddings for all units
    document_embedding = np.vstack(concatenated_embeddings)
    document_embedding = ensure_dimension(document_embedding,3072)
    return document_embedding
