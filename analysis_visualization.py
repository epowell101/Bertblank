import numpy as np
import faiss
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import logging
import time
import torch
import pandas as pd
import plotly.express as px
from scraping_processing import search_for_query_embeddings, ensure_dimension
import PyPDF2


def extract_title_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        metadata = reader.getDocumentInfo()
        title = metadata.get('/Title')
        if title:
            return title
        else:
            # Attempt to extract the title from the text (e.g., first non-empty line)
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text = page.extractText().strip()
                lines = text.split('\n')
                for line in lines:
                    if line.strip():
                        return line.strip()
    return None

def search_with_embeddings(query_embeddings, faissindex, k=5):
    # Search in the FAISS index
    query_embeddings=ensure_dimension(query_embeddings)
    D, I = faissindex.search(query_embeddings, k)
    return D, I

# def interpret_results(search_terms, I, pdf_urls):
#    for term_idx, term in enumerate(search_terms):
#        print(f"Search term: {term}")
#        for neighbor_idx in I[term_idx]:
#            pdf_url = pdf_urls[neighbor_idx]
#            title = extract_title_from_pdf(pdf_url)
#            print(f"  - Document {neighbor_idx}: {pdf_url}")
#            if title:
#                print(f"    - Title: {title}")


def embed_search_terms(search_terms, bert_tokenizer, bert_model):
    search_embeddings = []
    for term in search_terms:
        # Tokenize the term
        inputs = bert_tokenizer(term, return_tensors="pt", padding=True, truncation=True)
        # Pass the tokenized input to the BERT model
        with torch.no_grad():
            outputs = bert_model(**inputs)
            # Averaging the last hidden state or other desired processing
            embedding = np.mean(outputs.last_hidden_state.cpu().numpy(), axis=1)
            # embedding = np.squeeze(embedding) # Squeeze out the unnecessary dimension
            # Ensuring the correct dimension
            embedding = ensure_dimension(embedding)
            search_embeddings.append(embedding)
    return np.array(search_embeddings) # make sure the result is an array, not a list

def visualize_embeddings(embeddings, titles=None):
    # Apply t-SNE for dimensionality reduction to 3 components
    if isinstance(embeddings, list):
        embeddings = np.vstack(embeddings)
    print("Embeddings shape:", embeddings.shape)
    tsne = TSNE(n_components=3, perplexity=1, random_state=0)
    reduced_embeddings = tsne.fit_transform(embeddings)

    # Create a DataFrame to hold the reduced embeddings and titles
    df_embeddings = pd.DataFrame(reduced_embeddings, columns=['Component 1', 'Component 2', 'Component 3'])
    if titles is not None:
        df_embeddings['Title'] = titles

    # Create the 3D scatter plot
    hover_data = ['Title'] if titles is not None else None
    fig = px.scatter_3d(df_embeddings, x='Component 1', y='Component 2', z='Component 3', hover_data=hover_data)
    fig.show()


def analyze_and_visualize(embeddings, search_terms, pdf_urls, bert_tokenizer, bert_model, faissindex, k=5):
    logging.debug('analyze and visualize function')
    
    # Embed the search terms using BERT
    search_embeddings = embed_search_terms(search_terms, bert_tokenizer, bert_model)

    # Perform the search using the query embeddings and the existing FAISS index
    D, I = search_for_query_embeddings(search_embeddings, faissindex, k)

    # interpret_results(search_terms, I, pdf_urls)
    visualize_embeddings(embeddings)


def display_results(D, I, pdf_urls):
    logging.debug('display results function')
    print(f"Top {len(I[0])} documents related to the query:")
    for i, idx in enumerate(I[0]):
        likelihood_score = D[0][i]
        print(f"  - Document {idx}: {pdf_urls[idx]}, Likelihood Score: {likelihood_score}")


