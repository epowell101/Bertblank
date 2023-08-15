from scraping_processing import get_pdf_urls, embed_search_terms, summarize_and_extract_embeddings, process_pdfs, create_faiss_index
from analysis_visualization import analyze_and_visualize, search_for_query, display_results

import numpy as np
import os
import logging
import faiss
from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import BertTokenizer, BertModel


def check_for_existing_embeddings(path):
    if os.path.exists(path):
        return np.load(path)
    return None

def check_for_existing_summaries(summaries_path):
    if os.path.exists(summaries_path):
        with open(summaries_path, 'r') as file:
            summaries = [line.strip() for line in file]
        return summaries
    return None

def main():

    logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

    logging.info('Starting the application...')

    d = 768
    index = faiss.IndexFlatL2(d)

    url = 'https://github.com/epowell101/graph-fraud-detection-papers'
    embeddings_path = 'c:\\cloudODC\\PDPdata\\processed_embeddings.npy'
    summaries_path = 'c:\\cloudODC\\PDPdata\\processed_summaries.txt'

    # BART for Summarization
    bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
    bart_model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

    # BERT for Embeddings
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased')

    pdf_urls = get_pdf_urls(url)

    # Check for existing embeddings & summaries
    embeddings = check_for_existing_embeddings(embeddings_path)
    summaries = check_for_existing_summaries(summaries_path)

    # If either embeddings or summaries are missing, process PDFs
    if embeddings is None or summaries is None:
        recent_summaries, bert_embeddings = process_pdfs(pdf_urls, bart_tokenizer, bart_model, bert_tokenizer, bert_model, index)
    
        # If summaries are missing, update summaries
        if summaries is None:
            summaries = [summary for url, summary in recent_summaries]
            with open(summaries_path, 'w') as file:
                for summary in summaries:
                    file.write(f"{summary}\n")

        # If embeddings are missing, update embeddings
        if embeddings is None:
            embeddings = bert_embeddings
            np.save(embeddings_path, embeddings)

    # Create a search index
    index = create_faiss_index(embeddings)

    # Analyze and visualize the embeddings
    analyze_and_visualize(embeddings)

    # Perform a search query
    query = "heterogeneous directed temporal graphs"
    D, I = search_for_query(query, bert_tokenizer, bert_model, index)
    display_results(D, I, pdf_urls)

if __name__ == '__main__':
    main()



