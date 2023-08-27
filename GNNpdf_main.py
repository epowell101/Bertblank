from analysis_visualization import display_results
from scraping_processing import get_pdf_urls, process_pdfs, search_for_query
import os
import logging
import faiss
import numpy as np
import time
import pickle
from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import BertTokenizer, BertModel

def check_for_existing_embeddings(path):
    if os.path.exists(path):
        with open(path, 'rb') as f:
            try:
                return pickle.load(f)
            except EOFError:
                print(f"Failed to load embeddings from {path}. The file may be empty or corrupt.")
                return None
    else:
        return None

def check_for_existing_summaries(summaries_path):
    if os.path.exists(summaries_path):
        with open(summaries_path, 'r') as file:
            summaries = [line.strip() for line in file]
        return summaries
    return None

def main():

    logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')
    
    logging.getLogger('matplotlib').setLevel(logging.WARNING)

    logging.info('Starting the application...')

    k = 10
    desired_dimension = 3072
    faissindex = faiss.IndexFlatL2(desired_dimension)

    url = 'https://github.com/epowell101/graph-fraud-detection-papers'
    embeddings_path = 'c:\\cloudODC\\PDPdata\\processed_embeddings.npy'
    summaries_path = 'c:\\cloudODC\\PDPdata\\processed_summaries.txt'
    search_terms = ["heterogeneous", "dynamic", "static", "temporal", "homogeneous", "attention","transformer","supervised","self-supervised"]
    
    pdf_urls=get_pdf_urls(url)

    # BART for Summarization
    bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
    bart_model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

    # BERT for Embeddings
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased')

    # Check for existing embeddings & summaries
    final_embeddings = check_for_existing_embeddings(embeddings_path)
    summaries = check_for_existing_summaries(summaries_path)

    # If either embeddings or summaries are missing, process PDFs
    if final_embeddings is None or summaries is None:
        start_time = time.time()
        recent_summaries, final_embeddings = process_pdfs(pdf_urls, bart_tokenizer, bart_model, bert_tokenizer, bert_model, faissindex)
        elapsed_time = time.time() - start_time
        print(f"Elapsed time: {elapsed_time} seconds")
        # If summaries are missing, update summaries
        if summaries is None:
            summaries = [summary for url, summary in recent_summaries]
            with open(summaries_path, 'w',encoding='utf-8') as file:
                for summary in summaries:
                    file.write(f"{summary}\n")

    # If embeddings have been created in this run, save them using pickle
    if final_embeddings is not None:
        with open(embeddings_path, 'wb') as f:
            pickle.dump(final_embeddings, f)

    # Perform a search query
    query = "heterogeneous directed temporal"
    D, I = search_for_query(query, bert_tokenizer, bert_model, faissindex)
    display_results(D, I,pdf_urls)

if __name__ == '__main__':
    main()

def get_top_document(titles, summaries, search_result):
    distances, indices = search_result
    # Get the index of the top document (closest embedding)
    top_index = indices[0, 0]
    # Retrieve the title and summary using the top index
    top_title = titles[top_index]
    top_summary = summaries[top_index]
    return top_title, top_summary


