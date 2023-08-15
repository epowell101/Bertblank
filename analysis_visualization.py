import numpy as np
import faiss
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import logging

def search_embeddings(search_embeddings, document_embeddings, k=5):
    index = faiss.IndexFlatIP(search_embeddings.shape[1])
    index = faiss.IndexIDMap(index)
    index.add_with_ids(document_embeddings, np.array(range(len(document_embeddings))))
    return index.search(search_embeddings, k)

def interpret_results(search_terms, I, pdf_urls):
    for term_idx, term in enumerate(search_terms):
        print(f"Search term: {term}")
        for neighbor_idx in I[term_idx]:
            print(f"  - Document {neighbor_idx}: {pdf_urls[neighbor_idx]}")

def visualize_embeddings(embeddings, labels):
    # Apply t-SNE for dimensionality reduction
    tsne = TSNE(n_components=2, random_state=0)
    reduced_embeddings = tsne.fit_transform(embeddings)

    # Scatter plot
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=labels)
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.title('Visualization of Document Embeddings')
    plt.show()

# Assuming embeddings are loaded, and search_terms is a list of sentences or terms
def analyze_and_visualize(embeddings, search_terms, pdf_urls, labels):
    logging.debug('analyze and visualize function')
    document_embeddings = embeddings
    search_embeddings, _ = search_embeddings(search_terms, document_embeddings)
    D, I = search_embeddings(search_embeddings, document_embeddings)
    interpret_results(search_terms, I, pdf_urls)
    visualize_embeddings(document_embeddings, labels) # labels can be categories or other meta-data

def display_results(D, I, pdf_urls):
    logging.debug('display results function')
    print(f"Top {len(I[0])} documents related to the query:")
    for i, idx in enumerate(I[0]):
        likelihood_score = D[0][i]
        print(f"  - Document {idx}: {pdf_urls[idx]}, Likelihood Score: {likelihood_score}")


