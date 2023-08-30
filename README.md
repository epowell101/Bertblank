# Bertblank - using Bert for embeddings & more plus some Bart
## This software:
- takes a URL
- builds a list of URLs of PDFs on that URL
- reads those PDFs
- summarizes them using a Bart summarization model
- embeds them using Bert
- indexes them using FAISS
- searches the embeddings in two ways:
    - preembedded terms like heterogeneous
    - an added search phrase
- outputs:
    - written summaries of the docs
    - search results as lists & dfs 
    - so far simple visualization

## What is the point?
- yes to try out the technology
- to help builders who do not have time to read the papers emerging in a field by:
    - enabling them to find those papers that are worthy of additional scrutiny

## Future directions?
- improved visualization
- improved user experience for primary workflow
- hardening and resilience
- maybe use Streamlit or other for mouseover GUI of visualization & search terms
- look again at Bertopic - we have built a lot from the ground up that they have -> what else is missing?
- look at a knowledge graph approach
- do some cool predictions like topics and collaborators (run a betting pool on a future AI conference? - need a revenue model after all ;))

## Caveats?
- not yet well tested or documented
- the embeddings themselves are not uploaded here - and they will be large when you make them
