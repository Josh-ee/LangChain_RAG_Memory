# LangChain_RAG_Memory
Example of Conversational Memory with RAG that cites sources


## Introduction

This repository is an example of how to create and use Retrieval Augmented Generation (RAG) with LangChain. 
This is done using open-source models and does not require any API or paid service
Here are the libraries used:

   1. Vector Storage -> ChromaDB
   2. Embedding Model -> [BAAI/bge-small-en-v1.5'model](https://huggingface.co/spaces/mteb/leaderboard) from HuggingFaceBgeEmbeddings
   3. LLM -> [Mistral-11B-OmniMix](https://huggingface.co/TheBloke/Mistral-11B-OmniMix-GGUF) the 4bit quantized GGUF version from TheBloke
   4. User Interface (UI) -> [Chainlit](https://docs.chainlit.io/integrations/llama-index)



# Vector Database and RAG with LangChain

The `create_vectorDB.ipynb` notebook guides you through the process of creating a vector database using Chroma DB, which stores embeddings from Hugging Face's language models. This vector database is then used by the demo script for RAG.

The `demo_RAG.ipynb` notebook demonstrates how to utilize the created vector database to answer questions based on the documents it contains.


## Part 1: Creating the Vector Database with ChromaDB and Hugging Face Embeddings

Use the `create_vectorDB.ipynb` to create the `LC_VectorDB`
   1. Download an example PDF from arXiv
   2. Convert the PDF to LangChain Documents
   3. Prepare the documents by splitting the data
   4. Create and store the Vector DB


## Part 2: Utilizing the Vector Database with an Open Source LLM Model

Run the `demo_RAG.ipynb` which will step you through 4 different examples:
   1. Load the Foundational LLM and ask a question
   2. Use the LLM with RAG from LC_VectorDB
   3. Conversational Memory without RAG
   4. Conversational Memory with RAG and Sources

## Performance

I developed this code on my M2 Max with 32GB of RAM. However, you can scale the embedding model and/or the LLM model to better match with your system.
All of the necessary imports for Mac to utilize MPS are present in the notebooks.
