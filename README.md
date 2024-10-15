
### Setup Instructions

1. **Clone the repo**
    ```bash
    git clone https://github.com/AranitheOracle/Hack.git
    ```


1. **Navigate to the Folder**:
   ```bash
   cd hack

### Directory Descriptions:- 
-------------
#### 1. Query_Generator

The `Query_Generator` directory contains files necessary for refining user queries based on image analysis and natural language processing.

- **main.py**: This is the main script for running the query generation module. It utilizes models to extract information from images and refines the user query accordingly.
- **README.md**: Documentation specific to the Query Generator module.
- **requirements.txt**: Lists the dependencies needed to run the query generation module.

#### 2. SOTA_RAG

The `SOTA_RAG` directory is dedicated to the Retrieval-Augmented Generation (RAG) process. It includes files for storing the corpus, model training data, and vectorized document embeddings.

- **corpus.json**: Contains the main corpus of documents used in the RAG pipeline.
- **flow.pdf**: Visual representation of the RAG process flow.
- **rag_sota.ipynb**: A Jupyter notebook for experiments and testing within the SOTA RAG module.
- **README.md**: Documentation for setting up and using the SOTA RAG module.
- **requirements.txt**: Lists dependencies required to run the SOTA RAG module.
- **train.json**: JSON file containing training data to fine-tune or validate the model.
- **vector_store.zip**: Compressed file containing pre-computed vector embeddings for document retrieval.

---

Each directory has its own `README.md` and `requirements.txt` file for module-specific setup instructions. The contents are structured to support streamlined development and easy navigation.

