
Each folder has a dedicated `main.py` to run each module and a `requirements.txt` to install dependencies.

---

## 1. SOTA RAG

The **SOTA RAG** module implements a retrieval-augmented generation pipeline that retrieves relevant documents and generates responses based on query content.

### Features

- **Document Retrieval**: Uses Chroma/FAISS vector databases to search for relevant information.
- **Query Matching**: Combines query semantics with document embeddings.
- **Response Generation**: Provides a response based on retrieved content.

### Setup Instructions

1. **Navigate to the Folder**:
   ```bash
   cd sota_rag
