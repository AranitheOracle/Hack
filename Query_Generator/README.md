# Image-Query Alignment using SOTA Models

This project uses state-of-the-art (SOTA) models to align user queries with information from uploaded images, refining search results for more accurate content discovery. With **BLIP2Model** for image processing, **spaCy** for NLP-based query refinement, and Yahoo Search for web searches, we create a dynamic search experience.

## Overview

1. **Image Content Extraction**:
   - The **BLIP2Model** is used to analyze the uploaded image and extract its primary information.
   
2. **Query Refinement**:
   - **spaCy** identifies nouns and adjectives in the user's query, refining terms to match the image context.

3. **Web Search**:
   - Yahoo Search fetches results that align with both the image and the refined query.

## Tech Stack

- **Python**: Core language.
- **Hugging Face Transformers**: Provides the BLIP2Model.
- **spaCy**: Powers NLP-based query refinement.
- **Streamlit**: Interactive web interface.

## Setup Instructions

### 1. Clone the Repository

   - You have already cloned the repo
   ```bash
      cd Query_Generator
   ```

### 2. Install the requirements

```bash
pip install -r requirements.txt
```

### 3. Run the file

```bash
streamlit run main.py
```

