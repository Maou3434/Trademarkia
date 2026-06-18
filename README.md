# Trademarkia AI & ML Engineer Task: Semantic Search System

This project implements a lightweight semantic search system using the **20 Newsgroups dataset**. It features data cleaning, fuzzy clustering, a custom semantic cache, and a FastAPI service.

## Project Structure

```text
.
├── Dataset/                  # Contains 20_newsgroups.tar.gz and mini_newsgroups.tar.gz
├── processed_data/           # Cleaned and normalized dataset (CSV)
├── scripts/
│   ├── cleanup_dataset.py    # Main script for dataset cleaning and normalization
│   ├── verify_cleanup.py     # Script to verify cleaning on sample data
│   └── inspect_buggy_rows.py # Utility to inspect specific rows for noise
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

## System Components

### 1. Dataset Cleaning & Normalization
The 20 Newsgroups dataset is inherently noisy. The `cleanup_dataset.py` script performs:
- **Header Removal**: Strips Usenet headers and metadata.
- **Quote Stripping**: Intelligently removes email-style quotes, including those with initials (e.g., `AB> `).
- **Noise Filtering**: Uses regex and character density heuristics.
- **Signature Removal**: Detects and removes standard and common non-standard signatures.

### 2. Embedding Generation
Implemented in `scripts/generate_embeddings.py`:
- **Model**: `BAAI/bge-large-en-v1.5` (1024 dimensions).
- **Optimizations**: GPU-accelerated inference with batch size 128 and smart truncation for long documents.
- **Output**: `processed_data/embeddings.npy`.

### 3. Vector Store Setup
Implemented in `scripts/setup_vector_store.py`:
- **Engine**: `FAISS-GPU` (Facebook AI Similarity Search).
- **Index Type**: `IndexFlatIP` (Maximum Inner Product) for normalized cosine similarity.
- **Output**: `processed_data/newsgroups_faiss.index`.

## Setup and Installation

1. **Virtual Environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   ```

2. **Dependencies**:
   ```bash
   pip install -r requirements.txt
   # For GPU acceleration (WSL/Linux):
   pip install sentence-transformers faiss-gpu tqdm
   ```

3. **Execution Pipeline**:
   ```bash
   # Step 1: Clean the dataset
   python scripts/cleanup_dataset.py

   # Step 2: Generate embeddings (Requires GPU)
   python scripts/generate_embeddings.py

   # Step 3: Initialize vector store
   python scripts/setup_vector_store.py
   ```

## Submission Details
- **Email Access**: recruitments@trademarkia.com
- **Submission Link**: [Google Form](https://forms.gle/4RpHZpAi8rbG9QCE8)

---
*Developed as part of the Trademarkia AI&ML Engineer Task.*
