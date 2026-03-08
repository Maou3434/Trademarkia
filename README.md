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
- **Noise Filtering**: Uses regex and character density heuristics to remove horizontal rules, ASCII art, and decorative separators.
- **Signature Removal**: Detects and removes standard and common non-standard signatures/footers.
- **Normalization**: Joins lines and collapses excess whitespace for optimal embedding performance.

### 2. Fuzzy Clustering (Implementation Ongoing)
*Note: Clustering implementation details follow the prompt requirements for soft assignments and meaningful semantic boundaries.*

### 3. Semantic Cache (Implementation Ongoing)
*Note: Custom first-principles cache avoiding redundant computation on similar queries.*

### 4. FastAPI Service (Implementation Ongoing)
*Note: Service endpoint exposing search and cache stats.*

## Setup and Installation

1. **Virtual Environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   ```

2. **Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Data Preparation**:
   Run the cleanup script to generate the processed dataset:
   ```bash
   python scripts/cleanup_dataset.py
   ```
   The cleaned data will be saved to `processed_data/cleaned_newsgroups.csv`.

## Submission Details
- **Email Access**: recruitments@trademarkia.com
- **Submission Link**: [Google Form](https://forms.gle/4RpHZpAi8rbG9QCE8)

---
*Developed as part of the Trademarkia AI&ML Engineer Task.*
