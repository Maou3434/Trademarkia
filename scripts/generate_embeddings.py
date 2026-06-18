#!/usr/bin/env python3
import os
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm

# Configuration
INPUT_FILE = "processed_data/cleaned_newsgroups.csv"
OUTPUT_DIR = "processed_data"
EMBEDDINGS_FILE = os.path.join(OUTPUT_DIR, "embeddings.npy")
MODEL_NAME = "BAAI/bge-large-en-v1.5"

def main():
    # 1. Load Data
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found. Please run cleanup_dataset.py first.")
        return

    print(f"Loading data from {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    
    # Pre-truncate to speed up tokenizer (BGE-Large 512 tokens ~ 2000-3000 chars)
    # This prevents very long documents from stalling the pre-processing phase.
    texts = df['text'].astype(str).str[:2500].tolist() 
    print(f"Loaded {len(texts)} documents.")

    # 2. Initialize Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    print(f"Loading model {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME, device=device)

    # 3. Generate Embeddings
    print("Generating embeddings (this may take a few minutes on GPU)...")
    # Batch size 128 is optimized for modern GPUs (8GB+ VRAM)
    embeddings = model.encode(
        texts, 
        batch_size=128, 
        show_progress_bar=True, 
        convert_to_numpy=True,
        normalize_embeddings=True # Recommended for BGE models for Cosine Similarity
    )

    # 4. Save
    print(f"Saving embeddings to {EMBEDDINGS_FILE}...")
    np.save(EMBEDDINGS_FILE, embeddings)
    print("Done!")

if __name__ == "__main__":
    main()
