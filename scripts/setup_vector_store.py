import faiss
import numpy as np
import os

# Configuration
EMBEDDINGS_FILE = "processed_data/embeddings.npy"
INDEX_FILE = "processed_data/newsgroups_faiss.index"

def main():
    # 1. Load Embeddings
    if not os.path.exists(EMBEDDINGS_FILE):
        print(f"Error: {EMBEDDINGS_FILE} not found. Run generate_embeddings.py first.")
        return

    print(f"Loading embeddings from {EMBEDDINGS_FILE}...")
    embeddings = np.load(EMBEDDINGS_FILE).astype('float32')
    dimension = embeddings.shape[1]
    print(f"Loaded {embeddings.shape[0]} vectors of dimension {dimension}.")

    # 2. Initialize FAISS Index
    # We use IndexFlatIP (Inner Product) because embeddings are normalized (Cosine Similarity)
    cpu_index = faiss.IndexFlatIP(dimension)

    # 3. Move to GPU
    try:
        res = faiss.StandardGpuResources()
        # Using a simple flat index on GPU for max precision and simplicity for this scale (~20k vectors)
        index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        print("Using FAISS-GPU.")
    except Exception as e:
        print(f"Warning: GPU failed, falling back to CPU. Error: {e}")
        index = cpu_index

    # 4. Add vectors
    print("Adding vectors to index...")
    index.add(embeddings)
    print(f"Total vectors in index: {index.ntotal}")

    # 5. Save (Must move back to CPU to save)
    print(f"Saving index to {INDEX_FILE}...")
    if hasattr(index, "index_gpu_to_cpu"):
         faiss.write_index(faiss.index_gpu_to_cpu(index), INDEX_FILE)
    else:
         faiss.write_index(index, INDEX_FILE)
    
    print("Done!")

if __name__ == "__main__":
    main()
