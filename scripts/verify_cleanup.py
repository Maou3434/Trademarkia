import os
import tarfile
import re
from scripts.cleanup_dataset import clean_text

# Use the mini dataset for faster verification
DATASET_PATH = "Dataset/mini_newsgroups.tar.gz"

def verify_samples(num_samples=3):
    if not os.path.exists(DATASET_PATH):
        print(f"Error: {DATASET_PATH} not found.")
        return

    output_path = "verification_results.txt"
    print(f"--- Verifying Cleanup on {DATASET_PATH}. Saving to {output_path} ---")
    
    with open(output_path, "w", encoding="utf-8") as out_f:
        with tarfile.open(DATASET_PATH, "r:gz") as tar:
            count = 0
            for member in tar.getmembers():
                if member.isfile():
                    f = tar.extractfile(member)
                    if f:
                        content = f.read().decode('latin-1', errors='replace')
                        cleaned = clean_text(content)
                        
                        out_f.write(f"\nSample {count + 1}: {member.name}\n")
                        out_f.write("-" * 40 + "\n")
                        out_f.write("ORIGINAL (First 800 chars):\n")
                        out_f.write(content[:800] + "...\n")
                        out_f.write("-" * 40 + "\n")
                        out_f.write("CLEANED (First 800 chars):\n")
                        out_f.write(cleaned[:800] + "...\n")
                        out_f.write("-" * 40 + "\n")
                        
                        count += 1
                        if count >= num_samples:
                            break

if __name__ == "__main__":
    verify_samples()
