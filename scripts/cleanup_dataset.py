import os
import tarfile
import pandas as pd
import re
from typing import List, Tuple

# Configuration
DATASET_PATH = "Dataset/20_newsgroups.tar.gz"
OUTPUT_DIR = "processed_data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "cleaned_newsgroups.csv")

def remove_headers(text: str) -> str:
    """
    Remove Usenet headers from the email text.
    Headers are separated from the body by a double newline.
    """
    parts = text.split('\n\n', 1)
    if len(parts) > 1:
        return parts[1]
    return text

def remove_quotes(text: str) -> str:
    """
    Remove lines that appear to be quotes from previous messages.
    Commonly prefixed with '>', '|', or 'In article...'.
    """
    lines = text.split('\n')
    cleaned_lines = []
    
    # Pattern for "In article <id>, user writes:"
    quote_intro_pattern = re.compile(r'^(In article|On.*writes:|.*said:)', re.IGNORECASE)
    
    for line in lines:
        stripped = line.strip()
        # Skip lines starting with quote characters or common quote intros
        if stripped.startswith('>') or stripped.startswith('|'):
            continue
        if quote_intro_pattern.match(stripped):
            continue
        cleaned_lines.append(line)
        
    return '\n'.join(cleaned_lines)

def remove_footers(text: str) -> str:
    """
    Attempt to remove signatures/footers.
    Signatures often start with '--' or are at the very end of the message.
    """
    # Look for the standard signature delimiter '-- '
    parts = text.rsplit('\n-- \n', 1)
    if len(parts) > 1:
        return parts[0]
    
    # Also look for a sequence of short lines at the end (common in signatures)
    lines = text.split('\n')
    if len(lines) > 5:
        # Simple heuristic: if the last few lines are very short or contain 
        # contact info patterns, they might be a signature.
        # This is a bit aggressive, so we'll stick to basic rule-based for now.
        pass
    
    return text

def clean_text(text: str) -> str:
    """
    Master cleaning function.
    """
    # 1. Remove Headers (High noise, metadata)
    text = remove_headers(text)
    
    # 2. Remove Quotes (Redundant info for clustering)
    text = remove_quotes(text)
    
    # 3. Remove Footers (Signatures, contact info)
    text = remove_footers(text)
    
    # 4. Basic Normalization
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def extract_and_load(tar_path: str) -> List[Tuple[str, str]]:
    """
    Extract the tarball and return a list of (text, category) tuples.
    """
    data = []
    if not os.path.exists(tar_path):
        print(f"Error: Dataset not found at {tar_path}")
        return []

    print(f"Opening {tar_path}...")
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar.getmembers():
            if member.isfile():
                # Path format: 20_newsgroups/category/file_id
                path_parts = member.name.split('/')
                if len(path_parts) >= 3:
                    category = path_parts[1]
                    f = tar.extractfile(member)
                    if f:
                        try:
                            # Using latin-1 as it's common for these old newsgroup files
                            content = f.read().decode('latin-1')
                            data.append((content, category))
                        except Exception as e:
                            print(f"Failed to read {member.name}: {e}")
    return data

def main():
    # 1. Create output directory
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created directory: {OUTPUT_DIR}")

    # 2. Extract and Load
    raw_data = extract_and_load(DATASET_PATH)
    if not raw_data:
        return

    print(f"Loaded {len(raw_data)} documents. Starting cleanup...")

    # 3. Clean
    cleaned_records = []
    for i, (text, category) in enumerate(raw_data):
        cleaned_text_content = clean_text(text)
        # Only keep if there's meaningful text left
        if len(cleaned_text_content) > 50: 
            cleaned_records.append({
                'text': cleaned_text_content,
                'category': category
            })
        
        if (i + 1) % 1000 == 0:
            print(f"Processed {i + 1}/{len(raw_data)} documents...")

    # 4. Save to CSV
    df = pd.DataFrame(cleaned_records)
    df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')
    print(f"Cleanup complete! Saved {len(df)} documents to {OUTPUT_FILE}")

if __name__ == "__main__":
    # NOTE: Run this script to process the dataset.
    # Requirements: pandas
    # Path: c:\ML_Projects\Trademarkia\scripts\cleanup_dataset.py
    main()
