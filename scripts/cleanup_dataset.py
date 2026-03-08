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
    If multiple double newlines exist, we assume the first one separates the core headers.
    """
    # Standard header/body separator
    parts = re.split(r'\n\s*\n', text, 1)
    if len(parts) > 1:
        # Check if the first part looks like headers (contains 'From:', 'Subject:', etc.)
        if re.search(r'^(From|Subject|Date|Newsgroups|Path):', parts[0], re.MULTILINE | re.IGNORECASE):
            return parts[1]
    
    # Fallback: remove lines that look like headers from the top
    lines = text.split('\n')
    i = 0
    while i < len(lines) and (re.match(r'^[A-Za-z\-]+:', lines[i]) or not lines[i].strip()):
        i += 1
    return '\n'.join(lines[i:])

def remove_quotes(text: str) -> str:
    """
    Remove lines that appear to be quotes from previous messages.
    """
    lines = text.split('\n')
    cleaned_lines = []
    
    # Pattern for "... writes:" or "... wrote:" handle names, emails, etc.
    quote_intro_pattern = re.compile(r'^.{1,100}\s+(writes|wrote|says|said):', re.IGNORECASE)
    # Pattern for lines that are just quote markers and space
    quote_marker_pattern = re.compile(r'^\s*[>|:]\s*')
    
    for line in lines:
        stripped = line.strip()
        # Skip lines starting with quote characters
        if quote_marker_pattern.match(line):
            continue
        # Skip quote intros
        if quote_intro_pattern.match(stripped):
            continue
        # Skip lines that are just "In article ..."
        if stripped.lower().startswith('in article'):
            continue
            
        cleaned_lines.append(line)
        
    return '\n'.join(cleaned_lines)

def remove_noise(text: str) -> str:
    """
    Remove horizontal rules, repeated punctuation, and other non-semantic artifacts.
    """
    # Remove horizontal rules (lines of -, =, _, *, etc.)
    text = re.sub(r'^[ \t]*[\-\=\_\*]{4,}[ \t]*$', '', text, flags=re.MULTILINE)
    
    # Remove email addresses and URLs (often noise for semantic search)
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    return text

def remove_footers(text: str) -> str:
    """
    Attempt to remove signatures/footers.
    """
    # Standard signature delimiter
    text = re.split(r'\n-- \n', text, 1)[0]
    text = re.split(r'\n--\n', text, 1)[0]
    
    # Heuristic: Remove typical "Thanks," or "Regards," at the end if followed by a name
    lines = text.split('\n')
    if len(lines) > 3:
        last_few = lines[-3:]
        for i, line in enumerate(last_few):
            if re.match(r'^(Thanks|Regards|Cheers|Sincerely),?$', line.strip(), re.IGNORECASE):
                return '\n'.join(lines[:- (3-i)])
                
    return text

def clean_text(text: str) -> str:
    """
    Master cleaning function.
    """
    # 1. Remove Headers (High noise, metadata)
    text = remove_headers(text)
    
    # 2. Remove Quotes (Redundant info for clustering)
    text = remove_quotes(text)
    
    # 3. Remove Noise (Horizontal rules, etc.)
    text = remove_noise(text)
    
    # 4. Remove Footers (Signatures)
    text = remove_footers(text)
    
    # 5. Normalization
    # Replace multiple newlines with single ones for character-based normalization later
    text = re.sub(r'\n+', ' ', text)
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
