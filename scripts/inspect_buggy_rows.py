import pandas as pd
import os

CSV_PATH = "processed_data/cleaned_newsgroups_v2.csv"

def inspect_rows(indices):
    if not os.path.exists(CSV_PATH):
        print(f"Error: {CSV_PATH} not found.")
        return

    df = pd.read_csv(CSV_PATH)
    
    with open("buggy_rows_detailed.txt", "w", encoding="utf-8") as f:
        f.write(f"Total rows in CSV: {len(df)}\n")
        
        for idx in indices:
            f.write(f"\n{'='*50}\n")
            f.write(f"ROW INDEX: {idx}\n")
            f.write(f"{'='*50}\n")
            if idx < len(df):
                text = df.iloc[idx]['text']
                f.write(f"Category: {df.iloc[idx]['category']}\n")
                f.write(f"Length: {len(text)}\n")
                f.write("-" * 20 + "\n")
                f.write(text + "\n")
            else:
                f.write("Index out of range\n")

if __name__ == "__main__":
    # User mentioned rows 213, 1166, 1388-1392, 1661, 1662.
    # Assuming user might be 1-indexed (spreadsheet style) or 0-indexed.
    # I'll check a range around each.
    target_indices = [212, 213, 1165, 1166] + list(range(1387, 1393)) + [1660, 1661, 1662]
    inspect_rows(target_indices)
