import os
import json
import faiss
import numpy as np
import argparse
from sentence_transformers import SentenceTransformer

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Scrape Kaggle discussion links from competition leaderboard pages.")

    parser.add_argument(
        "--output_file", 
        type=str, 
        default="models/rag_index/feature_index.faiss",
        help="Name of the output faiss index file"
    )
    
    args = parser.parse_args()

    model = SentenceTransformer("all-MiniLM-L6-v2")

    with open("data/json_dataset/data_postprocessed_clean.json", "r", encoding="utf-8") as f:
        raw_data = json.load(f) 

    texts = [item["text"] for item in raw_data]

    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)  # или IndexIDMap(IndexFlatL2(d)) если нужны ID
    index.add(embeddings)

    faiss.write_index(index, args.output_file)
    
    