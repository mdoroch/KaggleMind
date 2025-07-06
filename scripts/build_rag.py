import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

if __name__ == "__main__":

    model = SentenceTransformer("all-MiniLM-L6-v2")

    with open("data/json_dataset/data_postprocessed_clean.json", "r", encoding="utf-8") as f:
        raw_data = json.load(f) 

    texts = [item["text"] for item in raw_data]

    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)  # или IndexIDMap(IndexFlatL2(d)) если нужны ID
    index.add(embeddings)

    faiss.write_index(index, "models/rag_index/feature_index.faiss")

    # with open("records.json", "w", encoding="utf-8") as f:
    #     json.dump([{"text": text} for text in texts], f, ensure_ascii=False, indent=2)
    
    