import os
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def download_models():
    # Model 1: Embedder
    print("Downloading sentence-transformers/all-mpnet-base-v2...")
    SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    
    # Model 2: Ranker (Cross-Encoder)
    print("Downloading cross-encoder/ms-marco-MiniLM-L-6-v2...")
    model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    AutoTokenizer.from_pretrained(model_name)
    AutoModelForSequenceClassification.from_pretrained(model_name)
    
    print("Models downloaded successfully.")

if __name__ == "__main__":
    download_models()
