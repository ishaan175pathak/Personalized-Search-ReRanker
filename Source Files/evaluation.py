import torch
import pickle
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from main import RankerModel, device, BATCH_SIZE, MODEL_NAME  # importing model class and reuse constants

# ---------------- Load test split ----------------
with open("test_data.pkl", "rb") as f:
    test_q, test_d, test_l = pickle.load(f)

# ---------------- Load model and tokenizer ----------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = RankerModel().to(device)
model.load_state_dict(torch.load("reranker_model.pt"))
model.eval()

# ---------------- Rerank and Evaluate ----------------
def rerank_batch(query, doc_texts):
    scores = []
    for i in range(0, len(doc_texts), BATCH_SIZE):
        batch_docs = doc_texts[i:i+BATCH_SIZE]
        encoded = tokenizer([query]*len(batch_docs), batch_docs, return_tensors='pt',
                            truncation=True, padding=True, max_length=256).to(device)
        with torch.no_grad():
            output = model(encoded['input_ids'], encoded['attention_mask'])
            scores.extend(output.cpu().tolist())
    return scores

def mrr_at_10(test_q, test_d, test_l):
    grouped = {}
    for q, d, l in zip(test_q, test_d, test_l):
        grouped.setdefault(q, []).append((d, l))

    mrr_total = 0
    for query, doc_label_list in tqdm(grouped.items(), desc="Evaluating"):
        doc_texts = [doc for doc, _ in doc_label_list]
        labels = [label for _, label in doc_label_list]

        scores = rerank_batch(query, doc_texts)
        ranked = sorted(zip(scores, labels), reverse=True)

        for rank, (_, label) in enumerate(ranked[:10], start=1):
            if label > 0:
                mrr_total += 1.0 / rank
                break

    return mrr_total / len(grouped)

# ---------------- Run Evaluation ----------------

if __name__ == "__main__":
    score = mrr_at_10(test_q, test_d, test_l)
    print(f"\n Evaluation MRR@10: {score:.4f}")