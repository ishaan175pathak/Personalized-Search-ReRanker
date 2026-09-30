# behavior_reranker.py
import torch
import numpy as np
import pickle
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer
from main import RankerModel, device, BATCH_SIZE, MODEL_NAME

# -------------------- Load Data ----------------------
with open("test_data.pkl", "rb") as f:
    test_q, test_d, test_l = pickle.load(f)

with open("synthetic_user_logs.pkl", "rb") as f:
    historical_user_logs = pickle.load(f)

# ------------------- Load Model ----------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = RankerModel().to(device)
model.load_state_dict(torch.load("reranker_model.pt"))
model.eval()

# -------- Build Historical User Embeddings ----------
def build_historical_embeddings(historical_logs, tokenizer, model):
    user_embeddings = {}
    for user_id, data in historical_logs.items():
        embeddings = []
        for query in data["queries"]:
            encoded = tokenizer(query, return_tensors="pt", truncation=True, padding=True).to(device)
            with torch.no_grad():
                output = model.encoder(**encoded)
                cls_embedding = output.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
                embeddings.append(cls_embedding)
        if embeddings:
            user_embeddings[user_id] = np.mean(embeddings, axis=0)
    return user_embeddings

# -------- Find Most Similar User to a Query ----------
def most_similar_user(current_query, historical_embeddings, tokenizer, model):
    encoded = tokenizer(current_query, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        current_embedding = model.encoder(**encoded).last_hidden_state[:, 0, :].cpu().numpy()

    best_user = None
    best_score = -1
    for user_id, hist_emb in historical_embeddings.items():
        sim = cosine_similarity(current_embedding, hist_emb.reshape(1, -1))[0][0]
        if sim > best_score:
            best_user = user_id
            best_score = sim
    return best_user, best_score

# ---------- Reranker + Behavior-aware Boost -----------
def behavior_aware_reranker(model, test_q, test_d, test_l, tokenizer, historical_logs, k=10, sim_threshold=0.6):
    user_embeddings = build_historical_embeddings(historical_logs, tokenizer, model)
    grouped = {}
    for q, d, l in zip(test_q, test_d, test_l):
        grouped.setdefault(q, []).append((d, l))

    mrr_total = 0

    for query, doc_label_list in tqdm(grouped.items(), desc="Behavior-aware Reranking"):
        docs = [d for d, _ in doc_label_list]
        labels = [l for _, l in doc_label_list]

        # Get model scores
        scores = []
        for i in range(0, len(docs), BATCH_SIZE):
            batch_docs = docs[i:i+BATCH_SIZE]
            encoded = tokenizer([query]*len(batch_docs), batch_docs,
                                truncation=True, padding=True, return_tensors="pt", max_length=256).to(device)
            with torch.no_grad():
                output = model(encoded['input_ids'], encoded['attention_mask'])
                scores.extend(output.view(-1).cpu().tolist())

        # Find similar user and boost clicked docs
        similar_user, sim_score = most_similar_user(query, user_embeddings, tokenizer, model)
        if sim_score > sim_threshold:
            clicked = set(historical_logs[similar_user]["clicked_docs"])
            boosted = [(d, s * 1.7 if d in clicked else s) for d, s in zip(docs, scores)]
        else:
            boosted = list(zip(docs, scores))

        ranked = sorted(boosted, key=lambda x: x[1], reverse=True)
        ranked_labels = [label for doc, label in sorted(zip(docs, labels), key=lambda x: scores[docs.index(x[0])], reverse=True)]

        for rank, label in enumerate(ranked_labels[:k], start=1):
            if label > 0:
                mrr_total += 1.0 / rank
                break

    return round(mrr_total / len(grouped), 4)

# -------------------- Main Loop --------------------------
if __name__ == "__main__":
    final_mrr = behavior_aware_reranker(model, test_q, test_d, test_l, tokenizer, historical_user_logs)
    print(f"\n🧠 Behavior-aware MRR@10: {final_mrr}")
