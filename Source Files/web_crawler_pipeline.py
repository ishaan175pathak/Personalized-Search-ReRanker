import torch
import pickle
from transformers import AutoTokenizer
from web_crawler import crawl
from main import RankerModel, device, MODEL_NAME, BATCH_SIZE
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ---------------- Load Behavior Logs (optional) ------------------
try:
    with open("synthetic_user_logs.pkl", "rb") as f:
        historical_logs = pickle.load(f)
except:
    historical_logs = {}

# ---------------- Load Model & Tokenizer -------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = RankerModel().to(device)
model.load_state_dict(torch.load("reranker_model.pt", map_location=device))
model.eval()

# ---------------- Behavior Similarity Module ---------------------
def build_user_embeddings(historical_logs, tokenizer, model):
    user_embeddings = {}
    for uid, log in historical_logs.items():
        embeddings = []
        for q in log["queries"]:
            encoded = tokenizer(q, return_tensors="pt", truncation=True, padding=True).to(device)
            with torch.no_grad():
                emb = model.encoder(**encoded).last_hidden_state[:, 0, :].squeeze().cpu().numpy()
                embeddings.append(emb)
        if embeddings:
            user_embeddings[uid] = np.mean(embeddings, axis=0)
    return user_embeddings

def most_similar_user(query, user_embeds, tokenizer, model):
    encoded = tokenizer(query, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        q_emb = model.encoder(**encoded).last_hidden_state[:, 0, :].cpu().numpy()
    best_user, best_score = None, -1
    for uid, emb in user_embeds.items():
        sim = cosine_similarity(q_emb, emb.reshape(1, -1))[0][0]
        if sim > best_score:
            best_user, best_score = uid, sim
    return best_user, best_score

# ---------------- Reranking Logic -------------------------------
def rerank_documents(query, crawled_docs, model, tokenizer, user_logs, use_behavior=True):
    doc_texts = [doc['content'] for doc in crawled_docs]
    doc_urls = [doc['url'] for doc in crawled_docs]

    scores = []
    for i in range(0, len(doc_texts), BATCH_SIZE):
        batch = doc_texts[i:i+BATCH_SIZE]
        encoded = tokenizer([query]*len(batch), batch, return_tensors='pt', padding=True,
                            truncation=True, max_length=256).to(device)
        with torch.no_grad():
            out = model(encoded['input_ids'], encoded['attention_mask'])
            scores.extend(out.view(-1).cpu().tolist())

    # Boost if similar user found
    if use_behavior and user_logs:
        user_embeddings = build_user_embeddings(user_logs, tokenizer, model)
        best_user, score = most_similar_user(query, user_embeddings, tokenizer, model)
        if score > 0.7:
            clicked = set(user_logs[best_user]['clicked_docs'])
            boosted = [(u, s * 1.5 if u in clicked else s) for u, s in zip(doc_urls, scores)]
        else:
            boosted = list(zip(doc_urls, scores))
    else:
        boosted = list(zip(doc_urls, scores))

    ranked = sorted(boosted, key=lambda x: x[1], reverse=True)
    return ranked[:10]

# ---------------- Main Loop -------------------------------------
if __name__ == "__main__":
    user_query = input("🔍 Enter your search query: ")
    seed_url = f"https://en.wikipedia.org/wiki/{user_query.replace(' ', '_')}"

    print("\n🌐 Crawling web pages...")
    crawled_results = crawl(seed_url, max_pages=5)
    if not crawled_results:
        print("❌ No results found during crawl.")
        exit()

    print("\n🤖 Reranking crawled content...")
    top_results = rerank_documents(user_query, crawled_results, model, tokenizer, historical_logs)

    print("\n📋 Top Ranked Results:")
    for i, (url, score) in enumerate(top_results, start=1):
        print(f"{i}. {url} (Score: {score:.4f})")
