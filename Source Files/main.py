import os
import torch
import pandas as pd
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from torch.optim import AdamW
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import DataLoader, Dataset, random_split
from torch import nn
import gzip
import gc
import pickle
from sklearn.model_selection import train_test_split

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cwd = Path.cwd() # Current working directory

# ------------------------------ CONFIG ----------------------------------
MODEL_NAME = "sentence-transformers/msmarco-distilbert-base-v4" # Pre-trained tokenizer
MAX_LEN = 256
BATCH_SIZE = 8
LEARNING_RATE = 3e-5
EPOCHS = 2
DOCS_DIR = f"{cwd}/final project/datasets/msmarco_v2_doc"  # Folder containing 16 gz files
TOP100_PATH = f"{cwd}/final project/datasets/docv2_train_top100.txt.gz" # using only 100,000 datapoints
QRELS_PATH = f"{cwd}/final project/datasets/docv2_train_qrels.tsv"
QUERIES_PATH = f"{cwd}/final project/datasets/docv2_train_queries.tsv"

# ------------------------------ DATASET ---------------------------------
class RankingDataset(Dataset):
    def __init__(self, queries, docs, labels, tokenizer):
        self.queries = queries
        self.docs = docs
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        encoded = self.tokenizer(
            self.queries[idx], self.docs[idx],
            max_length=MAX_LEN, padding="max_length",
            truncation=True, return_tensors="pt"
        )
        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'label': torch.tensor(self.labels[idx], dtype=torch.float),
            'query': self.queries[idx]
        }

# --------------------------- MODEL WRAPPER ------------------------------
class RankerModel(nn.Module):
    def __init__(self):
        super(RankerModel, self).__init__()
        self.encoder = AutoModel.from_pretrained(MODEL_NAME)
        self.linear = nn.Linear(self.encoder.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        scores = self.linear(cls_output)
        return scores.squeeze(-1)

# ------------------------ LOAD AND PREPARE DATA -------------------------
def extract_required_docids(top100_path):
    required_docids = set()
    with gzip.open(top100_path, 'rt', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                docid = parts[2]
                required_docids.add(docid)
    print(f"Total unique docids extracted: {len(required_docids)}")
    return required_docids

def build_qid_map(top100_df, qrels_df):
    doc_to_true_qid = dict(zip(qrels_df.docid, qrels_df.qid))
    qid_mapping = {}
    for fake_qid in top100_df.qid.unique():
        docs = top100_df[top100_df.qid == fake_qid]['docid'].unique()
        for doc in docs:
            if doc in doc_to_true_qid:
                qid_mapping[str(fake_qid)] = str(doc_to_true_qid[doc])
                break
    return qid_mapping

def load_data():
    print('loading data')
    queries_df = pd.read_csv(QUERIES_PATH, sep="\t", names=["qid", "query"])
    qrels_df = pd.read_csv(QRELS_PATH, sep="\t", names=["qid", "Q0", "docid", "label"])
    top100_df = pd.read_csv(TOP100_PATH, sep=" ", names=["qid", "Q0", "docid", "rank", "score", "run"])

    # using only the first 1,00,000 datapoints
    queries_df = queries_df.iloc[:100_000, :]
    qrels_df = qrels_df.iloc[:100_000, :]
    top100_df = top100_df.iloc[:100_000, :]    


    qid_mapping = build_qid_map(top100_df, qrels_df)
 
    queries = dict(zip(queries_df.qid.astype(str), queries_df["query"]))
    qrels = dict(zip(zip(qrels_df.qid.astype(str), qrels_df.docid.astype(str)), qrels_df.label))

    pairs = []
    for _, row in top100_df.iterrows():
        raw_qid, docid = str(row.qid), str(row.docid)
        true_qid = qid_mapping.get(raw_qid)
        if true_qid and true_qid in queries:
            label = qrels.get((true_qid, docid), 0)
            pairs.append((true_qid, queries[true_qid], docid, label))

    required_docids = extract_required_docids(TOP100_PATH)
    docid_to_text = load_docs(required_docids)

    query_texts = []
    doc_texts = []
    labels = []

    for qid, query, docid, label in pairs:
        doc_text = docid_to_text.get(docid, "").strip()
        if doc_text:  # Keep only if document text is non-empty
            query_texts.append(query)
            doc_texts.append(doc_text)
            labels.append(label)

    return query_texts, doc_texts, labels

# ------------------------ DOCUMENT TEXT LOADING -------------------------
def load_docs(required_docids):
    docid_to_text = {}
    files = [f for f in os.listdir(DOCS_DIR) if f.endswith(".gz")]
    for file in tqdm(files, desc="Reading document files"):
        with gzip.open(os.path.join(DOCS_DIR, file), 'rt', encoding='utf-8') as f:
            for line in f:
                doc = json.loads(line)
                docid = doc.get("docid")
                if docid in required_docids:
                    text = doc.get("body", "")
                    docid_to_text[docid] = text
        if len(docid_to_text) >= len(required_docids):
            break
    return docid_to_text

# -------------------------- TRAINING LOOP -------------------------------
def train_model(model, dataloader, optimizer):
    print('Training the model')
    model.train()
    loss_fn = nn.BCEWithLogitsLoss()
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            scores = model(input_ids, attention_mask)
            loss = loss_fn(scores, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f"Epoch {epoch+1} Loss: {total_loss:.4f}")
        torch.cuda.empty_cache(); gc.collect()

# ------------------ BEHAVIOR-AWARE RERANKING MODULE --------------------
def user_behavior_similarity(current_query, historical_embeddings, tokenizer, model):
    current_tokenized = tokenizer(current_query, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        current_embedding = model.encoder(**current_tokenized).last_hidden_state[:, 0, :].cpu().numpy()

    similarities = []
    for past_emb in historical_embeddings:
        score = cosine_similarity(current_embedding, past_emb.reshape(1, -1))[0][0]
        similarities.append(score)

    return np.mean(similarities) if similarities else 0.0

# -------------------------- MAIN EXECUTION ------------------------------
if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = RankerModel().to(device)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    queries, docs, labels = load_data()

    # Split data into train and test sets
    train_q, test_q, train_d, test_d, train_l, test_l = train_test_split(queries, docs, labels, test_size=0.2, random_state=42)

    train_dataset = RankingDataset(train_q, train_d, train_l, tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    train_model(model, train_dataloader, optimizer)
    torch.save(model.state_dict(), "reranker_model.pt")

    with open("test_data.pkl", "wb") as f:
        pickle.dump((test_q, test_d, test_l), f)

    with open("train_data.pkl", "wb") as f:
        pickle.dump((train_q, train_d, train_l), f)

    print("\nModel training complete and saved.")