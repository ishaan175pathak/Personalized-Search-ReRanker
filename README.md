# Personalized Search Re-Ranker

A search re-ranking system that combines a **fine-tuned neural re-ranker** with **user-behavior-based personalization** and a **live web crawler**. Given a query, the pipeline crawls relevant Wikipedia content, re-ranks candidate documents with a transformer model, boosts documents that similar users previously clicked, and returns the top 5 results.

> Full write-up: see the project report in this repo (*Intelligent Search Optimization Using Deep Learning-Based Behavioral Profiling*, and `Personalized_Reranker_Final_Report.docx`).

## How it works

```
User query
   │
   ▼
Web crawler ── Wikipedia seed URL from the query → follow links → extract text (BeautifulSoup)
   │
   ▼
Neural re-ranker ── fine-tuned msmarco-distilbert-base-v4 scores each [query, document] pair
   │
   ▼
Behavior-aware boosting ── compare the query embedding with users' past-query embeddings;
   │                        if a sufficiently similar user is found, boost the documents they clicked
   ▼
Top 5 results
```

## Components

| Module | What it does |
|---|---|
| **Data loading** | Loads queries, relevance judgments (qrels) and top-100 BM25 candidates from the TREC Deep Learning Track 2023; resolves query-id mismatches and pulls document text from the MS MARCO corpus |
| **Re-ranker** | `msmarco-distilbert-base-v4` fine-tuned on `[query, document]` pairs with Binary Cross-Entropy loss and the AdamW optimizer, trained on a 100,000-pair subset |
| **Web crawler** | Python crawler that builds a Wikipedia seed URL from the query, follows linked pages and extracts text with BeautifulSoup |
| **User behavior simulation** | Synthetic logs of past queries and clicked documents, used to prototype personalization |
| **Behavior-aware re-ranking** | Embedding similarity between the current query and historical user queries; clicked documents from similar users get a score boost |
| **Live pipeline** | Connects crawling, re-ranking and boosting into one real-time query flow |
| **Evaluation** | MRR@10 |

## Results

| System | MRR@10 |
|---|---|
| Fine-tuned re-ranker (baseline) | ~0.041 |
| + Behavior-aware boosting | ~0.041 (similar) |

The behavior-aware variant does not move MRR here because the click logs are **synthetic**, so there is no real user signal to learn from. The absolute MRR is also low, reflecting the small training subset (100k pairs) and limited compute. The value of this project is the end-to-end system design; the numbers are a starting baseline.

## Repository contents

```
Personalized-Search-ReRanker/
├── Source Files/                                   # implementation code
├── Intelligent Search Optimization Using Deep Learning-Based Behavioral Profiling Project Report.pdf
├── Personalized_Reranker_Final_Report.docx         # final report with architecture diagram
└── README.md
```

<!-- TODO(Ishaan): add install + run instructions and a requirements.txt after confirming the entry-point script in "Source Files/". -->

## Future work

- Replace synthetic logs with real user interaction data
- Serve the pipeline behind a web interface
- Improve personalization with topic modeling
- Train on the full dataset and evaluate on more queries

## Tech stack

Python · PyTorch · Sentence-Transformers (DistilBERT) · BeautifulSoup · TREC Deep Learning 2023 · MS MARCO

## Reference work

Beel et al. (2013), personalized re-ranking with content and behavior; Pant et al. (2004), behavior-driven web crawling; Zhao et al. (2019), embedding-based personalized re-ranking (arXiv:1904.06813).

## Author

**Ishaan Pathak** · [GitHub](https://github.com/ishaan175pathak)
