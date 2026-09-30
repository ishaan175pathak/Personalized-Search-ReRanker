import random
import pickle

def generate_synthetic_user_logs(queries, docs, num_users=10, history_per_user=3):
    """
    Generate synthetic user behavior logs.
    
    Parameters:
        queries: list of query strings
        docs: list of document strings
        num_users: number of users to simulate
        history_per_user: number of past interactions per user
    
    Returns:
        Dictionary of user logs in the format:
        {
            "user_1": {
                "queries": [...],
                "clicked_docs": [...]
            },
            ...
        }
    """
    user_logs = {}
    
    # Zip queries and docs into pairs
    query_doc_pairs = list(zip(queries, docs))

    for i in range(num_users):
        user_id = f"user_{i+1}"
        user_logs[user_id] = {
            "queries": [],
            "clicked_docs": []
        }

        sampled_pairs = random.sample(query_doc_pairs, min(history_per_user, len(query_doc_pairs)))
        
        for q, d in sampled_pairs:
            user_logs[user_id]["queries"].append(q)
            user_logs[user_id]["clicked_docs"].append(d)

    return user_logs

# importing test data for generating synthetic users

# ---------------- Load test split ----------------
with open("test_data.pkl", "rb") as f:
    test_q, test_d, test_l = pickle.load(f)


# creating synthetic users
synthetic_user_logs = generate_synthetic_user_logs(test_q, test_d, num_users=15, history_per_user=4)

# Save to disk
with open("synthetic_user_logs.pkl", "wb") as f:
    pickle.dump(synthetic_user_logs, f)

print("✅ Synthetic user logs saved to 'synthetic_user_logs.pkl'")
