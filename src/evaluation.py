import numpy as np
import pandas as pd

def rmse(predictions, targets):
    return np.sqrt(np.mean((predictions - targets) ** 2))

def max_gap(table, metric_col="recall@10"):
    vals = table[metric_col].values

    if len(vals) == 0:
        return 0.0

    return float(np.max(vals) - np.min(vals))

def recall_at_k_for_users_model(
    model,
    user_items,
    test_df,
    users,
    u2i,
    m2i,
    K=10
):

    recalls = []

    for uid in test_df["user_id"].unique():

        if uid not in u2i:
            continue

        user_test = test_df[test_df["user_id"] == uid]

        relevant = set(
            m2i[m]
            for m in user_test[user_test["rating"] >= 4]["movie_id"]
            if m in m2i
        )

        if len(relevant) == 0:
            continue

        uidx = u2i[uid]

        recs = model.recommend(
            uidx,
            user_items[uidx],
            N=K
        )

        recommended = set([r[0] for r in recs])

        recall = len(recommended & relevant) / len(relevant)

        recalls.append((uid, recall))

    return pd.DataFrame(recalls, columns=["user_id", "recall@10"])
