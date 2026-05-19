# Evaluates recommendation quality

import numpy as np
import pandas as pd


# --------------------------------------------------
# Basic Error Metrics
# --------------------------------------------------

def rmse(predictions, targets):
    predictions = np.array(predictions)
    targets = np.array(targets)

    return np.sqrt(
        np.mean((predictions - targets) ** 2)
    )


def mae(predictions, targets):
    predictions = np.array(predictions)
    targets = np.array(targets)

    return np.mean(
        np.abs(predictions - targets)
    )


# --------------------------------------------------
# Fairness Gap Helpers
# --------------------------------------------------

def max_gap(table, metric_col="recall@10"):
    vals = table[metric_col].dropna().values

    if len(vals) == 0:
        return 0.0

    return float(np.max(vals) - np.min(vals))


def max_gap_with_pair(
    table,
    group_col,
    metric_col="recall@10"
):
    """
    Return the max gap and the worst/best groups.
    """

    clean = table.dropna(subset=[metric_col]).copy()

    if len(clean) == 0:
        return {
            "gap": 0.0,
            "worst_group": None,
            "best_group": None,
            "worst_value": None,
            "best_value": None,
        }

    worst_idx = clean[metric_col].idxmin()
    best_idx = clean[metric_col].idxmax()

    worst_row = clean.loc[worst_idx]
    best_row = clean.loc[best_idx]

    return {
        "gap": float(best_row[metric_col] - worst_row[metric_col]),
        "worst_group": worst_row[group_col],
        "best_group": best_row[group_col],
        "worst_value": float(worst_row[metric_col]),
        "best_value": float(best_row[metric_col]),
    }


# --------------------------------------------------
# Recommendation Output Normalization
# --------------------------------------------------

def extract_item_indices(recs):
    """
    Normalize outputs from implicit.recommend().

    Different versions/settings may return:
    1. list of tuples: [(item, score), ...]
    2. tuple of arrays: (item_ids, scores)

    This function returns only item ids.
    """

    if isinstance(recs, tuple) and len(recs) == 2:
        item_ids, _ = recs
        return list(item_ids)

    return [r[0] for r in recs]


def extract_item_scores(recs):
    """
    Normalize outputs from implicit.recommend().

    Returns:
        [(item_id, score), ...]
    """

    if isinstance(recs, tuple) and len(recs) == 2:
        item_ids, scores = recs
        return list(zip(item_ids, scores))

    return list(recs)


# --------------------------------------------------
# Recall@K
# --------------------------------------------------

def recall_at_k(recommended_items, relevant_items, K=10):
    """
    Compute Recall@K for one user.
    """

    recommended_items = set(list(recommended_items)[:K])
    relevant_items = set(relevant_items)

    if len(relevant_items) == 0:
        return None

    return len(recommended_items & relevant_items) / len(relevant_items)


def precision_at_k(recommended_items, relevant_items, K=10):
    """
    Compute Precision@K for one user.
    """

    recommended_items = set(list(recommended_items)[:K])
    relevant_items = set(relevant_items)

    if K == 0:
        return None

    return len(recommended_items & relevant_items) / K


# --------------------------------------------------
# ALS Model Evaluation
# --------------------------------------------------

def recall_at_k_for_users_model(
    model,
    user_items,
    test_df,
    users,
    u2i,
    m2i,
    K=10,
    relevance_threshold=4
):
    """
    Compute per-user Recall@K for an implicit ALS model.

    Relevant items are test-set movies with rating >= relevance_threshold.
    """

    recalls = []

    for uid in test_df["user_id"].unique():

        if uid not in u2i:
            continue

        user_test = test_df[test_df["user_id"] == uid]

        relevant = set(
            m2i[m]
            for m in user_test[
                user_test["rating"] >= relevance_threshold
            ]["movie_id"]
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

        recommended = extract_item_indices(recs)

        recall = recall_at_k(
            recommended,
            relevant,
            K=K
        )

        recalls.append({
            "user_id": uid,
            f"recall@{K}": recall,
            "num_relevant": len(relevant)
        })

    return pd.DataFrame(recalls)


def ranking_metrics_for_users_model(
    model,
    user_items,
    test_df,
    users,
    u2i,
    m2i,
    K=10,
    relevance_threshold=4
):
    """
    Compute Recall@K and Precision@K per user.
    """

    rows = []

    for uid in test_df["user_id"].unique():

        if uid not in u2i:
            continue

        user_test = test_df[test_df["user_id"] == uid]

        relevant = set(
            m2i[m]
            for m in user_test[
                user_test["rating"] >= relevance_threshold
            ]["movie_id"]
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

        recommended = extract_item_indices(recs)

        rows.append({
            "user_id": uid,
            f"recall@{K}": recall_at_k(
                recommended,
                relevant,
                K=K
            ),
            f"precision@{K}": precision_at_k(
                recommended,
                relevant,
                K=K
            ),
            "num_relevant": len(relevant)
        })

    return pd.DataFrame(rows)


# --------------------------------------------------
# Reranked Recommendation Evaluation
# --------------------------------------------------

def recall_at_k_with_recommender_function(
    recommender_fn,
    test_df,
    u2i,
    m2i,
    K=10,
    relevance_threshold=4
):
    """
    Evaluate any custom recommender function.

    recommender_fn(uid) should return either:
        [item_id, item_id, ...]
    or:
        [(item_id, score), ...]
    """

    rows = []

    for uid in test_df["user_id"].unique():

        if uid not in u2i:
            continue

        user_test = test_df[test_df["user_id"] == uid]

        relevant = set(
            m2i[m]
            for m in user_test[
                user_test["rating"] >= relevance_threshold
            ]["movie_id"]
            if m in m2i
        )

        if len(relevant) == 0:
            continue

        recs = recommender_fn(uid)

        if len(recs) == 0:
            recommended = []
        elif isinstance(recs[0], tuple):
            recommended = [x[0] for x in recs]
        else:
            recommended = recs

        rows.append({
            "user_id": uid,
            f"recall@{K}": recall_at_k(
                recommended,
                relevant,
                K=K
            ),
            "num_relevant": len(relevant)
        })

    return pd.DataFrame(rows)


# --------------------------------------------------
# Summary Helpers
# --------------------------------------------------

def summarize_metric(df, metric_col="recall@10"):
    """
    Summarize a metric over users.
    """

    if len(df) == 0:
        return {
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
            "count": 0
        }

    return {
        "mean": float(df[metric_col].mean()),
        "median": float(df[metric_col].median()),
        "std": float(df[metric_col].std()),
        "count": int(len(df))
    }


def compare_before_after(
    before_table,
    after_table,
    group_col,
    metric_col="recall@10"
):
    """
    Compare demographic metric tables before and after repair.
    """

    before = before_table[[group_col, metric_col]].rename(
        columns={metric_col: "before"}
    )

    after = after_table[[group_col, metric_col]].rename(
        columns={metric_col: "after"}
    )

    merged = before.merge(
        after,
        on=group_col,
        how="outer"
    )

    merged["delta"] = merged["after"] - merged["before"]

    return merged