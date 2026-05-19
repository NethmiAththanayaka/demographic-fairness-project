# Fairness-aware reranking utilities

import numpy as np
import pandas as pd

from src.evaluation import (
    extract_item_scores,
    recall_at_k_with_recommender_function,
    max_gap
)

from src.fairness_metrics import demo_table


# --------------------------------------------------
# Basic Long-Tail Reranking
# --------------------------------------------------

def recommend_with_fair_rerank(
    model,
    user_items,
    user_id,
    u2i,
    longtail_items,
    K=10,
    C=200,
    alpha=0.1
):
    """
    Rerank recommendations by adding alpha bonus to long-tail items.
    """

    if user_id not in u2i:
        return []

    uidx = u2i[user_id]

    recs = model.recommend(
        uidx,
        user_items[uidx],
        N=C
    )

    recs = extract_item_scores(recs)

    reranked = []

    for item_idx, score in recs:

        bonus = alpha if item_idx in longtail_items else 0.0

        reranked.append(
            (item_idx, float(score) + bonus)
        )

    reranked.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return reranked[:K]


# --------------------------------------------------
# User-Group-Aware Reranking
# --------------------------------------------------

def recommend_with_group_rerank(
    model,
    user_items,
    user_id,
    u2i,
    users_df,
    target_group,
    demographic="age_group",
    boost_items=None,
    K=10,
    C=200,
    alpha=0.1
):
    """
    Apply reranking only to users from a target demographic group.
    """

    if boost_items is None:
        boost_items = set()

    if user_id not in u2i:
        return []

    user_row = users_df[
        users_df["user_id"] == user_id
    ]

    if len(user_row) == 0:
        active_alpha = 0.0
    else:
        group_value = user_row.iloc[0][demographic]
        active_alpha = alpha if group_value == target_group else 0.0

    uidx = u2i[user_id]

    recs = model.recommend(
        uidx,
        user_items[uidx],
        N=C
    )

    recs = extract_item_scores(recs)

    reranked = []

    for item_idx, score in recs:

        bonus = active_alpha if item_idx in boost_items else 0.0

        reranked.append(
            (item_idx, float(score) + bonus)
        )

    reranked.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return reranked[:K]


# --------------------------------------------------
# Group-Specific Alpha Map
# --------------------------------------------------

def recommend_with_group_alpha_map(
    model,
    user_items,
    user_id,
    u2i,
    users_df,
    alpha_map,
    demographic="age_group",
    boost_items=None,
    K=10,
    C=200
):
    """
    Use a different alpha per demographic group.

    Example:
        alpha_map = {
            "Under18": 0.2,
            "56+": 0.1
        }
    """

    if boost_items is None:
        boost_items = set()

    if user_id not in u2i:
        return []

    user_row = users_df[
        users_df["user_id"] == user_id
    ]

    if len(user_row) == 0:
        active_alpha = 0.0
    else:
        group_value = user_row.iloc[0][demographic]
        active_alpha = alpha_map.get(group_value, 0.0)

    uidx = u2i[user_id]

    recs = model.recommend(
        uidx,
        user_items[uidx],
        N=C
    )

    recs = extract_item_scores(recs)

    reranked = []

    for item_idx, score in recs:

        bonus = active_alpha if item_idx in boost_items else 0.0

        reranked.append(
            (item_idx, float(score) + bonus)
        )

    reranked.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return reranked[:K]


# --------------------------------------------------
# Popularity / Long-Tail Item Helpers
# --------------------------------------------------

def get_item_popularity(
    train_df,
    item_col="movie_id"
):
    """
    Count how many times each item appears in training data.
    """

    return (
        train_df[item_col]
        .value_counts()
        .rename("popularity")
        .reset_index()
        .rename(columns={"index": item_col})
    )


def get_longtail_items(
    train_df,
    m2i,
    item_col="movie_id",
    quantile=0.25
):
    """
    Define long-tail items as items below a popularity quantile.
    Returns matrix item indices.
    """

    counts = train_df[item_col].value_counts()

    threshold = counts.quantile(quantile)

    longtail_movie_ids = counts[
        counts <= threshold
    ].index

    longtail_indices = {
        m2i[m]
        for m in longtail_movie_ids
        if m in m2i
    }

    return longtail_indices


def get_popular_items_for_group(
    train_df,
    users_df,
    target_group,
    demographic="age_group",
    item_col="movie_id",
    user_col="user_id",
    m2i=None,
    top_n=200
):
    """
    Find items most interacted with by a demographic group.
    Useful for boosting items preferred by a disadvantaged group.
    """

    merged = train_df.merge(
        users_df[[user_col, demographic]],
        on=user_col,
        how="left"
    )

    group_items = merged[
        merged[demographic] == target_group
    ][item_col].value_counts()

    top_items = group_items.head(top_n).index

    if m2i is None:
        return set(top_items)

    return {
        m2i[m]
        for m in top_items
        if m in m2i
    }


# --------------------------------------------------
# Evaluate Reranking
# --------------------------------------------------

def evaluate_longtail_rerank(
    model,
    user_items,
    test_df,
    users_df,
    u2i,
    m2i,
    longtail_items,
    demographic="age_group",
    K=10,
    C=200,
    alpha=0.1
):
    """
    Evaluate long-tail reranking for all users.
    """

    def recommender_fn(uid):
        return recommend_with_fair_rerank(
            model=model,
            user_items=user_items,
            user_id=uid,
            u2i=u2i,
            longtail_items=longtail_items,
            K=K,
            C=C,
            alpha=alpha
        )

    recall_df = recall_at_k_with_recommender_function(
        recommender_fn=recommender_fn,
        test_df=test_df,
        u2i=u2i,
        m2i=m2i,
        K=K
    )

    metric_col = f"recall@{K}"

    table = demo_table(
        recall_df,
        users_df,
        demographic=demographic,
        metric_col=metric_col
    )

    return {
        "alpha": alpha,
        "overall_recall": float(recall_df[metric_col].mean()) if len(recall_df) > 0 else 0.0,
        "gap": max_gap(table, metric_col=metric_col),
        "recall_df": recall_df,
        "group_table": table
    }


def evaluate_group_rerank(
    model,
    user_items,
    test_df,
    users_df,
    u2i,
    m2i,
    target_group,
    demographic="age_group",
    boost_items=None,
    K=10,
    C=200,
    alpha=0.1
):
    """
    Evaluate reranking applied only to a target demographic group.
    """

    if boost_items is None:
        boost_items = set()

    def recommender_fn(uid):
        return recommend_with_group_rerank(
            model=model,
            user_items=user_items,
            user_id=uid,
            u2i=u2i,
            users_df=users_df,
            target_group=target_group,
            demographic=demographic,
            boost_items=boost_items,
            K=K,
            C=C,
            alpha=alpha
        )

    recall_df = recall_at_k_with_recommender_function(
        recommender_fn=recommender_fn,
        test_df=test_df,
        u2i=u2i,
        m2i=m2i,
        K=K
    )

    metric_col = f"recall@{K}"

    table = demo_table(
        recall_df,
        users_df,
        demographic=demographic,
        metric_col=metric_col
    )

    return {
        "alpha": alpha,
        "target_group": target_group,
        "overall_recall": float(recall_df[metric_col].mean()) if len(recall_df) > 0 else 0.0,
        "gap": max_gap(table, metric_col=metric_col),
        "recall_df": recall_df,
        "group_table": table
    }


def evaluate_group_alpha_map_rerank(
    model,
    user_items,
    test_df,
    users_df,
    u2i,
    m2i,
    alpha_map,
    demographic="age_group",
    boost_items=None,
    K=10,
    C=200
):
    """
    Evaluate reranking with a different alpha for each group.
    """

    if boost_items is None:
        boost_items = set()

    def recommender_fn(uid):
        return recommend_with_group_alpha_map(
            model=model,
            user_items=user_items,
            user_id=uid,
            u2i=u2i,
            users_df=users_df,
            alpha_map=alpha_map,
            demographic=demographic,
            boost_items=boost_items,
            K=K,
            C=C
        )

    recall_df = recall_at_k_with_recommender_function(
        recommender_fn=recommender_fn,
        test_df=test_df,
        u2i=u2i,
        m2i=m2i,
        K=K
    )

    metric_col = f"recall@{K}"

    table = demo_table(
        recall_df,
        users_df,
        demographic=demographic,
        metric_col=metric_col
    )

    return {
        "alpha_map": alpha_map,
        "overall_recall": float(recall_df[metric_col].mean()) if len(recall_df) > 0 else 0.0,
        "gap": max_gap(table, metric_col=metric_col),
        "recall_df": recall_df,
        "group_table": table
    }


# --------------------------------------------------
# Sweep Alpha Values
# --------------------------------------------------

def sweep_longtail_alpha(
    model,
    user_items,
    test_df,
    users_df,
    u2i,
    m2i,
    longtail_items,
    demographic="age_group",
    alpha_values=None,
    K=10,
    C=200
):
    """
    Sweep alpha values for long-tail reranking.
    """

    if alpha_values is None:
        alpha_values = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]

    rows = []
    full_results = {}

    for alpha in alpha_values:

        result = evaluate_longtail_rerank(
            model=model,
            user_items=user_items,
            test_df=test_df,
            users_df=users_df,
            u2i=u2i,
            m2i=m2i,
            longtail_items=longtail_items,
            demographic=demographic,
            K=K,
            C=C,
            alpha=alpha
        )

        rows.append({
            "alpha": alpha,
            "overall_recall": result["overall_recall"],
            "gap": result["gap"]
        })

        full_results[alpha] = result

    return pd.DataFrame(rows), full_results


def sweep_group_alpha(
    model,
    user_items,
    test_df,
    users_df,
    u2i,
    m2i,
    target_group,
    demographic="age_group",
    boost_items=None,
    alpha_values=None,
    K=10,
    C=200
):
    """
    Sweep alpha values for group-specific reranking.
    """

    if alpha_values is None:
        alpha_values = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]

    rows = []
    full_results = {}

    for alpha in alpha_values:

        result = evaluate_group_rerank(
            model=model,
            user_items=user_items,
            test_df=test_df,
            users_df=users_df,
            u2i=u2i,
            m2i=m2i,
            target_group=target_group,
            demographic=demographic,
            boost_items=boost_items,
            K=K,
            C=C,
            alpha=alpha
        )

        rows.append({
            "alpha": alpha,
            "target_group": target_group,
            "overall_recall": result["overall_recall"],
            "gap": result["gap"]
        })

        full_results[alpha] = result

    return pd.DataFrame(rows), full_results