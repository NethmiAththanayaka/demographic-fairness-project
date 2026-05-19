# Fairness mitigation utilities

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix

from src.model import train_als
from src.evaluation import recall_at_k_for_users_model, max_gap
from src.fairness_metrics import demo_table


# --------------------------------------------------
# Weighted Training Matrix
# --------------------------------------------------

def build_weighted_matrix_for_group(
    train_df,
    users_df,
    target_group,
    demographic="age_group",
    boost=2.0,
    user_col="user_id",
    item_col="movie_id",
    rating_col="rating"
):
    """
    Build sparse training matrix where interactions from one
    demographic group receive higher weight.
    """

    train = train_df.merge(
        users_df[[user_col, demographic]],
        on=user_col,
        how="left"
    )

    user_ids = train[user_col].unique()
    item_ids = train[item_col].unique()

    u2i = {u: i for i, u in enumerate(user_ids)}
    i2u = {i: u for u, i in u2i.items()}

    m2i = {m: i for i, m in enumerate(item_ids)}
    i2m = {i: m for m, i in m2i.items()}

    weights = np.where(
        train[demographic] == target_group,
        boost,
        1.0
    )

    values = train[rating_col].values * weights

    rows = train[user_col].map(u2i)
    cols = train[item_col].map(m2i)

    matrix = coo_matrix(
        (values, (rows, cols)),
        shape=(len(user_ids), len(item_ids))
    ).tocsr()

    return matrix, u2i, i2u, m2i, i2m


# --------------------------------------------------
# Train Weighted ALS
# --------------------------------------------------

def train_weighted_als_for_group(
    train_df,
    users_df,
    target_group,
    demographic="age_group",
    boost=2.0,
    factors=64,
    regularization=0.01,
    iterations=20
):
    """
    Train ALS with boosted interactions for a target group.
    """

    user_items, u2i, i2u, m2i, i2m = build_weighted_matrix_for_group(
        train_df=train_df,
        users_df=users_df,
        target_group=target_group,
        demographic=demographic,
        boost=boost
    )

    model = train_als(
        user_items,
        factors=factors,
        regularization=regularization,
        iterations=iterations
    )

    return model, user_items, u2i, i2u, m2i, i2m


# --------------------------------------------------
# Evaluate One Boost
# --------------------------------------------------

def evaluate_group_boost(
    train_df,
    test_df,
    users_df,
    target_group,
    demographic="age_group",
    boost=2.0,
    K=10,
    factors=64,
    regularization=0.01,
    iterations=20
):
    """
    Train/evaluate one demographic weighting setting.
    """

    model, user_items, u2i, i2u, m2i, i2m = train_weighted_als_for_group(
        train_df=train_df,
        users_df=users_df,
        target_group=target_group,
        demographic=demographic,
        boost=boost,
        factors=factors,
        regularization=regularization,
        iterations=iterations
    )

    recall_df = recall_at_k_for_users_model(
        model=model,
        user_items=user_items,
        test_df=test_df,
        users=users_df,
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

    gap = max_gap(
        table,
        metric_col=metric_col
    )

    overall_recall = float(
        recall_df[metric_col].mean()
    ) if len(recall_df) > 0 else 0.0

    return {
        "boost": boost,
        "target_group": target_group,
        "demographic": demographic,
        "overall_recall": overall_recall,
        "gap": gap,
        "recall_df": recall_df,
        "group_table": table,
        "model": model,
        "user_items": user_items,
        "u2i": u2i,
        "i2u": i2u,
        "m2i": m2i,
        "i2m": i2m
    }


# --------------------------------------------------
# Sweep Boost Values
# --------------------------------------------------

def sweep_group_boosts(
    train_df,
    test_df,
    users_df,
    target_group,
    demographic="age_group",
    boost_values=None,
    K=10,
    factors=64,
    regularization=0.01,
    iterations=20
):
    """
    Try multiple boost values and collect fairness/utility results.
    """

    if boost_values is None:
        boost_values = [1.0, 1.25, 1.5, 2.0, 3.0, 5.0]

    rows = []
    full_results = {}

    for boost in boost_values:
        result = evaluate_group_boost(
            train_df=train_df,
            test_df=test_df,
            users_df=users_df,
            target_group=target_group,
            demographic=demographic,
            boost=boost,
            K=K,
            factors=factors,
            regularization=regularization,
            iterations=iterations
        )

        rows.append({
            "boost": boost,
            "target_group": target_group,
            "demographic": demographic,
            "overall_recall": result["overall_recall"],
            "gap": result["gap"]
        })

        full_results[boost] = result

    return pd.DataFrame(rows), full_results


# --------------------------------------------------
# Choose Best Boost
# --------------------------------------------------

def choose_best_boost(
    sweep_df,
    eps=None,
    gap_col="gap",
    utility_col="overall_recall"
):
    """
    Choose best boost.

    If eps is provided:
        choose highest utility among settings with gap <= eps.

    Otherwise:
        choose smallest gap.
    """

    if len(sweep_df) == 0:
        return None

    df = sweep_df.copy()

    if eps is not None:
        df = df[df[gap_col] <= eps]

        if len(df) == 0:
            return None

        best_idx = df[utility_col].idxmax()
        return df.loc[best_idx].to_dict()

    best_idx = df[gap_col].idxmin()
    return df.loc[best_idx].to_dict()


# --------------------------------------------------
# Before/After Comparison
# --------------------------------------------------

def compare_baseline_and_mitigation(
    baseline_table,
    mitigated_table,
    demographic=None,
    metric_col="recall@10"
):
    """
    Compare demographic recall before and after mitigation.
    """

    if demographic is None:
        demographic = baseline_table.columns[0]

    before = baseline_table[[demographic, metric_col]].rename(
        columns={metric_col: "before"}
    )

    after = mitigated_table[[demographic, metric_col]].rename(
        columns={metric_col: "after"}
    )

    merged = before.merge(
        after,
        on=demographic,
        how="outer"
    )

    merged["delta"] = merged["after"] - merged["before"]

    return merged