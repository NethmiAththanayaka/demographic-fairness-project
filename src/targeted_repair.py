# SMT-style targeted sparse repair utilities

import itertools
import pandas as pd

from src.evaluation import (
    extract_item_scores,
    recall_at_k_with_recommender_function,
    max_gap
)

from src.fairness_metrics import demo_table, max_gap_with_pair
from src.smt_verification import repair_summary


# --------------------------------------------------
# Identify disadvantaged group
# --------------------------------------------------

def get_disadvantaged_group(
    group_table,
    group_col=None,
    metric_col="recall@10"
):
    if group_col is None:
        group_col = group_table.columns[0]

    clean = group_table.dropna(subset=[metric_col])

    if len(clean) == 0:
        return None

    idx = clean[metric_col].idxmin()

    return clean.loc[idx, group_col]


def get_advantaged_group(
    group_table,
    group_col=None,
    metric_col="recall@10"
):
    if group_col is None:
        group_col = group_table.columns[0]

    clean = group_table.dropna(subset=[metric_col])

    if len(clean) == 0:
        return None

    idx = clean[metric_col].idxmax()

    return clean.loc[idx, group_col]


# --------------------------------------------------
# User / group helpers
# --------------------------------------------------

def user_belongs_to_group(
    user_id,
    users_df,
    target_group,
    demographic="age_group"
):
    row = users_df[users_df["user_id"] == user_id]

    if len(row) == 0:
        return False

    return row.iloc[0][demographic] == target_group


# --------------------------------------------------
# Candidate extraction
# --------------------------------------------------

def get_candidates_with_scores(
    model,
    user_items,
    user_id,
    u2i,
    C=200
):
    if user_id not in u2i:
        return []

    uidx = u2i[user_id]

    recs = model.recommend(
        uidx,
        user_items[uidx],
        N=C
    )

    return extract_item_scores(recs)


def get_boundary_candidates(
    model,
    user_items,
    user_id,
    u2i,
    K=10,
    C=200,
    window=20
):
    """
    Return candidates around the top-K boundary.

    These are the easiest items to move into/out of top-K.
    """

    recs = get_candidates_with_scores(
        model,
        user_items,
        user_id,
        u2i,
        C=C
    )

    if len(recs) == 0:
        return []

    start = max(0, K - window)
    end = min(len(recs), K + window)

    return recs[start:end]


# --------------------------------------------------
# Targeted repair recommendation
# --------------------------------------------------

def recommend_targeted_repair(
    model,
    user_items,
    user_id,
    u2i,
    users_df,
    target_group,
    repair_items,
    demographic="age_group",
    K=10,
    C=200,
    alpha=0.1
):
    """
    Apply alpha bonus only to selected repair_items
    and only for users in the disadvantaged target group.
    """

    if user_id not in u2i:
        return []

    active = user_belongs_to_group(
        user_id=user_id,
        users_df=users_df,
        target_group=target_group,
        demographic=demographic
    )

    recs = get_candidates_with_scores(
        model=model,
        user_items=user_items,
        user_id=user_id,
        u2i=u2i,
        C=C
    )

    reranked = []

    for item_idx, score in recs:
        bonus = alpha if active and item_idx in repair_items else 0.0
        reranked.append((item_idx, float(score) + bonus))

    reranked.sort(key=lambda x: x[1], reverse=True)

    return reranked[:K]


# --------------------------------------------------
# Evaluate one repair set
# --------------------------------------------------

def evaluate_targeted_repair(
    model,
    user_items,
    test_df,
    users_df,
    u2i,
    m2i,
    target_group,
    repair_items,
    demographic="age_group",
    K=10,
    C=200,
    alpha=0.1
):
    def recommender_fn(uid):
        return recommend_targeted_repair(
            model=model,
            user_items=user_items,
            user_id=uid,
            u2i=u2i,
            users_df=users_df,
            target_group=target_group,
            repair_items=repair_items,
            demographic=demographic,
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
        "target_group": target_group,
        "alpha": alpha,
        "num_repair_items": len(repair_items),
        "repair_items": set(repair_items),
        "overall_recall": float(recall_df[metric_col].mean()) if len(recall_df) > 0 else 0.0,
        "gap": max_gap(table, metric_col=metric_col),
        "recall_df": recall_df,
        "group_table": table
    }


# --------------------------------------------------
# Build repair universe
# --------------------------------------------------

def select_repair_universe(
    model,
    user_items,
    test_df,
    users_df,
    u2i,
    target_group,
    demographic="age_group",
    K=10,
    C=200,
    window=20,
    max_users=100,
    max_items=50
):
    """
    Select a small set of candidate items for sparse repair.

    We collect boundary candidates from disadvantaged users.
    """

    target_users = users_df[
        users_df[demographic] == target_group
    ]["user_id"].tolist()

    test_users = set(test_df["user_id"].unique())

    target_users = [
        u for u in target_users
        if u in test_users and u in u2i
    ]

    target_users = target_users[:max_users]

    item_counts = {}

    for uid in target_users:
        candidates = get_boundary_candidates(
            model=model,
            user_items=user_items,
            user_id=uid,
            u2i=u2i,
            K=K,
            C=C,
            window=window
        )

        for item_idx, _ in candidates:
            item_counts[item_idx] = item_counts.get(item_idx, 0) + 1

    ranked = sorted(
        item_counts.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return [item for item, _ in ranked[:max_items]]


# --------------------------------------------------
# Sparse repair search
# --------------------------------------------------

def search_minimal_repair_set(
    model,
    user_items,
    test_df,
    users_df,
    u2i,
    m2i,
    target_group,
    repair_universe,
    demographic="age_group",
    K=10,
    C=200,
    alpha=0.1,
    eps=0.01,
    max_set_size=3
):
    """
    Brute-force sparse search over repair item subsets.

    Finds the smallest item subset that satisfies gap <= eps.
    """

    best_result = None
    tried = []

    for r in range(1, max_set_size + 1):

        for subset in itertools.combinations(repair_universe, r):

            result = evaluate_targeted_repair(
                model=model,
                user_items=user_items,
                test_df=test_df,
                users_df=users_df,
                u2i=u2i,
                m2i=m2i,
                target_group=target_group,
                repair_items=set(subset),
                demographic=demographic,
                K=K,
                C=C,
                alpha=alpha
            )

            row = {
                "set_size": r,
                "repair_items": set(subset),
                "overall_recall": result["overall_recall"],
                "gap": result["gap"]
            }

            tried.append(row)

            if result["gap"] <= eps:
                if best_result is None:
                    best_result = result
                elif result["overall_recall"] > best_result["overall_recall"]:
                    best_result = result

        if best_result is not None:
            break

    return best_result, pd.DataFrame(tried)


# --------------------------------------------------
# Full targeted repair pipeline
# --------------------------------------------------

def run_targeted_repair(
    model,
    user_items,
    test_df,
    users_df,
    u2i,
    m2i,
    baseline_group_table,
    demographic="age_group",
    K=10,
    C=200,
    alpha=0.1,
    eps=0.01,
    max_users=100,
    max_items=50,
    max_set_size=3
):
    metric_col = f"recall@{K}"

    target_group = get_disadvantaged_group(
        baseline_group_table,
        group_col=demographic,
        metric_col=metric_col
    )

    before_gap = max_gap(
        baseline_group_table,
        metric_col=metric_col
    )

    repair_universe = select_repair_universe(
        model=model,
        user_items=user_items,
        test_df=test_df,
        users_df=users_df,
        u2i=u2i,
        target_group=target_group,
        demographic=demographic,
        K=K,
        C=C,
        max_users=max_users,
        max_items=max_items
    )

    best_result, tried_df = search_minimal_repair_set(
        model=model,
        user_items=user_items,
        test_df=test_df,
        users_df=users_df,
        u2i=u2i,
        m2i=m2i,
        target_group=target_group,
        repair_universe=repair_universe,
        demographic=demographic,
        K=K,
        C=C,
        alpha=alpha,
        eps=eps,
        max_set_size=max_set_size
    )

    if best_result is None:
        return {
            "success": False,
            "target_group": target_group,
            "before_gap": before_gap,
            "after_gap": None,
            "repair_universe": repair_universe,
            "best_result": None,
            "tried": tried_df
        }

    after_gap = best_result["gap"]

    return {
        "success": True,
        "target_group": target_group,
        "before_gap": before_gap,
        "after_gap": after_gap,
        "repair_summary": repair_summary(before_gap, after_gap, eps),
        "repair_universe": repair_universe,
        "best_result": best_result,
        "tried": tried_df
    }


# --------------------------------------------------
# Comparison table
# --------------------------------------------------

def targeted_before_after_table(
    baseline_table,
    repaired_table,
    demographic=None,
    metric_col="recall@10"
):
    if demographic is None:
        demographic = baseline_table.columns[0]

    before = baseline_table[[demographic, metric_col]].rename(
        columns={metric_col: "before"}
    )

    after = repaired_table[[demographic, metric_col]].rename(
        columns={metric_col: "after"}
    )

    merged = before.merge(
        after,
        on=demographic,
        how="outer"
    )

    merged["delta"] = merged["after"] - merged["before"]

    return merged