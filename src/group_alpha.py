# Group-specific alpha repair utilities

import pandas as pd

from src.reranking import evaluate_group_alpha_map_rerank
from src.smt_verification import choose_best_fair_alpha


def build_single_group_alpha_map(
    target_group,
    alpha
):
    """
    Build alpha map for one disadvantaged group.
    """
    return {
        target_group: alpha
    }


def build_multi_group_alpha_map(
    groups,
    alpha
):
    """
    Apply same alpha to multiple groups.
    """
    return {
        group: alpha
        for group in groups
    }


def evaluate_alpha_map(
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
    Evaluate one group-specific alpha map.
    """

    return evaluate_group_alpha_map_rerank(
        model=model,
        user_items=user_items,
        test_df=test_df,
        users_df=users_df,
        u2i=u2i,
        m2i=m2i,
        alpha_map=alpha_map,
        demographic=demographic,
        boost_items=boost_items,
        K=K,
        C=C
    )


def sweep_single_group_alpha(
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
    Sweep alpha for one target demographic group.
    """

    if alpha_values is None:
        alpha_values = [
            0.0,
            0.01,
            0.02,
            0.05,
            0.1,
            0.2,
            0.5,
            1.0
        ]

    rows = []
    full_results = {}

    for alpha in alpha_values:

        alpha_map = build_single_group_alpha_map(
            target_group,
            alpha
        )

        result = evaluate_alpha_map(
            model=model,
            user_items=user_items,
            test_df=test_df,
            users_df=users_df,
            u2i=u2i,
            m2i=m2i,
            alpha_map=alpha_map,
            demographic=demographic,
            boost_items=boost_items,
            K=K,
            C=C
        )

        rows.append({
            "alpha": alpha,
            "target_group": target_group,
            "demographic": demographic,
            "overall_recall": result["overall_recall"],
            "gap": result["gap"]
        })

        full_results[alpha] = result

    return pd.DataFrame(rows), full_results


def select_best_group_alpha(
    sweep_df,
    eps=0.01
):
    """
    Select highest-recall alpha satisfying fairness gap <= eps.
    """

    return choose_best_fair_alpha(
        sweep_df,
        gap_col="gap",
        recall_col="overall_recall",
        eps=eps
    )


def compare_group_alpha_to_baseline(
    baseline_result,
    repaired_result
):
    """
    Compare baseline and group-alpha repair.
    """

    return {
        "baseline_recall": baseline_result["overall_recall"],
        "repaired_recall": repaired_result["overall_recall"],
        "recall_delta": repaired_result["overall_recall"] - baseline_result["overall_recall"],
        "baseline_gap": baseline_result["gap"],
        "repaired_gap": repaired_result["gap"],
        "gap_delta": repaired_result["gap"] - baseline_result["gap"],
    }