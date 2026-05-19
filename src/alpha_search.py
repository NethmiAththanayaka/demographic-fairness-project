# Alpha-search utilities for fairness/utility tradeoff

import pandas as pd

from src.reranking import (
    evaluate_longtail_rerank,
    evaluate_group_rerank,
    evaluate_group_alpha_map_rerank,
)

from src.smt_verification import (
    choose_first_fair_alpha,
    choose_best_fair_alpha,
    smt_choose_alpha,
)


def sweep_alpha_longtail(
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
    if alpha_values is None:
        alpha_values = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]

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
            alpha=alpha,
        )

        rows.append({
            "alpha": alpha,
            "demographic": demographic,
            "overall_recall": result["overall_recall"],
            "gap": result["gap"],
        })

        full_results[alpha] = result

    return pd.DataFrame(rows), full_results


def sweep_alpha_group(
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
    if alpha_values is None:
        alpha_values = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]

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
            alpha=alpha,
        )

        rows.append({
            "alpha": alpha,
            "target_group": target_group,
            "demographic": demographic,
            "overall_recall": result["overall_recall"],
            "gap": result["gap"],
        })

        full_results[alpha] = result

    return pd.DataFrame(rows), full_results


def select_alpha(
    sweep_df,
    eps=0.01,
    strategy="best_fair",
    gap_col="gap",
    recall_col="overall_recall"
):
    """
    Select alpha from a sweep table.

    strategy:
        "first_fair" -> smallest alpha with gap <= eps
        "best_fair"  -> fair alpha with highest recall
        "smt"        -> same feasibility idea, using Z3 helper
        "min_gap"    -> alpha with smallest gap
        "max_recall" -> alpha with highest recall
    """

    if len(sweep_df) == 0:
        return None

    if strategy == "first_fair":
        return choose_first_fair_alpha(
            sweep_df,
            gap_col=gap_col,
            recall_col=recall_col,
            eps=eps
        )

    if strategy == "best_fair":
        return choose_best_fair_alpha(
            sweep_df,
            gap_col=gap_col,
            recall_col=recall_col,
            eps=eps
        )

    if strategy == "smt":
        return smt_choose_alpha(
            sweep_df,
            eps=eps,
            gap_col=gap_col,
            recall_col=recall_col
        )

    if strategy == "min_gap":
        idx = sweep_df[gap_col].idxmin()
        return sweep_df.loc[idx].to_dict()

    if strategy == "max_recall":
        idx = sweep_df[recall_col].idxmax()
        return sweep_df.loc[idx].to_dict()

    raise ValueError(f"Unknown alpha selection strategy: {strategy}")


def summarize_alpha_sweep(
    sweep_df,
    eps_values=None,
    gap_col="gap",
    recall_col="overall_recall"
):
    if eps_values is None:
        eps_values = [0.001, 0.005, 0.01, 0.02, 0.05]

    rows = []

    for eps in eps_values:
        first_fair = select_alpha(
            sweep_df,
            eps=eps,
            strategy="first_fair",
            gap_col=gap_col,
            recall_col=recall_col
        )

        best_fair = select_alpha(
            sweep_df,
            eps=eps,
            strategy="best_fair",
            gap_col=gap_col,
            recall_col=recall_col
        )

        rows.append({
            "eps": eps,
            "first_fair_alpha": None if first_fair is None else first_fair["alpha"],
            "first_fair_recall": None if first_fair is None else first_fair[recall_col],
            "first_fair_gap": None if first_fair is None else first_fair[gap_col],
            "best_fair_alpha": None if best_fair is None else best_fair["alpha"],
            "best_fair_recall": None if best_fair is None else best_fair[recall_col],
            "best_fair_gap": None if best_fair is None else best_fair[gap_col],
        })

    return pd.DataFrame(rows)


def compare_alpha_to_baseline(
    baseline_result,
    selected_result,
    metric_col="overall_recall",
    gap_col="gap"
):
    if selected_result is None:
        return {
            "selected": False,
            "baseline_recall": baseline_result.get(metric_col, None),
            "selected_recall": None,
            "recall_delta": None,
            "baseline_gap": baseline_result.get(gap_col, None),
            "selected_gap": None,
            "gap_delta": None,
        }

    return {
        "selected": True,
        "baseline_recall": baseline_result.get(metric_col, None),
        "selected_recall": selected_result.get(metric_col, None),
        "recall_delta": selected_result.get(metric_col, 0.0) - baseline_result.get(metric_col, 0.0),
        "baseline_gap": baseline_result.get(gap_col, None),
        "selected_gap": selected_result.get(gap_col, None),
        "gap_delta": selected_result.get(gap_col, 0.0) - baseline_result.get(gap_col, 0.0),
    }


def build_alpha_map(
    target_group,
    alpha,
    default_alpha=0.0
):
    """
    Convenience helper for group-specific alpha repair.
    """

    return {
        target_group: alpha
    } if alpha != default_alpha else {}


def evaluate_selected_group_alpha(
    model,
    user_items,
    test_df,
    users_df,
    u2i,
    m2i,
    target_group,
    selected_alpha,
    demographic="age_group",
    boost_items=None,
    K=10,
    C=200
):
    alpha_map = build_alpha_map(
        target_group=target_group,
        alpha=selected_alpha
    )

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