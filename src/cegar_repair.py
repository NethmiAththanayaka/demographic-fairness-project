import pandas as pd

from src.evaluation import recall_at_k_for_users_model, max_gap
from src.fairness_metrics import demo_table, max_gap_with_pair
from src.smt_verification import verify_table
from src.reranking import (
    get_popular_items_for_group,
    evaluate_group_alpha_map_rerank,
)


def get_counterexample(group_table, demographic, metric_col="recall@10"):
    pair = max_gap_with_pair(
        group_table,
        group_col=demographic,
        metric_col=metric_col,
    )

    return {
        "worst_group": pair["worst_group"],
        "best_group": pair["best_group"],
        "gap": pair["gap"],
        "worst_value": pair["worst_value"],
        "best_value": pair["best_value"],
    }


def evaluate_baseline_fairness(
    model,
    user_items,
    test_df,
    users_df,
    u2i,
    m2i,
    demographic="age_group",
    K=10,
):
    recall_df = recall_at_k_for_users_model(
        model=model,
        user_items=user_items,
        test_df=test_df,
        users=users_df,
        u2i=u2i,
        m2i=m2i,
        K=K,
    )

    metric_col = f"recall@{K}"

    group_table = demo_table(
        recall_df,
        users_df,
        demographic=demographic,
        metric_col=metric_col,
    )

    return {
        "recall_df": recall_df,
        "group_table": group_table,
        "gap": max_gap(group_table, metric_col=metric_col),
        "overall_recall": float(recall_df[metric_col].mean())
        if len(recall_df) > 0
        else 0.0,
    }


def get_group_value(group_table, group_name, demographic, metric_col):
    row = group_table[group_table[demographic] == group_name]

    if len(row) == 0:
        return None

    return float(row.iloc[0][metric_col])


def get_underperforming_groups(
    group_table,
    demographic,
    metric_col="recall@10",
    mode="below_mean",
    max_groups=3,
):
    """
    Select multiple groups to repair.

    mode:
        below_mean  -> all groups below mean, limited by max_groups
        bottom_k    -> bottom max_groups groups
    """

    table = group_table.copy()
    table = table.dropna(subset=[metric_col])

    if len(table) == 0:
        return []

    if mode == "below_mean":
        threshold = table[metric_col].mean()

        candidates = table[
            table[metric_col] < threshold
        ].copy()

    elif mode == "bottom_k":
        candidates = table.copy()

    else:
        raise ValueError(f"Unknown selection mode: {mode}")

    candidates = candidates.sort_values(metric_col, ascending=True)

    return candidates[demographic].head(max_groups).tolist()


def evaluate_alpha_map_repair(
    model,
    user_items,
    test_df,
    users_df,
    u2i,
    m2i,
    alpha_map,
    boost_item_map,
    demographic="age_group",
    K=10,
    C=200,
):
    boost_items = set()

    for items in boost_item_map.values():
        boost_items.update(items)

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
        C=C,
    )


def is_safe_multigroup_repair(
    before_result,
    after_result,
    target_groups,
    best_group,
    demographic,
    metric_col,
    utility_tolerance=0.001,
    best_group_tolerance=0.0005,
):
    before_table = before_result["group_table"]
    after_table = after_result["group_table"]

    before_gap = before_result["gap"]
    after_gap = after_result["gap"]

    gap_improves = after_gap < before_gap

    utility_safe = (
        after_result["overall_recall"]
        >= before_result["overall_recall"] - utility_tolerance
    )

    before_best = get_group_value(
        before_table,
        best_group,
        demographic,
        metric_col,
    )

    after_best = get_group_value(
        after_table,
        best_group,
        demographic,
        metric_col,
    )

    if before_best is None or after_best is None:
        return False

    best_not_harmed = after_best >= before_best - best_group_tolerance

    target_improvements = []

    for group in target_groups:
        before_val = get_group_value(
            before_table,
            group,
            demographic,
            metric_col,
        )

        after_val = get_group_value(
            after_table,
            group,
            demographic,
            metric_col,
        )

        if before_val is None or after_val is None:
            return False

        target_improvements.append(after_val >= before_val)

    targets_improve = all(target_improvements)

    return (
        gap_improves
        and utility_safe
        and best_not_harmed
        and targets_improve
    )


def search_safe_multigroup_repair_candidate(
    model,
    user_items,
    train_df,
    test_df,
    users_df,
    u2i,
    m2i,
    current_result,
    alpha_map,
    boost_item_map,
    target_groups,
    best_group,
    demographic="age_group",
    K=10,
    C=200,
    alpha_candidates=None,
    top_n_candidates=None,
    utility_tolerance=0.001,
    best_group_tolerance=0.0005,
    target_eps=0.02,
):
    if alpha_candidates is None:
        alpha_candidates = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]

    if top_n_candidates is None:
        top_n_candidates = [50, 100, 200, 500]

    metric_col = f"recall@{K}"

    best_candidate = None
    best_score = None
    tried_rows = []

    for top_n in top_n_candidates:
        candidate_boost_item_map = dict(boost_item_map)

        for group in target_groups:
            candidate_boost_item_map[group] = get_popular_items_for_group(
                train_df=train_df,
                users_df=users_df,
                target_group=group,
                demographic=demographic,
                item_col="movie_id",
                user_col="user_id",
                m2i=m2i,
                top_n=top_n,
            )

        for alpha_delta in alpha_candidates:
            trial_alpha_map = dict(alpha_map)

            for group in target_groups:
                old_alpha = trial_alpha_map.get(group, 0.0)
                trial_alpha_map[group] = min(old_alpha + alpha_delta, 1.0)

            trial_result = evaluate_alpha_map_repair(
                model=model,
                user_items=user_items,
                test_df=test_df,
                users_df=users_df,
                u2i=u2i,
                m2i=m2i,
                alpha_map=trial_alpha_map,
                boost_item_map=candidate_boost_item_map,
                demographic=demographic,
                K=K,
                C=C,
            )

            safe = is_safe_multigroup_repair(
                before_result=current_result,
                after_result=trial_result,
                target_groups=target_groups,
                best_group=best_group,
                demographic=demographic,
                metric_col=metric_col,
                utility_tolerance=utility_tolerance,
                best_group_tolerance=best_group_tolerance,
            )

            gap_reduction = current_result["gap"] - trial_result["gap"]
            recall_delta = (
                trial_result["overall_recall"]
                - current_result["overall_recall"]
            )

            target_deltas = {}

            for group in target_groups:
                before_val = get_group_value(
                    current_result["group_table"],
                    group,
                    demographic,
                    metric_col,
                )

                after_val = get_group_value(
                    trial_result["group_table"],
                    group,
                    demographic,
                    metric_col,
                )

                target_deltas[group] = after_val - before_val

            repair_size = sum(trial_alpha_map.values())
            satisfies_eps = trial_result["gap"] <= target_eps

            tried_rows.append(
                {
                    "target_groups": tuple(target_groups),
                    "best_group": best_group,
                    "top_n": top_n,
                    "alpha_delta": alpha_delta,
                    "gap": trial_result["gap"],
                    "gap_reduction": gap_reduction,
                    "overall_recall": trial_result["overall_recall"],
                    "recall_delta": recall_delta,
                    "safe": safe,
                    "satisfies_eps": satisfies_eps,
                    "repair_size": repair_size,
                    "alpha_map": dict(trial_alpha_map),
                    "target_deltas": dict(target_deltas),
                }
            )

            if safe:
                min_target_delta = min(target_deltas.values())

                score = (
                    int(satisfies_eps),
                    -repair_size,
                    trial_result["overall_recall"],
                    gap_reduction,
                    min_target_delta,
                )

                if best_candidate is None or score > best_score:
                    best_candidate = {
                        "result": trial_result,
                        "alpha_map": trial_alpha_map,
                        "boost_item_map": candidate_boost_item_map,
                        "top_n": top_n,
                        "alpha_delta": alpha_delta,
                        "repair_size": repair_size,
                        "satisfies_eps": satisfies_eps,
                        "score": score,
                    }

                    best_score = score

    return best_candidate, pd.DataFrame(tried_rows)


def cegar_alpha_repair_loop(
    model,
    user_items,
    train_df,
    test_df,
    users_df,
    u2i,
    m2i,
    demographic="age_group",
    eps=0.01,
    K=10,
    C=200,
    max_iters=20,
    patience=5,
    alpha_candidates=None,
    top_n_candidates=None,
    utility_tolerance=0.001,
    best_group_tolerance=0.0005,
    target_eps=None,
    repair_mode="below_mean",
    max_repair_groups=3,
):
    if target_eps is None:
        target_eps = eps

    metric_col = f"recall@{K}"

    current = evaluate_baseline_fairness(
        model=model,
        user_items=user_items,
        test_df=test_df,
        users_df=users_df,
        u2i=u2i,
        m2i=m2i,
        demographic=demographic,
        K=K,
    )

    best_result = current
    best_gap = current["gap"]
    best_alpha_map = {}

    alpha_map = {}
    boost_item_map = {}

    history = []
    all_candidate_rows = []

    no_improve_count = 0

    for it in range(max_iters):
        verification = verify_table(
            current["group_table"],
            metric_col=metric_col,
            eps=eps,
        )

        counterexample = get_counterexample(
            current["group_table"],
            demographic=demographic,
            metric_col=metric_col,
        )

        best_group = counterexample["best_group"]

        target_groups = get_underperforming_groups(
            group_table=current["group_table"],
            demographic=demographic,
            metric_col=metric_col,
            mode=repair_mode,
            max_groups=max_repair_groups,
        )

        if len(target_groups) == 0:
            target_groups = [counterexample["worst_group"]]

        violation_amount = max(0.0, current["gap"] - eps)

        history.append(
            {
                "iteration": it,
                "gap": current["gap"],
                "overall_recall": current["overall_recall"],
                "violation": verification["violation"],
                "violation_amount": violation_amount,
                "target_groups": tuple(target_groups),
                "best_group": best_group,
                "worst_group": counterexample["worst_group"],
                "worst_value": counterexample["worst_value"],
                "best_value": counterexample["best_value"],
                "alpha_map": dict(alpha_map),
            }
        )

        if current["gap"] < best_gap:
            best_gap = current["gap"]
            best_result = current
            best_alpha_map = dict(alpha_map)
            no_improve_count = 0
        else:
            no_improve_count += 1

        if not verification["violation"]:
            return {
                "success": True,
                "final_result": current,
                "best_result": best_result,
                "history": pd.DataFrame(history),
                "candidate_history": pd.concat(all_candidate_rows, ignore_index=True)
                if all_candidate_rows
                else pd.DataFrame(),
                "best_alpha_map": best_alpha_map,
                "message": "Fairness constraint satisfied.",
            }

        candidate, tried_df = search_safe_multigroup_repair_candidate(
            model=model,
            user_items=user_items,
            train_df=train_df,
            test_df=test_df,
            users_df=users_df,
            u2i=u2i,
            m2i=m2i,
            current_result=current,
            alpha_map=alpha_map,
            boost_item_map=boost_item_map,
            target_groups=target_groups,
            best_group=best_group,
            demographic=demographic,
            K=K,
            C=C,
            alpha_candidates=alpha_candidates,
            top_n_candidates=top_n_candidates,
            utility_tolerance=utility_tolerance,
            best_group_tolerance=best_group_tolerance,
            target_eps=target_eps,
        )

        tried_df["iteration"] = it
        all_candidate_rows.append(tried_df)

        if candidate is None:
            no_improve_count += 1

            if no_improve_count >= patience:
                break

            continue

        current = candidate["result"]
        alpha_map = candidate["alpha_map"]
        boost_item_map = candidate["boost_item_map"]

        if current["gap"] < best_gap:
            best_gap = current["gap"]
            best_result = current
            best_alpha_map = dict(alpha_map)
            no_improve_count = 0
        else:
            no_improve_count += 1

        if no_improve_count >= patience:
            break

    return {
        "success": False,
        "final_result": current,
        "best_result": best_result,
        "history": pd.DataFrame(history),
        "candidate_history": pd.concat(all_candidate_rows, ignore_index=True)
        if all_candidate_rows
        else pd.DataFrame(),
        "best_alpha_map": best_alpha_map,
        "message": "Could not satisfy fairness constraint within limits. Returning best safe repair found.",
    }