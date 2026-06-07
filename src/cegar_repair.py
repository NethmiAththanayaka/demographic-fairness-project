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


def is_safe_repair(
    before_result,
    after_result,
    target_group,
    best_group,
    demographic,
    metric_col,
    utility_tolerance=0.001,
    best_group_tolerance=0.0005,
):
    """
    Accept repair only if:
      1. gap decreases
      2. target group improves
      3. overall recall does not drop much
      4. best group is not harmed much
    """

    before_table = before_result["group_table"]
    after_table = after_result["group_table"]

    before_gap = before_result["gap"]
    after_gap = after_result["gap"]

    before_target = get_group_value(
        before_table,
        target_group,
        demographic,
        metric_col,
    )

    after_target = get_group_value(
        after_table,
        target_group,
        demographic,
        metric_col,
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

    if before_target is None or after_target is None:
        return False

    if before_best is None or after_best is None:
        return False

    gap_improves = after_gap < before_gap
    target_improves = after_target >= before_target
    utility_safe = (
        after_result["overall_recall"]
        >= before_result["overall_recall"] - utility_tolerance
    )
    best_not_harmed = after_best >= before_best - best_group_tolerance

    return (
        gap_improves
        and target_improves
        and utility_safe
        and best_not_harmed
    )


def search_safe_repair_candidate(
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
    target_group,
    best_group,
    demographic="age_group",
    K=10,
    C=200,
    alpha_candidates=None,
    top_n_candidates=None,
    utility_tolerance=0.001,
    best_group_tolerance=0.0005,
):
    """
    Search candidate repairs and return the best safe one.

    This does NOT reduce the advantaged group.
    It only tries to improve the target group.
    """

    if alpha_candidates is None:
        alpha_candidates = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]

    if top_n_candidates is None:
        top_n_candidates = [50, 100, 200, 500]

    metric_col = f"recall@{K}"

    best_candidate = None
    best_score = None
    tried_rows = []

    for top_n in top_n_candidates:
        candidate_items = get_popular_items_for_group(
            train_df=train_df,
            users_df=users_df,
            target_group=target_group,
            demographic=demographic,
            item_col="movie_id",
            user_col="user_id",
            m2i=m2i,
            top_n=top_n,
        )

        for alpha_delta in alpha_candidates:
            trial_alpha_map = dict(alpha_map)
            trial_boost_item_map = dict(boost_item_map)

            old_alpha = trial_alpha_map.get(target_group, 0.0)
            trial_alpha_map[target_group] = min(
                old_alpha + alpha_delta,
                1.0,
            )

            trial_boost_item_map[target_group] = candidate_items

            trial_result = evaluate_alpha_map_repair(
                model=model,
                user_items=user_items,
                test_df=test_df,
                users_df=users_df,
                u2i=u2i,
                m2i=m2i,
                alpha_map=trial_alpha_map,
                boost_item_map=trial_boost_item_map,
                demographic=demographic,
                K=K,
                C=C,
            )

            safe = is_safe_repair(
                before_result=current_result,
                after_result=trial_result,
                target_group=target_group,
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

            target_before = get_group_value(
                current_result["group_table"],
                target_group,
                demographic,
                metric_col,
            )
            target_after = get_group_value(
                trial_result["group_table"],
                target_group,
                demographic,
                metric_col,
            )

            tried_rows.append({
                "target_group": target_group,
                "best_group": best_group,
                "top_n": top_n,
                "alpha_delta": alpha_delta,
                "new_alpha": trial_alpha_map[target_group],
                "gap": trial_result["gap"],
                "gap_reduction": gap_reduction,
                "overall_recall": trial_result["overall_recall"],
                "recall_delta": recall_delta,
                "target_before": target_before,
                "target_after": target_after,
                "target_delta": target_after - target_before,
                "safe": safe,
            })

            if safe:
                score = (
                    gap_reduction,
                    recall_delta,
                    target_after - target_before,
                )

                if best_candidate is None or score > best_score:
                    best_candidate = {
                        "result": trial_result,
                        "alpha_map": trial_alpha_map,
                        "boost_item_map": trial_boost_item_map,
                        "top_n": top_n,
                        "alpha_delta": alpha_delta,
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
):
    """
    Safe Counterexample-Guided Fairness Repair.

    At each iteration:
      1. Verify fairness.
      2. If violated, identify worst/best group pair.
      3. Search candidate repairs that only help the worst group.
      4. Accept only safe repairs:
            - gap decreases
            - target group improves
            - overall recall preserved
            - best group not harmed
      5. Repeat.
    """

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

        target_group = counterexample["worst_group"]
        best_group = counterexample["best_group"]
        violation_amount = max(0.0, current["gap"] - eps)

        history.append({
            "iteration": it,
            "gap": current["gap"],
            "overall_recall": current["overall_recall"],
            "violation": verification["violation"],
            "violation_amount": violation_amount,
            "target_group": target_group,
            "best_group": best_group,
            "worst_value": counterexample["worst_value"],
            "best_value": counterexample["best_value"],
            "alpha_map": dict(alpha_map),
        })

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
                if all_candidate_rows else pd.DataFrame(),
                "best_alpha_map": best_alpha_map,
                "message": "Fairness constraint satisfied.",
            }

        candidate, tried_df = search_safe_repair_candidate(
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
            target_group=target_group,
            best_group=best_group,
            demographic=demographic,
            K=K,
            C=C,
            alpha_candidates=alpha_candidates,
            top_n_candidates=top_n_candidates,
            utility_tolerance=utility_tolerance,
            best_group_tolerance=best_group_tolerance,
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
        if all_candidate_rows else pd.DataFrame(),
        "best_alpha_map": best_alpha_map,
        "message": "Could not satisfy fairness constraint within limits. Returning best safe repair found.",
    }