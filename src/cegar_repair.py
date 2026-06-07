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
    eta=2.0,
    alpha_max=1.0,
    max_iters=20,
    top_n_items=200,
    patience=5,
):
    """
    Counterexample-guided fairness repair with memory.

    At each iteration:
      1. Evaluate group fairness.
      2. SMT verifies gap <= eps.
      3. If violated, identify worst group as counterexample.
      4. Add/update that group in alpha_map.
      5. Re-evaluate repaired recommendations.
      6. Keep the best repair found.
    """

    metric_col = f"recall@{K}"

    history = []

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

    no_improve_count = 0
    seen_states = set()

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
        violation_amount = max(0.0, current["gap"] - eps)

        history.append(
            {
                "iteration": it,
                "gap": current["gap"],
                "overall_recall": current["overall_recall"],
                "violation": verification["violation"],
                "violation_amount": violation_amount,
                "target_group": target_group,
                "best_group": counterexample["best_group"],
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
                "best_alpha_map": best_alpha_map,
                "message": "Fairness constraint satisfied.",
            }

        state_key = (
            round(current["gap"], 6),
            target_group,
            counterexample["best_group"],
            tuple(sorted(alpha_map.items())),
        )

        if state_key in seen_states:
            no_improve_count += 1
        else:
            seen_states.add(state_key)

        if no_improve_count >= patience:
            break

        if target_group not in boost_item_map:
            boost_item_map[target_group] = get_popular_items_for_group(
                train_df=train_df,
                users_df=users_df,
                target_group=target_group,
                demographic=demographic,
                item_col="movie_id",
                user_col="user_id",
                m2i=m2i,
                top_n=top_n_items,
            )

        old_alpha = alpha_map.get(target_group, 0.0)
        new_alpha = min(old_alpha + eta * violation_amount, alpha_max)
        alpha_map[target_group] = new_alpha

        current = evaluate_alpha_map_repair(
            model=model,
            user_items=user_items,
            test_df=test_df,
            users_df=users_df,
            u2i=u2i,
            m2i=m2i,
            alpha_map=alpha_map,
            boost_item_map=boost_item_map,
            demographic=demographic,
            K=K,
            C=C,
        )

        if all(alpha >= alpha_max for alpha in alpha_map.values()):
            break

    return {
        "success": False,
        "final_result": current,
        "best_result": best_result,
        "history": pd.DataFrame(history),
        "best_alpha_map": best_alpha_map,
        "message": "Could not satisfy fairness constraint within limits. Returning best repair found.",
    }