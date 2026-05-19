# MovieLens experiment pipeline

import pandas as pd

from src.data_loader import load_movielens
from src.preprocessing import (
    time_based_user_split,
    enrich_user_demographics,
)
from src.matrix_builder import build_sparse_matrix
from src.model import train_als

from src.baselines import (
    evaluate_global_mean_baseline,
    evaluate_user_item_bias_baseline,
)

from src.evaluation import (
    recall_at_k_for_users_model,
    max_gap,
)

from src.fairness_metrics import (
    demo_table_gender,
    demo_table_age,
    demo_table_intersectional,
    fairness_summary_from_table,
)

from src.mitigation import sweep_group_boosts
from src.reranking import (
    get_longtail_items,
    get_popular_items_for_group,
    evaluate_longtail_rerank,
    evaluate_group_rerank,
    sweep_longtail_alpha,
    sweep_group_alpha,
)
from src.alpha_search import select_alpha
from src.smt_verification import verify_table


# --------------------------------------------------
# Prepare MovieLens
# --------------------------------------------------

def prepare_movielens_data(data_dir="data/ml-1m"):
    ratings, users, movies = load_movielens(data_dir)

    users = enrich_user_demographics(users)

    return ratings, users, movies


# --------------------------------------------------
# Train ALS Baseline
# --------------------------------------------------

def train_movielens_als(
    ratings,
    factors=64,
    regularization=0.01,
    iterations=20,
):
    train, test = time_based_user_split(
        ratings,
        user_col="user_id",
        time_col="timestamp",
        train_frac=0.8,
    )

    user_items, u2i, i2u, m2i, i2m = build_sparse_matrix(train)

    model = train_als(
        user_items,
        factors=factors,
        regularization=regularization,
        iterations=iterations,
    )

    return model, user_items, train, test, u2i, i2u, m2i, i2m


# --------------------------------------------------
# RMSE Baselines
# --------------------------------------------------

def run_movielens_rmse_baselines(
    data_dir="data/ml-1m",
):
    ratings, users, movies = prepare_movielens_data(data_dir)

    train, test = time_based_user_split(
        ratings,
        user_col="user_id",
        time_col="timestamp",
        train_frac=0.8,
    )

    results = {}

    for demographic in ["gender", "age_group"]:
        global_result = evaluate_global_mean_baseline(
            train_df=train,
            test_df=test,
            users_df=users,
            demographic=demographic,
        )

        bias_result = evaluate_user_item_bias_baseline(
            train_df=train,
            test_df=test,
            users_df=users,
            demographic=demographic,
        )

        results[demographic] = {
            "global_mean": global_result,
            "user_item_bias": bias_result,
        }

    return results


# --------------------------------------------------
# ALS Fairness Baseline
# --------------------------------------------------

def evaluate_movielens_als_fairness(
    model,
    user_items,
    test,
    users,
    u2i,
    m2i,
    demographic="gender",
    K=10,
):
    recall_df = recall_at_k_for_users_model(
        model=model,
        user_items=user_items,
        test_df=test,
        users=users,
        u2i=u2i,
        m2i=m2i,
        K=K,
        relevance_threshold=4,
    )

    metric_col = f"recall@{K}"

    if demographic == "gender":
        table = demo_table_gender(
            recall_df,
            users,
            metric_col=metric_col,
        )

    elif demographic == "age":
        table = demo_table_age(
            recall_df,
            users,
            metric_col=metric_col,
        )

    elif demographic == "gender_age":
        table = demo_table_intersectional(
            recall_df,
            users,
            cols=("gender", "age_group"),
            metric_col=metric_col,
            min_users=5,
        )

    else:
        from src.fairness_metrics import demo_table

        table = demo_table(
            recall_df,
            users,
            demographic=demographic,
            metric_col=metric_col,
        )

    summary = fairness_summary_from_table(
        table,
        metric_col=metric_col,
    )

    return {
        "demographic": demographic,
        "recall_df": recall_df,
        "group_table": table,
        "summary": summary,
        "gap": max_gap(table, metric_col=metric_col),
        "overall_recall": float(recall_df[metric_col].mean())
        if len(recall_df) > 0
        else 0.0,
    }


def run_movielens_baseline(
    data_dir="data/ml-1m",
    K=10,
    factors=64,
    regularization=0.01,
    iterations=20,
):
    ratings, users, movies = prepare_movielens_data(data_dir)

    model, user_items, train, test, u2i, i2u, m2i, i2m = train_movielens_als(
        ratings=ratings,
        factors=factors,
        regularization=regularization,
        iterations=iterations,
    )

    results = {}

    for demographic in ["gender", "age", "gender_age"]:
        results[demographic] = evaluate_movielens_als_fairness(
            model=model,
            user_items=user_items,
            test=test,
            users=users,
            u2i=u2i,
            m2i=m2i,
            demographic=demographic,
            K=K,
        )

    return {
        "ratings": ratings,
        "users": users,
        "movies": movies,
        "model": model,
        "user_items": user_items,
        "train": train,
        "test": test,
        "u2i": u2i,
        "i2u": i2u,
        "m2i": m2i,
        "i2m": i2m,
        "results": results,
    }


# --------------------------------------------------
# Explicit Gender/Age Fairness
# --------------------------------------------------

def run_movielens_gender_age_fairness(
    data_dir="data/ml-1m",
    K=10,
    factors=64,
    regularization=0.01,
    iterations=20,
):
    baseline = run_movielens_baseline(
        data_dir=data_dir,
        K=K,
        factors=factors,
        regularization=regularization,
        iterations=iterations,
    )

    return {
        "gender": baseline["results"]["gender"],
        "age": baseline["results"]["age"],
        "gender_age": baseline["results"]["gender_age"],
        "full": baseline,
    }


# --------------------------------------------------
# Weighted ALS Mitigation
# --------------------------------------------------

def run_movielens_weighted_mitigation(
    data_dir="data/ml-1m",
    target_group="Under18",
    demographic="age_group",
    boost_values=None,
    K=10,
    factors=64,
    regularization=0.01,
    iterations=20,
):
    ratings, users, movies = prepare_movielens_data(data_dir)

    train, test = time_based_user_split(
        ratings,
        user_col="user_id",
        time_col="timestamp",
        train_frac=0.8,
    )

    sweep_df, full_results = sweep_group_boosts(
        train_df=train,
        test_df=test,
        users_df=users,
        target_group=target_group,
        demographic=demographic,
        boost_values=boost_values,
        K=K,
        factors=factors,
        regularization=regularization,
        iterations=iterations,
    )

    return {
        "sweep": sweep_df,
        "results": full_results,
        "train": train,
        "test": test,
        "users": users,
        "movies": movies,
    }


# --------------------------------------------------
# Long-Tail / Alpha Reranking
# --------------------------------------------------

def run_movielens_alpha_reranking(
    data_dir="data/ml-1m",
    demographic="age_group",
    target_group="Under18",
    alpha_values=None,
    eps=0.01,
    K=10,
    C=200,
    factors=64,
    regularization=0.01,
    iterations=20,
):
    ratings, users, movies = prepare_movielens_data(data_dir)

    model, user_items, train, test, u2i, i2u, m2i, i2m = train_movielens_als(
        ratings=ratings,
        factors=factors,
        regularization=regularization,
        iterations=iterations,
    )

    metric_col = f"recall@{K}"

    baseline_result = evaluate_movielens_als_fairness(
        model=model,
        user_items=user_items,
        test=test,
        users=users,
        u2i=u2i,
        m2i=m2i,
        demographic=demographic,
        K=K,
    )

    longtail_items = get_longtail_items(
        train_df=train,
        m2i=m2i,
        item_col="movie_id",
        quantile=0.25,
    )

    group_items = get_popular_items_for_group(
        train_df=train,
        users_df=users,
        target_group=target_group,
        demographic=demographic,
        item_col="movie_id",
        user_col="user_id",
        m2i=m2i,
        top_n=200,
    )

    longtail_sweep, longtail_results = sweep_longtail_alpha(
        model=model,
        user_items=user_items,
        test_df=test,
        users_df=users,
        u2i=u2i,
        m2i=m2i,
        longtail_items=longtail_items,
        demographic=demographic,
        alpha_values=alpha_values,
        K=K,
        C=C,
    )

    group_sweep, group_results = sweep_group_alpha(
        model=model,
        user_items=user_items,
        test_df=test,
        users_df=users,
        u2i=u2i,
        m2i=m2i,
        target_group=target_group,
        demographic=demographic,
        boost_items=group_items,
        alpha_values=alpha_values,
        K=K,
        C=C,
    )

    selected_longtail = select_alpha(
        longtail_sweep,
        eps=eps,
        strategy="best_fair",
    )

    selected_group = select_alpha(
        group_sweep,
        eps=eps,
        strategy="best_fair",
    )

    return {
        "baseline": baseline_result,
        "longtail_items": longtail_items,
        "group_items": group_items,
        "longtail_sweep": longtail_sweep,
        "longtail_results": longtail_results,
        "group_sweep": group_sweep,
        "group_results": group_results,
        "selected_longtail": selected_longtail,
        "selected_group": selected_group,
        "model": model,
        "user_items": user_items,
        "train": train,
        "test": test,
        "users": users,
        "movies": movies,
        "u2i": u2i,
        "i2u": i2u,
        "m2i": m2i,
        "i2m": i2m,
    }


# --------------------------------------------------
# SMT Verification Wrapper
# --------------------------------------------------

def run_movielens_smt_checks(
    data_dir="data/ml-1m",
    eps=0.01,
    K=10,
):
    baseline = run_movielens_baseline(
        data_dir=data_dir,
        K=K,
    )

    metric_col = f"recall@{K}"

    checks = {}

    for demographic, result in baseline["results"].items():
        table = result["group_table"]

        checks[demographic] = verify_table(
            table,
            metric_col=metric_col,
            eps=eps,
        )

    return {
        "baseline": baseline,
        "checks": checks,
    }


# --------------------------------------------------
# Print Helpers
# --------------------------------------------------

def print_rmse_results(results):
    for demographic, result in results.items():
        print("\n==============================")
        print(f"RMSE Baselines by {demographic}")
        print("==============================")

        print("\nGlobal Mean Overall RMSE:")
        print(result["global_mean"]["overall_rmse"])

        print("\nGlobal Mean Group RMSE:")
        print(result["global_mean"]["group_rmse"])

        print("\nUser-Item Bias Overall RMSE:")
        print(result["user_item_bias"]["overall_rmse"])

        print("\nUser-Item Bias Group RMSE:")
        print(result["user_item_bias"]["group_rmse"])


def print_movielens_results(results):
    for demographic, result in results.items():
        print("\n==============================")
        print(f"MovieLens fairness: {demographic}")
        print("==============================")

        print(result["group_table"])
        print("Summary:", result["summary"])
        print("Gap:", result["gap"])
        print("Overall Recall:", result["overall_recall"])