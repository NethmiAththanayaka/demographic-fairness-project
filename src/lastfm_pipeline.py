import os
import pandas as pd

from src.data_loader import load_lastfm
from src.preprocessing import time_based_user_split
from src.matrix_builder import build_sparse_matrix
from src.model import train_als
from src.evaluation import recall_at_k_for_users_model, max_gap
from src.fairness_metrics import (
    demo_table_gender,
    demo_table_age,
    demo_table_country,
    demo_table_region,
    demo_table_intersectional,
    fairness_summary_from_table,
)


def prepare_lastfm_data(data_dir="data/lastfm"):
    interactions, users, artists = load_lastfm(data_dir)

    if "weight" in interactions.columns and "rating" not in interactions.columns:
        interactions = interactions.rename(columns={"weight": "rating"})

    if "artistID" in interactions.columns:
        interactions = interactions.rename(columns={"artistID": "movie_id"})

    if "userID" in interactions.columns:
        interactions = interactions.rename(columns={"userID": "user_id"})

    if "id" in users.columns and "user_id" not in users.columns:
        users = users.rename(columns={"id": "user_id"})

    if "gender" not in users.columns:
        users["gender"] = "Unknown"

    if "age" in users.columns:
        users["age_group"] = users["age"].fillna("Unknown").astype(str)
    else:
        users["age_group"] = "Unknown"

    if "country" not in users.columns:
        users["country"] = "Unknown"

    users["region"] = users["country"].apply(map_country_to_region)

    interactions = interactions[["user_id", "movie_id", "rating"]].copy()
    interactions["timestamp"] = range(len(interactions))

    return interactions, users, artists


def map_country_to_region(country):
    if pd.isna(country):
        return "Unknown"

    country = str(country).lower()

    north_america = {
        "united states", "usa", "us", "canada", "mexico"
    }

    europe = {
        "united kingdom", "uk", "england", "germany", "france",
        "spain", "italy", "netherlands", "poland", "sweden",
        "norway", "finland", "denmark", "belgium", "austria",
        "switzerland", "portugal", "ireland", "greece"
    }

    asia = {
        "china", "japan", "india", "south korea", "korea",
        "taiwan", "singapore", "malaysia", "thailand",
        "indonesia", "philippines", "vietnam", "sri lanka"
    }

    south_america = {
        "brazil", "argentina", "chile", "colombia", "peru",
        "venezuela", "uruguay"
    }

    if country in north_america:
        return "North America"
    if country in europe:
        return "Europe"
    if country in asia:
        return "Asia"
    if country in south_america:
        return "South America"

    return "Other"


def train_lastfm_als(
    interactions,
    factors=64,
    regularization=0.01,
    iterations=20,
):
    train, test = time_based_user_split(
        interactions,
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


def evaluate_lastfm_demographic(
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
        relevance_threshold=1,
    )

    metric_col = f"recall@{K}"

    table = None

    if demographic == "gender":
        table = demo_table_gender(recall_df, users, metric_col=metric_col)

    elif demographic == "age":
        table = demo_table_age(recall_df, users, metric_col=metric_col)

    elif demographic == "country":
        table = demo_table_country(recall_df, users, metric_col=metric_col)

    elif demographic == "region":
        table = demo_table_region(recall_df, users, metric_col=metric_col)

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


def evaluate_lastfm_gender(*args, **kwargs):
    return evaluate_lastfm_demographic(*args, demographic="gender", **kwargs)


def evaluate_lastfm_age(*args, **kwargs):
    return evaluate_lastfm_demographic(*args, demographic="age", **kwargs)


def evaluate_lastfm_country(*args, **kwargs):
    return evaluate_lastfm_demographic(*args, demographic="country", **kwargs)


def evaluate_lastfm_region(*args, **kwargs):
    return evaluate_lastfm_demographic(*args, demographic="region", **kwargs)


def evaluate_lastfm_intersectional(*args, **kwargs):
    return evaluate_lastfm_demographic(*args, demographic="gender_age", **kwargs)


def run_lastfm_pipeline(
    data_dir="data/lastfm",
    K=10,
    factors=64,
    regularization=0.01,
    iterations=20,
):
    interactions, users, artists = prepare_lastfm_data(data_dir)

    model, user_items, train, test, u2i, i2u, m2i, i2m = train_lastfm_als(
        interactions,
        factors=factors,
        regularization=regularization,
        iterations=iterations,
    )

    results = {}

    for demographic in ["gender", "age", "country", "region", "gender_age"]:
        try:
            results[demographic] = evaluate_lastfm_demographic(
                model=model,
                user_items=user_items,
                test=test,
                users=users,
                u2i=u2i,
                m2i=m2i,
                demographic=demographic,
                K=K,
            )
        except Exception as e:
            results[demographic] = {
                "error": str(e)
            }

    return {
        "model": model,
        "user_items": user_items,
        "train": train,
        "test": test,
        "users": users,
        "artists": artists,
        "u2i": u2i,
        "i2u": i2u,
        "m2i": m2i,
        "i2m": i2m,
        "results": results,
    }


def print_lastfm_results(results):
    for demographic, result in results.items():
        print("\n==============================")
        print(f"Last.fm fairness: {demographic}")
        print("==============================")

        if "error" in result:
            print("Error:", result["error"])
            continue

        print(result["group_table"])
        print("Summary:", result["summary"])