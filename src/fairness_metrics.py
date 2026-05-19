# Computes demographic fairness results

import numpy as np
import pandas as pd


# --------------------------------------------------
# Basic Demographic Table
# --------------------------------------------------

def demo_table(
    recall_df,
    users,
    demographic="gender",
    metric_col="recall@10",
    min_users=1
):
    """
    Compute average recommendation metric by demographic group.
    """

    merged = recall_df.merge(
        users,
        on="user_id",
        how="inner"
    )

    table = (
        merged
        .groupby(demographic)
        .agg(
            **{
                metric_col: (metric_col, "mean"),
                "num_users": ("user_id", "nunique"),
                "num_rows": ("user_id", "count")
            }
        )
        .reset_index()
    )

    table = table[
        table["num_users"] >= min_users
    ].reset_index(drop=True)

    return table


# --------------------------------------------------
# Specific Demographic Wrappers
# --------------------------------------------------

def demo_table_gender(
    recall_df,
    users,
    metric_col="recall@10",
    min_users=1
):
    return demo_table(
        recall_df,
        users,
        demographic="gender",
        metric_col=metric_col,
        min_users=min_users
    )


def demo_table_age(
    recall_df,
    users,
    metric_col="recall@10",
    min_users=1
):
    age_col = "age_group" if "age_group" in users.columns else "age"

    return demo_table(
        recall_df,
        users,
        demographic=age_col,
        metric_col=metric_col,
        min_users=min_users
    )


def demo_table_region(
    recall_df,
    users,
    metric_col="recall@10",
    min_users=1
):
    return demo_table(
        recall_df,
        users,
        demographic="region",
        metric_col=metric_col,
        min_users=min_users
    )


def demo_table_country(
    recall_df,
    users,
    metric_col="recall@10",
    min_users=1
):
    return demo_table(
        recall_df,
        users,
        demographic="country",
        metric_col=metric_col,
        min_users=min_users
    )


# --------------------------------------------------
# Intersectional Fairness
# --------------------------------------------------

def add_intersection_column(
    users,
    cols=("gender", "age_group"),
    new_col="intersection_group"
):
    """
    Add intersectional group column, e.g. gender × age_group.
    """

    users = users.copy()

    missing = [c for c in cols if c not in users.columns]

    if missing:
        raise ValueError(
            f"Missing columns for intersectional grouping: {missing}"
        )

    users[new_col] = users[list(cols)].astype(str).agg("_".join, axis=1)

    return users


def demo_table_intersectional(
    recall_df,
    users,
    cols=("gender", "age_group"),
    metric_col="recall@10",
    min_users=1,
    new_col="intersection_group"
):
    users_inter = add_intersection_column(
        users,
        cols=cols,
        new_col=new_col
    )

    return demo_table(
        recall_df,
        users_inter,
        demographic=new_col,
        metric_col=metric_col,
        min_users=min_users
    )


# --------------------------------------------------
# Gap / Worst Pair Computation
# --------------------------------------------------

def max_gap(
    table,
    metric_col="recall@10"
):
    vals = table[metric_col].dropna().values

    if len(vals) == 0:
        return 0.0

    return float(np.max(vals) - np.min(vals))


def max_gap_with_pair(
    table,
    group_col=None,
    metric_col="recall@10"
):
    """
    Return fairness gap and the worst/best performing groups.
    """

    clean = table.dropna(subset=[metric_col]).copy()

    if len(clean) == 0:
        return {
            "gap": 0.0,
            "worst_group": None,
            "best_group": None,
            "worst_value": None,
            "best_value": None,
        }

    if group_col is None:
        group_col = clean.columns[0]

    worst_idx = clean[metric_col].idxmin()
    best_idx = clean[metric_col].idxmax()

    worst_row = clean.loc[worst_idx]
    best_row = clean.loc[best_idx]

    return {
        "gap": float(best_row[metric_col] - worst_row[metric_col]),
        "worst_group": worst_row[group_col],
        "best_group": best_row[group_col],
        "worst_value": float(worst_row[metric_col]),
        "best_value": float(best_row[metric_col]),
    }


# --------------------------------------------------
# Fairness Summary
# --------------------------------------------------

def fairness_summary_from_table(
    table,
    group_col=None,
    metric_col="recall@10"
):
    """
    Summarize group-level fairness table.
    """

    clean = table.dropna(subset=[metric_col]).copy()

    if len(clean) == 0:
        return {
            "overall_metric": 0.0,
            "gap": 0.0,
            "worst_pair": (None, None),
            "worst_group": None,
            "best_group": None,
            "worst_value": None,
            "best_value": None,
            "num_groups": 0
        }

    if group_col is None:
        group_col = clean.columns[0]

    pair_info = max_gap_with_pair(
        clean,
        group_col=group_col,
        metric_col=metric_col
    )

    return {
        "overall_metric": float(clean[metric_col].mean()),
        "gap": pair_info["gap"],
        "worst_pair": (
            pair_info["worst_group"],
            pair_info["best_group"]
        ),
        "worst_group": pair_info["worst_group"],
        "best_group": pair_info["best_group"],
        "worst_value": pair_info["worst_value"],
        "best_value": pair_info["best_value"],
        "num_groups": int(len(clean))
    }


# --------------------------------------------------
# Multiple Fairness Tables at Once
# --------------------------------------------------

def compute_all_demographic_tables(
    recall_df,
    users,
    metric_col="recall@10",
    min_users=1
):
    """
    Compute fairness tables for all demographic columns available.
    """

    results = {}

    if "gender" in users.columns:
        results["gender"] = demo_table_gender(
            recall_df,
            users,
            metric_col=metric_col,
            min_users=min_users
        )

    if "age_group" in users.columns or "age" in users.columns:
        results["age"] = demo_table_age(
            recall_df,
            users,
            metric_col=metric_col,
            min_users=min_users
        )

    if "region" in users.columns:
        results["region"] = demo_table_region(
            recall_df,
            users,
            metric_col=metric_col,
            min_users=min_users
        )

    if "country" in users.columns:
        results["country"] = demo_table_country(
            recall_df,
            users,
            metric_col=metric_col,
            min_users=min_users
        )

    if "gender" in users.columns and "age_group" in users.columns:
        results["gender_age"] = demo_table_intersectional(
            recall_df,
            users,
            cols=("gender", "age_group"),
            metric_col=metric_col,
            min_users=min_users
        )

    return results


def summarize_all_demographic_tables(
    tables,
    metric_col="recall@10"
):
    """
    Convert multiple demographic fairness tables into summary rows.
    """

    rows = []

    for demographic, table in tables.items():

        if len(table) == 0:
            continue

        group_col = table.columns[0]

        summary = fairness_summary_from_table(
            table,
            group_col=group_col,
            metric_col=metric_col
        )

        rows.append({
            "demographic": demographic,
            "overall_metric": summary["overall_metric"],
            "gap": summary["gap"],
            "worst_group": summary["worst_group"],
            "best_group": summary["best_group"],
            "worst_value": summary["worst_value"],
            "best_value": summary["best_value"],
            "num_groups": summary["num_groups"]
        })

    return pd.DataFrame(rows)


# --------------------------------------------------
# Before / After Repair Comparison
# --------------------------------------------------

def compare_fairness_tables(
    before_table,
    after_table,
    group_col=None,
    metric_col="recall@10"
):
    """
    Compare group-level fairness before and after mitigation/repair.
    """

    if group_col is None:
        group_col = before_table.columns[0]

    before = before_table[[group_col, metric_col]].rename(
        columns={metric_col: "before"}
    )

    after = after_table[[group_col, metric_col]].rename(
        columns={metric_col: "after"}
    )

    merged = before.merge(
        after,
        on=group_col,
        how="outer"
    )

    merged["delta"] = merged["after"] - merged["before"]

    return merged