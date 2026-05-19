# General utility functions

import os
import random
import numpy as np
import pandas as pd


def set_seed(seed=42):
    """
    Set random seeds for reproducibility.
    """

    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path):
    """
    Create directory if it does not exist.
    """

    os.makedirs(path, exist_ok=True)
    return path


def safe_mean(values, default=0.0):
    """
    Mean that returns default for empty values.
    """

    values = list(values)

    if len(values) == 0:
        return default

    return float(np.mean(values))


def safe_max(values, default=0.0):
    values = list(values)

    if len(values) == 0:
        return default

    return float(np.max(values))


def safe_min(values, default=0.0):
    values = list(values)

    if len(values) == 0:
        return default

    return float(np.min(values))


def print_section(title):
    """
    Pretty print section header.
    """

    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def print_subsection(title):
    """
    Pretty print subsection header.
    """

    print("\n" + "-" * 40)
    print(title)
    print("-" * 40)


def save_dataframe(df, path, index=False):
    """
    Save DataFrame as CSV.
    """

    parent = os.path.dirname(path)

    if parent:
        ensure_dir(parent)

    df.to_csv(path, index=index)

    print(f"Saved: {path}")


def load_dataframe(path):
    """
    Load CSV file.
    """

    return pd.read_csv(path)


def dataframe_or_empty(rows):
    """
    Convert rows to DataFrame safely.
    """

    if rows is None or len(rows) == 0:
        return pd.DataFrame()

    return pd.DataFrame(rows)


def check_required_columns(df, required_cols, name="DataFrame"):
    """
    Validate required columns.
    """

    missing = [
        col for col in required_cols
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{name} is missing required columns: {missing}"
        )


def maybe_sample_df(df, n=None, random_state=42):
    """
    Optionally sample a DataFrame.
    """

    if n is None:
        return df

    if len(df) <= n:
        return df

    return df.sample(
        n=n,
        random_state=random_state
    ).reset_index(drop=True)


def dict_to_dataframe(d):
    """
    Convert dictionary of scalar values to one-row DataFrame.
    """

    return pd.DataFrame([d])


def flatten_results_dict(results):
    """
    Flatten nested result dictionaries for easier printing/saving.
    """

    rows = []

    for key, value in results.items():

        if isinstance(value, dict):
            row = {"name": key}

            for k, v in value.items():
                if not isinstance(v, (dict, list, tuple, pd.DataFrame)):
                    row[k] = v

            rows.append(row)

    return pd.DataFrame(rows)


def get_metric_col(K=10, metric="recall"):
    """
    Build metric column name such as recall@10.
    """

    return f"{metric}@{K}"


def normalize_user_id_column(df):
    """
    Normalize common user-id column names to user_id.
    """

    df = df.copy()

    rename_map = {}

    if "userID" in df.columns:
        rename_map["userID"] = "user_id"

    if "userid" in df.columns:
        rename_map["userid"] = "user_id"

    if "user" in df.columns:
        rename_map["user"] = "user_id"

    return df.rename(columns=rename_map)


def normalize_item_id_column(df):
    """
    Normalize common item-id column names to movie_id.
    """

    df = df.copy()

    rename_map = {}

    if "movieID" in df.columns:
        rename_map["movieID"] = "movie_id"

    if "movieId" in df.columns:
        rename_map["movieId"] = "movie_id"

    if "artistID" in df.columns:
        rename_map["artistID"] = "movie_id"

    if "item_id" in df.columns:
        rename_map["item_id"] = "movie_id"

    return df.rename(columns=rename_map)


def print_dataframe(name, df, max_rows=20):
    """
    Print a DataFrame with a title.
    """

    print_section(name)

    if df is None:
        print("None")
        return

    if len(df) > max_rows:
        print(df.head(max_rows))
        print(f"... showing first {max_rows} of {len(df)} rows")
    else:
        print(df)