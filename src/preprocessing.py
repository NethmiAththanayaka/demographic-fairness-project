import pandas as pd
from sklearn.model_selection import train_test_split


# --------------------------------------------------
# Simple Random Split
# --------------------------------------------------

def create_train_test(
    ratings,
    test_size=0.2,
    random_state=42
):
    """
    Random global train/test split.

    This is simple but NOT ideal for recommender systems,
    because future interactions may leak into training.
    """

    train, test = train_test_split(
        ratings,
        test_size=test_size,
        random_state=random_state
    )

    return train, test


# --------------------------------------------------
# Time-Based Per-User Split
# --------------------------------------------------

def time_based_user_split(
    ratings,
    user_col="user_id",
    time_col="timestamp",
    train_frac=0.8
):
    """
    Split each user's interactions chronologically.

    Earlier interactions -> train
    Later interactions -> test

    This matches realistic recommendation settings.
    """

    ratings = ratings.sort_values(
        [user_col, time_col]
    )

    train_parts = []
    test_parts = []

    for _, group in ratings.groupby(user_col):

        n = len(group)

        if n < 2:
            continue

        n_train = int(n * train_frac)

        if n_train <= 0:
            n_train = 1

        if n_train >= n:
            n_train = n - 1

        train_parts.append(group.iloc[:n_train])
        test_parts.append(group.iloc[n_train:])

    train = pd.concat(train_parts).reset_index(drop=True)
    test = pd.concat(test_parts).reset_index(drop=True)

    return train, test


# --------------------------------------------------
# Age Buckets
# --------------------------------------------------

AGE_MAP = {
    1: "Under18",
    18: "18-24",
    25: "25-34",
    35: "35-44",
    45: "45-49",
    50: "50-55",
    56: "56+"
}


def age_bucket(age_code):
    """
    Convert MovieLens age codes to readable labels.
    """

    return AGE_MAP.get(age_code, str(age_code))


# --------------------------------------------------
# Region Mapping
# --------------------------------------------------

def map_region(zipcode):
    """
    Very rough US-region grouping using ZIP prefix.
    """

    if pd.isna(zipcode):
        return "Unknown"

    zipcode = str(zipcode)

    if zipcode.startswith(("0", "1", "2")):
        return "East"

    elif zipcode.startswith(("3", "4", "5")):
        return "Central"

    else:
        return "West"


# --------------------------------------------------
# User Filtering
# --------------------------------------------------

def filter_users_with_min_interactions(
    ratings,
    min_interactions=5,
    user_col="user_id"
):
    """
    Remove users with too few interactions.

    Useful for stable fairness metrics.
    """

    counts = ratings[user_col].value_counts()

    keep_users = counts[
        counts >= min_interactions
    ].index

    filtered = ratings[
        ratings[user_col].isin(keep_users)
    ]

    return filtered.reset_index(drop=True)


# --------------------------------------------------
# Add Demographic Columns
# --------------------------------------------------

def enrich_user_demographics(users):
    """
    Add readable demographic columns.
    """

    users = users.copy()

    users["age_group"] = users["age"].apply(age_bucket)

    if "zip" in users.columns:
        users["region"] = users["zip"].apply(map_region)

    return users