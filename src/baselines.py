import numpy as np
import pandas as pd


# --------------------------------------------------
# Global Mean Baseline
# --------------------------------------------------

def global_mean_predictor(train_df):
    """
    Predict every rating using the global average rating.
    """

    global_mean = train_df["rating"].mean()

    return global_mean


def predict_global_mean(test_df, global_mean):
    """
    Return constant predictions for all test samples.
    """

    return np.full(len(test_df), global_mean)


# --------------------------------------------------
# User + Item Bias Baseline
# --------------------------------------------------

def fit_user_item_bias(train_df):
    """
    Simple baseline:

    prediction =
        global_mean
        + user_bias
        + item_bias
    """

    global_mean = train_df["rating"].mean()

    user_bias = (
        train_df.groupby("user_id")["rating"]
        .mean()
        - global_mean
    )

    item_bias = (
        train_df.groupby("movie_id")["rating"]
        .mean()
        - global_mean
    )

    return global_mean, user_bias, item_bias


def predict_user_item_bias(
    test_df,
    global_mean,
    user_bias,
    item_bias
):
    """
    Generate predictions using user/item biases.
    """

    preds = []

    for _, row in test_df.iterrows():

        u = row["user_id"]
        i = row["movie_id"]

        ub = user_bias.get(u, 0.0)
        ib = item_bias.get(i, 0.0)

        pred = global_mean + ub + ib

        pred = min(5.0, max(1.0, pred))

        preds.append(pred)

    return np.array(preds)


# --------------------------------------------------
# RMSE
# --------------------------------------------------

def rmse(predictions, targets):
    """
    Root Mean Squared Error.
    """

    predictions = np.array(predictions)
    targets = np.array(targets)

    return np.sqrt(
        np.mean((predictions - targets) ** 2)
    )


# --------------------------------------------------
# Group RMSE
# --------------------------------------------------

def group_rmse(
    test_df,
    predictions,
    users_df,
    demographic="gender"
):
    """
    Compute RMSE separately per demographic group.
    """

    df = test_df.copy()

    df["prediction"] = predictions

    df = df.merge(
        users_df,
        on="user_id"
    )

    results = []

    for group_name, group in df.groupby(demographic):

        score = rmse(
            group["prediction"],
            group["rating"]
        )

        results.append({
            demographic: group_name,
            "rmse": score,
            "count": len(group)
        })

    return pd.DataFrame(results)


# --------------------------------------------------
# Evaluate Global Mean Baseline
# --------------------------------------------------

def evaluate_global_mean_baseline(
    train_df,
    test_df,
    users_df,
    demographic="gender"
):
    """
    Full evaluation pipeline for global mean baseline.
    """

    global_mean = global_mean_predictor(train_df)

    preds = predict_global_mean(
        test_df,
        global_mean
    )

    overall_rmse = rmse(
        preds,
        test_df["rating"]
    )

    group_table = group_rmse(
        test_df,
        preds,
        users_df,
        demographic
    )

    return {
        "overall_rmse": overall_rmse,
        "group_rmse": group_table
    }


# --------------------------------------------------
# Evaluate User-Item Bias Baseline
# --------------------------------------------------

def evaluate_user_item_bias_baseline(
    train_df,
    test_df,
    users_df,
    demographic="gender"
):
    """
    Full evaluation pipeline for user-item bias baseline.
    """

    global_mean, user_bias, item_bias = (
        fit_user_item_bias(train_df)
    )

    preds = predict_user_item_bias(
        test_df,
        global_mean,
        user_bias,
        item_bias
    )

    overall_rmse = rmse(
        preds,
        test_df["rating"]
    )

    group_table = group_rmse(
        test_df,
        preds,
        users_df,
        demographic
    )

    return {
        "overall_rmse": overall_rmse,
        "group_rmse": group_table
    }