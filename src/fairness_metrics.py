def demo_table(recall_df, users, demographic="gender"):

    merged = recall_df.merge(users, on="user_id")

    table = (
        merged
        .groupby(demographic)["recall@10"]
        .mean()
        .reset_index()
    )

    return table

def fairness_summary_from_table(table):

    vals = table["recall@10"].values

    return {
        "overall_recall": float(table["recall@10"].mean()),
        "gap": float(vals.max() - vals.min()),
        "worst_pair": (
            table.iloc[vals.argmin(), 0],
            table.iloc[vals.argmax(), 0]
        )
    }
