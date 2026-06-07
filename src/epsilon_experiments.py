import pandas as pd


def certification_table(
    comparison_df,
    eps_values=None,
):
    if eps_values is None:
        eps_values = [0.01, 0.015, 0.02, 0.03]

    rows = []

    for _, row in comparison_df.iterrows():
        out = {
            "method": row["method"],
            "overall_recall": row["overall_recall"],
            "gap": row["gap"],
        }

        for eps in eps_values:
            out[f"certified_eps_{eps}"] = row["gap"] <= eps

        rows.append(out)

    return pd.DataFrame(rows)


def print_certification_table(table):
    print("\nCertification Table")
    print(table)