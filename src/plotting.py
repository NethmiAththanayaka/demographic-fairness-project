# Plotting utilities

import matplotlib.pyplot as plt


def plot_alpha_sweep(
    sweep_df,
    x_col="alpha",
    y_col="gap",
    title="Fairness Gap vs Alpha",
):
    plt.figure(figsize=(6, 4))
    plt.plot(sweep_df[x_col], sweep_df[y_col], marker="o")
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(title)
    plt.grid(True)
    plt.show()


def plot_recall_vs_alpha(
    sweep_df,
    x_col="alpha",
    y_col="overall_recall",
    title="Overall Recall vs Alpha",
):
    plt.figure(figsize=(6, 4))
    plt.plot(sweep_df[x_col], sweep_df[y_col], marker="o")
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(title)
    plt.grid(True)
    plt.show()


def plot_fairness_utility_tradeoff(
    sweep_df,
    x_col="gap",
    y_col="overall_recall",
    title="Fairness-Utility Tradeoff",
):
    plt.figure(figsize=(6, 4))
    plt.scatter(sweep_df[x_col], sweep_df[y_col])

    for _, row in sweep_df.iterrows():
        plt.annotate(
            str(row["alpha"]),
            (row[x_col], row[y_col]),
            fontsize=8,
        )

    plt.xlabel("Fairness Gap")
    plt.ylabel("Overall Recall")
    plt.title(title)
    plt.grid(True)
    plt.show()


def plot_group_metric_bar(
    group_table,
    group_col=None,
    metric_col="recall@10",
    title="Group Metric",
):
    if group_col is None:
        group_col = group_table.columns[0]

    plt.figure(figsize=(7, 4))
    plt.bar(group_table[group_col].astype(str), group_table[metric_col])
    plt.xlabel(group_col)
    plt.ylabel(metric_col)
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def plot_before_after_bars(
    comparison_df,
    group_col=None,
    before_col="before",
    after_col="after",
    title="Before vs After Repair",
):
    if group_col is None:
        group_col = comparison_df.columns[0]

    x = range(len(comparison_df))
    width = 0.35

    plt.figure(figsize=(8, 4))

    plt.bar(
        [i - width / 2 for i in x],
        comparison_df[before_col],
        width=width,
        label="Before",
    )

    plt.bar(
        [i + width / 2 for i in x],
        comparison_df[after_col],
        width=width,
        label="After",
    )

    plt.xticks(
        list(x),
        comparison_df[group_col].astype(str),
        rotation=45,
        ha="right",
    )

    plt.xlabel(group_col)
    plt.ylabel("Metric")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_gap_comparison(
    names,
    gaps,
    title="Fairness Gap Comparison",
):
    plt.figure(figsize=(6, 4))
    plt.bar(names, gaps)
    plt.ylabel("Fairness Gap")
    plt.title(title)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.show()


def plot_recall_comparison(
    names,
    recalls,
    title="Overall Recall Comparison",
):
    plt.figure(figsize=(6, 4))
    plt.bar(names, recalls)
    plt.ylabel("Overall Recall")
    plt.title(title)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.show()


def plot_rmse_by_group(
    group_rmse_df,
    group_col=None,
    metric_col="rmse",
    title="RMSE by Group",
):
    if group_col is None:
        group_col = group_rmse_df.columns[0]

    plt.figure(figsize=(7, 4))
    plt.bar(group_rmse_df[group_col].astype(str), group_rmse_df[metric_col])
    plt.xlabel(group_col)
    plt.ylabel(metric_col)
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def plot_alpha_results_all(sweep_df, title_prefix="Alpha Sweep"):
    plot_alpha_sweep(
        sweep_df,
        title=f"{title_prefix}: Fairness Gap vs Alpha",
    )

    plot_recall_vs_alpha(
        sweep_df,
        title=f"{title_prefix}: Overall Recall vs Alpha",
    )

    plot_fairness_utility_tradeoff(
        sweep_df,
        title=f"{title_prefix}: Fairness-Utility Tradeoff",
    )