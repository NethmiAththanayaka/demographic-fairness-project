import pandas as pd


def summarize_repair_results(
    baseline_result,
    weighted_result=None,
    alpha_result=None,
    cegar_result=None,
    metric_name="Recall@10"
):
    rows = []

    rows.append({
        "method": "Baseline ALS",
        "overall_recall": baseline_result["overall_recall"],
        "gap": baseline_result["gap"],
        "success": baseline_result["gap"] <= 0.01,
    })

    if weighted_result is not None:
        best_weighted = weighted_result["sweep"].sort_values("gap").iloc[0]
        rows.append({
            "method": "Weighted mitigation",
            "overall_recall": best_weighted["overall_recall"],
            "gap": best_weighted["gap"],
            "success": best_weighted["gap"] <= 0.01,
        })

    if alpha_result is not None:
        best_group = alpha_result["group_sweep"].sort_values("gap").iloc[0]
        rows.append({
            "method": "Group alpha reranking",
            "overall_recall": best_group["overall_recall"],
            "gap": best_group["gap"],
            "success": best_group["gap"] <= 0.01,
        })

    if cegar_result is not None:
        rows.append({
            "method": "CEGAR repair",
            "overall_recall": cegar_result["best_result"]["overall_recall"],
            "gap": cegar_result["best_result"]["gap"],
            "success": cegar_result["success"],
        })

    return pd.DataFrame(rows)