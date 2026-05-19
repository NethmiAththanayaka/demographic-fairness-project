# SMT-based fairness verification utilities

from z3 import *


# --------------------------------------------------
# Basic threshold checks
# --------------------------------------------------

def violated(gap, eps=0.01):
    """
    Simple Python-level check.
    """
    return gap > eps


def verify_maxgap(gap, eps=0.01):
    """
    Check whether the observed fairness gap violates epsilon.

    Returns:
        sat     -> violation exists, gap > eps
        unsat   -> no violation, gap <= eps
    """

    s = Solver()

    g = Real("g")
    e = Real("eps")

    s.add(g == float(gap))
    s.add(e == float(eps))

    s.add(g > e)

    return s.check()


def verify_no_violation(gap, eps=0.01):
    """
    Check whether the system satisfies gap <= eps.
    """

    s = Solver()

    g = Real("g")
    e = Real("eps")

    s.add(g == float(gap))
    s.add(e == float(eps))

    s.add(g <= e)

    return s.check()


# --------------------------------------------------
# Epsilon helpers
# --------------------------------------------------

def approx_eps_star(gap, margin=1e-6):
    """
    Smallest epsilon that would make current gap pass.
    """

    return float(gap + margin)


def check_gap_against_eps_list(gap, eps_values):
    """
    Check one gap against multiple epsilon thresholds.
    """

    rows = []

    for eps in eps_values:
        result = verify_maxgap(gap, eps)

        rows.append({
            "eps": eps,
            "gap": gap,
            "violation": result == sat,
            "z3_result": str(result)
        })

    return rows


# --------------------------------------------------
# Verify from fairness summary/table
# --------------------------------------------------

def verify_summary(summary, eps=0.01):
    """
    Verify fairness from summary dictionary.
    Expected summary contains key: 'gap'
    """

    gap = summary["gap"]

    return {
        "gap": gap,
        "eps": eps,
        "violation": violated(gap, eps),
        "z3_result": str(verify_maxgap(gap, eps)),
        "eps_star": approx_eps_star(gap)
    }


def verify_table(table, metric_col="recall@10", eps=0.01):
    """
    Verify fairness directly from a demographic metric table.
    """

    vals = table[metric_col].dropna().values

    if len(vals) == 0:
        return {
            "gap": 0.0,
            "eps": eps,
            "violation": False,
            "z3_result": "unsat",
            "eps_star": eps
        }

    gap = float(max(vals) - min(vals))

    return {
        "gap": gap,
        "eps": eps,
        "violation": violated(gap, eps),
        "z3_result": str(verify_maxgap(gap, eps)),
        "eps_star": approx_eps_star(gap)
    }


# --------------------------------------------------
# Pairwise demographic verification
# --------------------------------------------------

def verify_pairwise_table(
    table,
    group_col=None,
    metric_col="recall@10",
    eps=0.01
):
    """
    Check every pair of demographic groups.

    A violation exists when:
        abs(metric_i - metric_j) > eps
    """

    if group_col is None:
        group_col = table.columns[0]

    clean = table.dropna(subset=[metric_col]).copy()

    rows = []

    for i in range(len(clean)):
        for j in range(i + 1, len(clean)):

            g1 = clean.iloc[i][group_col]
            g2 = clean.iloc[j][group_col]

            v1 = float(clean.iloc[i][metric_col])
            v2 = float(clean.iloc[j][metric_col])

            gap = abs(v1 - v2)

            result = verify_abs_gap(v1, v2, eps)

            rows.append({
                "group_1": g1,
                "group_2": g2,
                "value_1": v1,
                "value_2": v2,
                "gap": gap,
                "eps": eps,
                "violation": result == sat,
                "z3_result": str(result)
            })

    return rows


def verify_abs_gap(value_1, value_2, eps=0.01):
    """
    Z3 check for:
        abs(value_1 - value_2) > eps
    """

    s = Solver()

    x = Real("x")
    y = Real("y")
    e = Real("eps")

    s.add(x == float(value_1))
    s.add(y == float(value_2))
    s.add(e == float(eps))

    s.add(Abs(x - y) > e)

    return s.check()


# --------------------------------------------------
# Alpha selection helpers
# --------------------------------------------------

def choose_first_fair_alpha(
    alpha_results,
    gap_col="gap",
    recall_col="overall_recall",
    eps=0.01
):
    """
    Choose the first alpha whose fairness gap is <= eps.

    alpha_results should be a DataFrame with:
        alpha, gap, overall_recall
    """

    fair = alpha_results[
        alpha_results[gap_col] <= eps
    ].copy()

    if len(fair) == 0:
        return None

    fair = fair.sort_values("alpha")

    return fair.iloc[0].to_dict()


def choose_best_fair_alpha(
    alpha_results,
    gap_col="gap",
    recall_col="overall_recall",
    eps=0.01
):
    """
    Choose the fair alpha with the best utility.

    Utility is measured using recall_col.
    """

    fair = alpha_results[
        alpha_results[gap_col] <= eps
    ].copy()

    if len(fair) == 0:
        return None

    best_idx = fair[recall_col].idxmax()

    return fair.loc[best_idx].to_dict()


def smt_choose_alpha(
    alpha_results,
    eps=0.01,
    alpha_col="alpha",
    gap_col="gap",
    recall_col="overall_recall"
):
    """
    SMT-style selection over a finite alpha grid.

    Goal:
        choose an alpha such that gap <= eps

    Tie-breaker:
        among fair alphas, select highest recall.

    This function uses Python to evaluate the finite candidates,
    but the fairness feasibility condition is checked with Z3.
    """

    candidates = []

    for _, row in alpha_results.iterrows():

        alpha = float(row[alpha_col])
        gap = float(row[gap_col])
        recall = float(row[recall_col])

        result = verify_no_violation(gap, eps)

        if result == sat:
            candidates.append({
                "alpha": alpha,
                "gap": gap,
                "overall_recall": recall,
                "z3_result": str(result)
            })

    if len(candidates) == 0:
        return None

    candidates = sorted(
        candidates,
        key=lambda x: x["overall_recall"],
        reverse=True
    )

    return candidates[0]


# --------------------------------------------------
# Repair feasibility checks
# --------------------------------------------------

def verify_repair_improves_gap(
    before_gap,
    after_gap
):
    """
    Check whether after_gap < before_gap.
    """

    s = Solver()

    b = Real("before_gap")
    a = Real("after_gap")

    s.add(b == float(before_gap))
    s.add(a == float(after_gap))

    s.add(a < b)

    return s.check()


def verify_repair_satisfies_eps(
    after_gap,
    eps=0.01
):
    """
    Check whether repaired system satisfies gap <= eps.
    """

    return verify_no_violation(after_gap, eps)


def repair_summary(
    before_gap,
    after_gap,
    eps=0.01
):
    """
    Summarize repair result.
    """

    improves = verify_repair_improves_gap(
        before_gap,
        after_gap
    )

    satisfies = verify_repair_satisfies_eps(
        after_gap,
        eps
    )

    return {
        "before_gap": before_gap,
        "after_gap": after_gap,
        "gap_reduction": before_gap - after_gap,
        "eps": eps,
        "improves": improves == sat,
        "satisfies_eps": satisfies == sat,
        "z3_improves": str(improves),
        "z3_satisfies_eps": str(satisfies)
    }