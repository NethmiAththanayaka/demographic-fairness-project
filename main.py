import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"

from src.utils import print_section
from src.movielens_pipeline import (
    run_movielens_rmse_baselines,
    run_movielens_baseline,
    run_movielens_weighted_mitigation,
    run_movielens_alpha_reranking,
    run_movielens_smt_checks,
    print_rmse_results,
    print_movielens_results,
)

from src.lastfm_pipeline import (
    run_lastfm_pipeline,
    print_lastfm_results,
)

from src.movielens_pipeline import run_movielens_cegar_repair
from src.cegar_compare import summarize_repair_results
from src.epsilon_experiments import certification_table

def run_movielens_all():
    print_section("MovieLens RMSE Baselines")
    rmse_results = run_movielens_rmse_baselines()
    print_rmse_results(rmse_results)

    print_section("MovieLens ALS Fairness Baseline")
    baseline = run_movielens_baseline()
    print_movielens_results(baseline["results"])

    print_section("MovieLens SMT Fairness Checks")
    smt_checks = run_movielens_smt_checks(eps=0.01)
    print(smt_checks["checks"])

    print_section("MovieLens Weighted Mitigation")
    mitigation = run_movielens_weighted_mitigation(
        target_group="Under18",
        demographic="age_group",
    )
    print(mitigation["sweep"])

    print_section("MovieLens Alpha Reranking")
    alpha_results = run_movielens_alpha_reranking(
        demographic="age_group",
        target_group="Under18",
    )

    print("\nLong-tail alpha sweep:")
    print(alpha_results["longtail_sweep"])

    print("\nGroup-specific alpha sweep:")
    print(alpha_results["group_sweep"])

    print("\nSelected long-tail alpha:")
    print(alpha_results["selected_longtail"])

    print("\nSelected group-specific alpha:")
    print(alpha_results["selected_group"])

    print_section("MovieLens Counterexample-Guided Repair")

    cegar = run_movielens_cegar_repair(
        demographic="intersection_group",
        eps=0.01,
    )

    print("\nCEGAR history:")
    print(cegar["history"])

    print("\nCEGAR message:")
    print(cegar["message"])

    print("\nBest alpha map:")
    print(cegar["best_alpha_map"])

    print("\nBest CEGAR group table:")
    print(cegar["best_result"]["group_table"])

    print_section("Repair Method Comparison")

    comparison = summarize_repair_results(
        baseline_result=baseline["results"]["gender_age"],
        weighted_result=mitigation,
        alpha_result=alpha_results,
        cegar_result=cegar,
    )

    print(comparison)

    print_section("Epsilon Certification Table")

    cert_table = certification_table(
        comparison,
        eps_values=[0.01, 0.015, 0.02, 0.03],
    )

    print(cert_table)

    return {
        "rmse_results": rmse_results,
        "baseline": baseline,
        "smt_checks": smt_checks,
        "mitigation": mitigation,
        "alpha_results": alpha_results,
        "cegar": cegar,
        "comparison": comparison,
        "cert_table": cert_table,
    }


def run_lastfm_if_available():
    if not os.path.exists("data/lastfm"):
        print_section("Last.fm Skipped")
        print("data/lastfm not found. Skipping Last.fm pipeline.")
        return None

    print_section("Last.fm Pipeline")
    lastfm_output = run_lastfm_pipeline()
    print_lastfm_results(lastfm_output["results"])

    return lastfm_output


if __name__ == "__main__":
    outputs = {}

    outputs["movielens"] = run_movielens_all()
    outputs["lastfm"] = run_lastfm_if_available()