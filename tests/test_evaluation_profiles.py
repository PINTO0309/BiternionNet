from biternionnet.evaluation import paired_cluster_bootstrap, summarize_angle_errors


def test_per_bin_metrics_include_counts_and_macro():
    metrics = summarize_angle_errors([90, 91, 270, 0], [10, 20, 30, 5])
    assert metrics["bin_090_count"] == 2
    assert metrics["bin_090_maad_deg"] == 15
    assert metrics["bin_270_count"] == 1
    assert metrics["maad_deg"] == 16.25


def test_paired_person_cluster_bootstrap_detects_consistent_improvement():
    baseline_runs = []
    candidate_runs = []
    for seed in range(3):
        baseline = []
        candidate = []
        for index in range(20):
            target = 90.0 if index < 10 else 270.0
            common = {
                "record_id": f"r{index}",
                "person_id": f"p{index // 2}",
                "target_deg": target,
            }
            baseline.append({**common, "prediction_deg": target + 10 + seed, "error_deg": 10 + seed})
            candidate.append({**common, "prediction_deg": target + 5 + seed, "error_deg": 5 + seed})
        baseline_runs.append(baseline)
        candidate_runs.append(candidate)
    result = paired_cluster_bootstrap(baseline_runs, candidate_runs, resamples=500, seed=3)
    assert result["overall"]["mean_improvement_deg"] == 5
    assert result["overall"]["ci_excludes_zero_positive"] is True
    assert result["bins"]["90"]["n"] == 10
    assert result["profile_sides"]["60_120"]["n"] == 10
    assert result["profile_sides"]["240_300"]["n"] == 10
    assert result["profile_sides"]["combined"]["ci_excludes_zero_positive"] is True
