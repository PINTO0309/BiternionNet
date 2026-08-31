"""Command-line interface for the TownCentre synthetic pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from .generate import (
    PipelineError,
    build_usage_report,
    collect_results,
    create_edit_cycle,
    create_plan,
    load_config,
    load_state,
    pending_request_count,
    prepare_resume,
    read_jsonl,
    refresh_status,
    sha256_file,
    submit_pending,
)
from .materialize import materialize_run, render_deim_crop_margin_sheet
from .models import install_model_assets
from .profiles import finalize_test_profiles, plan_profile_candidates
from .qa import (
    approve_human_review,
    config_from_recorded_qa_policy,
    prepare_human_review,
    promote_direct_production_labels,
    run_auto_qa,
)
from .quotas import top_up_plan

app = typer.Typer(help="Plan, collect, QA, and materialize TownCentre synthetic heads.")


def _print(value: object) -> None:
    typer.echo(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


@app.command("plan")
def plan_command(
    stage: str = typer.Option(..., help="validation, pilot, floor_120, or uniform_200"),
    batch_id: str = typer.Option(...),
    config: Path = typer.Option(
        Path("configs/synthetic_towncentre_batch.yaml"), exists=True, readable=True
    ),
    output_root: Path = typer.Option(Path("data/synthetic")),
    seed: int = typer.Option(20260831),
    approved_batch_dir: Optional[Path] = typer.Option(None, exists=True),
    bin_counts: Optional[str] = typer.Option(
        None, help="Optional comma-separated 19-bin top-up request counts."
    ),
    direct_production: bool = typer.Option(
        False,
        "--direct-production",
        help=(
            "Plan uniform_200 directly without Validation/Pilot parent approval. "
            "This is an explicit operator waiver of intermediate and human gates."
        ),
    ),
    single_batch: bool = typer.Option(
        False,
        "--single-batch",
        help="Write all planned requests to one Batch input file.",
    ),
    compact_prompts: bool = typer.Option(
        False,
        "--compact-prompts",
        help=(
            "Use the compact production prompt profile to fit an account's queued-token limit. "
            "Only valid with --direct-production."
        ),
    ),
) -> None:
    counts = [int(value) for value in bin_counts.split(",")] if bin_counts else None
    run_dir = create_plan(
        config,
        stage,
        batch_id,
        output_root,
        seed,
        approved_batch_dir,
        bin_counts=counts,
        direct_production=direct_production,
        single_batch=single_batch,
        compact_prompts=compact_prompts,
    )
    state = load_state(run_dir)
    _print(
        {
            "batch_dir": str(run_dir),
            "stage": stage,
            "requests": state["request_count"],
            "shards": len(state["shards"]),
            "quality": state["api_request"]["quality"],
            "model": state["api_request"]["model"],
            "documented_reference_projected_cost_usd": state[
                "reference_projected_cost_usd"
            ],
            "planning_projected_cost_usd": state["planning_projected_cost_usd"],
            "planning_cost_basis": state["planning_cost_basis"],
            "direct_production": state["direct_production"],
            "single_batch": state["single_batch"],
            "prompt_profile": state["prompt_profile"],
            "paid_request_submitted": False,
        }
    )


@app.command("edit-plan")
def edit_plan_command(
    parent_batch_dir: Path = typer.Option(..., exists=True, file_okay=False),
    batch_id: str = typer.Option(...),
    planning_cost_per_request_usd: float = typer.Option(
        ...,
        min=0.001,
        help="Observed per-edit cost used for the explicit submission spend guard.",
    ),
    max_edit_rounds: int = typer.Option(2, min=1, max=8),
    include_pitch_calibration_tail: bool = typer.Option(
        False,
        help="Also edit the two Validation records controlling a failed pitch-calibration tail.",
    ),
    output_root: Path = typer.Option(Path("data/synthetic")),
    edit_token_evidence_run: Optional[Path] = typer.Option(
        None,
        "--edit-token-evidence-run",
        exists=True,
        file_okay=False,
        help=(
            "Completed image-edit run whose observed input-token mean determines "
            "the minimum number of future edit Batches."
        ),
    ),
    only_reason: Optional[list[str]] = typer.Option(
        None,
        "--only-reason",
        help=(
            "Edit only records carrying this QA reason. Repeat the option to "
            "select more than one reason."
        ),
    ),
) -> None:
    run_dir = create_edit_cycle(
        parent_batch_dir,
        batch_id,
        output_root,
        max_edit_rounds=max_edit_rounds,
        planning_cost_per_request_usd=planning_cost_per_request_usd,
        include_pitch_calibration_tail=include_pitch_calibration_tail,
        edit_token_evidence_run_dir=edit_token_evidence_run,
        only_edit_reasons=set(only_reason) if only_reason else None,
    )
    state = load_state(run_dir)
    _print(
        {
            "batch_dir": str(run_dir),
            "edit_round": state["edit_round"],
            "target_images": state["target_count"],
            "edit_requests": state["request_count"],
            "planning_projected_cost_usd": state["planning_projected_cost_usd"],
            "planning_cost_basis": state["planning_cost_basis"],
            "forced_edit_policy": state.get("forced_edit_policy"),
            "edit_reason_filter": state.get("edit_reason_filter"),
            "paid_request_submitted": False,
        }
    )


@app.command("submit")
def submit_command(
    batch_dir: Path = typer.Option(..., exists=True),
    approve_requests: int = typer.Option(
        ..., min=1, help="Must exactly match pending requests."
    ),
    spend_cap_usd: float = typer.Option(..., min=0.001),
) -> None:
    ids = submit_pending(
        batch_dir,
        approved_request_count=approve_requests,
        spend_cap_usd=spend_cap_usd,
    )
    _print({"batch_ids": ids})


@app.command("status")
def status_command(batch_dir: Path = typer.Option(..., exists=True)) -> None:
    _print(refresh_status(batch_dir))


@app.command("collect")
def collect_command(batch_dir: Path = typer.Option(..., exists=True)) -> None:
    _print(collect_results(batch_dir))


@app.command("resume-plan")
def resume_command(batch_dir: Path = typer.Option(..., exists=True)) -> None:
    created = prepare_resume(batch_dir)
    state = load_state(batch_dir)
    _print(
        {
            "retry_requests": created,
            "pending_requests": pending_request_count(state),
            "submitted": False,
        }
    )


@app.command("usage-report")
def usage_report_command(
    batch_dir: Path = typer.Option(..., exists=True),
    actual_cost_usd: Optional[float] = typer.Option(
        None,
        min=0.001,
        help="Actual account charge for this run; omit when it has not been verified.",
    ),
) -> None:
    _print(build_usage_report(batch_dir, actual_cost_usd=actual_cost_usd))


@app.command("qa")
def qa_command(
    batch_dir: Path = typer.Option(..., exists=True),
    detector_model: Optional[Path] = typer.Option(
        Path("data/models/deimv2_wholebody49_boxes_only.onnx")
    ),
    pose_model: Optional[Path] = typer.Option(
        Path("data/models/sixdrepnet360_1x3x224x224_full.onnx")
    ),
    landmark_model: Optional[Path] = typer.Option(
        Path("data/models/hrffa_vitl_ibug68_1x3x320x320.onnx")
    ),
    calibration: Optional[Path] = typer.Option(None),
    rear_policy: Optional[Path] = typer.Option(None),
    policy: Optional[Path] = typer.Option(
        Path("configs/synthetic_qa_policy_v2.yaml"),
        exists=True,
        readable=True,
        help="Hash-bound QA policy override. Defaults to the current repository policy.",
    ),
    cpu: bool = typer.Option(
        False,
        help="Force CPU; otherwise prefer TensorRT, then CUDA, then CPU. All ONNX runs stay batch-1.",
    ),
) -> None:
    _print(
        run_auto_qa(
            batch_dir,
            detector_model=detector_model,
            pose_model=pose_model,
            landmark_model=landmark_model,
            calibration_path=calibration,
            rear_policy_path=rear_policy,
            qa_policy_path=policy,
            cpu=cpu,
        )
    )


@app.command("review-prepare")
def review_prepare_command(batch_dir: Path = typer.Option(..., exists=True)) -> None:
    _print(prepare_human_review(batch_dir))


@app.command("promote-direct-labels")
def promote_direct_labels_command(
    batch_dir: Path = typer.Option(..., exists=True, file_okay=False),
) -> None:
    """Accept every quality-passed direct-production image as labelled data."""
    _print(promote_direct_production_labels(batch_dir))


@app.command("margin-sheet")
def margin_sheet_command(
    batch_dir: Path = typer.Option(..., exists=True),
    output: Path = typer.Option(...),
    anchor_manifest: Path = typer.Option(
        Path("data/towncentre/manifest.jsonl"), exists=True, readable=True
    ),
) -> None:
    state = load_state(batch_dir)
    config = load_config(Path(state["config_path"]))
    qa_report_path = batch_dir / "qa_report.json"
    if not qa_report_path.exists():
        raise PipelineError("run auto QA before rendering the fixed crop margin")
    qa_report = json.loads(qa_report_path.read_text(encoding="utf-8"))
    config_from_recorded_qa_policy(config, qa_report)
    result = render_deim_crop_margin_sheet(
        batch_dir,
        output=output,
        anchor_manifest=anchor_manifest,
    )
    _print({"output": str(result)})


@app.command("approve")
def approve_command(
    batch_dir: Path = typer.Option(..., exists=True),
    reviewer: str = typer.Option(...),
    sign_calibration_approved: bool = typer.Option(False, "--approve-sign-calibration"),
    evaluation_protocol: Path = typer.Option(..., exists=True, readable=True),
    usage_report: Path = typer.Option(..., exists=True, readable=True),
    account_verified_snapshot: Optional[str] = typer.Option(None),
) -> None:
    _print(
        approve_human_review(
            batch_dir,
            reviewer=reviewer,
            sign_calibration_approved=sign_calibration_approved,
            evaluation_protocol=evaluation_protocol,
            usage_report=usage_report,
            account_verified_snapshot=account_verified_snapshot,
        )
    )


@app.command("materialize")
def materialize_command(
    batch_dir: Path = typer.Option(..., exists=True),
    output_root: Path = typer.Option(Path("data/synthetic")),
    anchor_manifest: Path = typer.Option(
        Path("data/towncentre/manifest.jsonl"), exists=True
    ),
    neighbour_manifest: Path = typer.Option(
        Path("data/towncentre/manifest_nb3.jsonl"), exists=True
    ),
    seed: int = typer.Option(20260831),
) -> None:
    _print(
        materialize_run(
            batch_dir,
            output_root=output_root,
            anchor_manifest=anchor_manifest,
            neighbour_manifest=neighbour_manifest,
            seed=seed,
        )
    )


@app.command("top-up-plan")
def top_up_command(
    annotations: Path = typer.Option(..., exists=True, readable=True),
    target: str = typer.Option("floor_120"),
    config: Path = typer.Option(
        Path("configs/synthetic_towncentre_batch.yaml"), exists=True
    ),
) -> None:
    _print(top_up_plan(load_config(config), read_jsonl(annotations), target))


@app.command("profiles-plan")
def profiles_plan_command(
    output_csv: Path = typer.Option(Path("data/towncentre/test_profiles_review.csv")),
    image_root: Path = typer.Option(Path("data/TownCentreHeadImages"), exists=True),
    existing_manifest: Path = typer.Option(
        Path("data/towncentre/manifest.jsonl"), exists=True
    ),
    candidate_count: int = typer.Option(600, min=200),
    seed: int = typer.Option(20260831),
) -> None:
    _print(
        plan_profile_candidates(
            image_root,
            existing_manifest,
            output_csv,
            candidate_count=candidate_count,
            seed=seed,
        )
    )


@app.command("profiles-finalize")
def profiles_finalize_command(
    reviewed_csv: Path = typer.Option(..., exists=True, readable=True),
    output_jsonl: Path = typer.Option(Path("data/towncentre/test_profiles.jsonl")),
    protocol_output: Path = typer.Option(
        Path("data/towncentre/test_profiles_protocol.json")
    ),
    existing_manifest: Path = typer.Option(
        Path("data/towncentre/manifest.jsonl"), exists=True
    ),
) -> None:
    _print(
        finalize_test_profiles(
            reviewed_csv,
            existing_manifest,
            output_jsonl,
            protocol_output,
        )
    )


@app.command("verify-model")
def verify_model_command(
    path: Path = typer.Option(..., exists=True, readable=True),
    expected_sha256: str = typer.Option(...),
) -> None:
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise typer.BadParameter(f"SHA-256 mismatch: {actual}")
    _print({"path": str(path), "sha256": actual, "verified": True})


@app.command("install-models")
def install_models_command(
    source_repo: Path = typer.Option(..., exists=True, file_okay=False, readable=True),
    repository_root: Path = typer.Option(
        Path("."), exists=True, file_okay=False, writable=True
    ),
    config: Path = typer.Option(
        Path("configs/synthetic_towncentre_batch.yaml"), exists=True, readable=True
    ),
) -> None:
    loaded = load_config(config)
    _print(
        install_model_assets(
            loaded["models"],
            source_repo=source_repo.resolve(),
            repository_root=repository_root.resolve(),
        )
    )


def main() -> int:
    try:
        app()
        return 0
    except PipelineError as exc:
        typer.echo(f"error: {exc}", err=True)
        return 2
