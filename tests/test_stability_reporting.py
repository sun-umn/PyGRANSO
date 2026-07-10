import csv
import json
from argparse import Namespace

import pytest
import torch

import torch_osqp_stability as stability
from torch_osqp_stability import (
    CSV_COLUMNS,
    environment_manifest,
    source_tree_fingerprint,
)


def test_source_tree_fingerprint_is_deterministic_and_nonempty():
    first = source_tree_fingerprint()
    second = source_tree_fingerprint()
    assert first == second
    digest, file_count = first
    assert len(digest) == 64
    assert file_count > 20


def test_manifest_records_exact_source_provenance():
    args = Namespace(
        device="cpu",
        dtype="float64",
        seeds=1,
        stress_seeds=0,
        time_limit_seconds=7200.0,
    )
    manifest = environment_manifest(args)
    assert len(manifest["source_tree_sha256"]) == 64
    assert manifest["source_file_count"] > 20
    assert isinstance(manifest["git_status_entries"], (list, type(None)))


def test_time_limit_exit_writes_partial_artifacts(tmp_path, monkeypatch):
    def fake_evaluate(family, seed, device, dtype):
        row = {column: None for column in CSV_COLUMNS}
        row.update(
            {
                "test_family": family,
                "case_name": f"{family}_seed_{seed}",
                "seed": seed,
                "device": str(device),
                "dtype": str(dtype),
                "overall_status": "passed",
                "release_gate": "passed",
                "failure_reason": "",
            }
        )
        problem = (
            torch.eye(1, dtype=dtype, device=device),
            torch.zeros((1, 1), dtype=dtype, device=device),
            None,
            None,
            torch.zeros((1, 1), dtype=dtype, device=device),
            torch.ones((1, 1), dtype=dtype, device=device),
        )
        return row, problem

    ticks = iter([0.0, 0.2, 0.4])
    monkeypatch.setattr(stability, "evaluate", fake_evaluate)
    monkeypatch.setattr(stability.time, "perf_counter", lambda: next(ticks))

    args = Namespace(
        output=str(tmp_path),
        device="cpu",
        dtype="float64",
        seeds=1,
        stress_seeds=1,
        time_limit_seconds=0.1,
    )
    with pytest.raises(SystemExit, match="1/3 cases"):
        stability.run(args)

    with (tmp_path / "torch_osqp_stability_results.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    manifest = json.loads(
        (tmp_path / "torch_osqp_stability_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    summary = (tmp_path / "torch_osqp_stability_summary.md").read_text(
        encoding="utf-8"
    )

    assert len(rows) == 1
    assert rows[0]["case_name"] == "randomized_bounds_seed_0"
    assert manifest["timed_out"] is True
    assert manifest["partial_results"] is True
    assert manifest["completed_cases"] == 1
    assert manifest["planned_cases"] == 3
    assert manifest["timeout_after_case"] == {
        "completed_cases": 1,
        "test_family": "randomized_bounds",
        "seed": 0,
    }
    assert "- Cases: 1/3" in summary
    assert "- Timed out: True" in summary
