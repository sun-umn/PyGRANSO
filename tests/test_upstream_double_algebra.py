"""Numerical sections from pinned tests/lin_alg; these exercise tensor operations.

Fixture values are the generated double arrays from tests/lin_alg/generate_problem.py
at 1572ae068e9ce9ca723cf8223548ade1ff7acc29. No adjoint or indirect-solver
implementation is invoked by these controls.
"""

import json
from pathlib import Path

import pytest
import torch


@pytest.fixture
def vectors(available_device):
    path = Path(__file__).parent / "fixtures" / "upstream_double_algebra.json"
    return {
        key: torch.tensor(values, device=available_device, dtype=torch.float64)
        for key, values in json.loads(path.read_text()).items()
    }


@pytest.mark.parametrize(
    "d,reference", [("ones", "AtA"), ("vn", "AtRnA"), ("v1", "AtRA")]
)
def test_atda_diagonal(vectors, d, reference):
    a = vectors["A"]
    diagonal = torch.ones_like(vectors["v1"]) if d == "ones" else vectors[d]
    result = torch.diagonal(a.T @ (diagonal[:, None] * a))
    torch.testing.assert_close(result, vectors[reference], rtol=0, atol=1e-4)


def test_concat(vectors):
    a, b = vectors["v1"], vectors["v2"]
    result = torch.cat((a, b))
    assert torch.equal(result[: len(a)], a)
    assert torch.equal(result[len(a) :], b)


@pytest.mark.parametrize("size", [0, 5, 10])
def test_subvector_selection(vectors, size):
    v = vectors["v1"]
    mask = (
        vectors["mask"].bool()
        if size == 5
        else torch.full_like(v, size == 10, dtype=torch.bool)
    )
    result = v[mask]
    expected = vectors["subvector"] if size == 5 else v[:size]
    assert torch.equal(result, expected)


@pytest.mark.parametrize("size", [0, 5, 10])
def test_subvector_assignment(vectors, size):
    result = vectors["v1"].clone()
    start = 2 if size != 10 else 0
    source = vectors["subvector"] if size == 5 else vectors["v2"]
    result[start : start + size] = source[:size]
    reference = {0: "v1", 5: "assigned", 10: "v2"}[size]
    assert torch.equal(result, vectors[reference])


def test_set_values(vectors):
    v = vectors["v1"]
    a, b, c = (vectors[k] for k in ("sc1", "sc2", "sc3"))
    signs = vectors["sca_cond"]
    for actual, key in [
        (torch.ones_like(v) * a, "same"),
        (torch.zeros_like(v), "zero"),
        (torch.where(signs < 0, a, torch.where(signs > 0, c, b)), "sca_cond_res"),
        (torch.where(v < a, b, v), "sca_lt"),
        (torch.where(v > a, b, v), "sca_gt"),
    ]:
        torch.testing.assert_close(actual, vectors[key], rtol=0, atol=1e-4)


def close(left, right):
    return left.shape == right.shape and torch.allclose(left, right, atol=1e-4, rtol=0)


def test_vector_equality(vectors):
    a, b = vectors["v1"], vectors["v2"]
    assert close(a, a) and close(b[:-2], b[:-2].clone())
    assert not close(a, b) and not close(b, b[:-2])


def test_matrix_equality(vectors):
    a, b = vectors["mat_vec_A"], vectors["submat_A5"]
    assert close(a, a) and close(a, b)
    assert not close(a, vectors["submat_A4"]) and not close(a, vectors["mat_ops_A"])
    assert not close(3.0 * a, b)
    for index in (0, -1):
        changed = a.clone()
        changed[index] *= 4.0
        assert not close(changed, b)
