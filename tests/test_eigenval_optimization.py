"""
Test eigenvalue optimization consistency between CPU and CUDA devices.
"""

import time

import pytest
import scipy.io
import torch
from torch import linalg as LA

from pygranso.pygranso import pygranso
from pygranso.pygransoStruct import pygransoStruct


def load_eigenval_data(device):
    """Load the eigenvalue optimization data."""
    file = "examples/data/spec_radius_opt_data.mat"
    mat = scipy.io.loadmat(file)
    mat_struct = mat["sys"]
    mat_struct = mat_struct[0, 0]

    # Convert to torch tensors with specified device and dtype
    A = torch.from_numpy(mat_struct["A"]).to(device=device, dtype=torch.double)
    B = torch.from_numpy(mat_struct["B"]).to(device=device, dtype=torch.double)
    C = torch.from_numpy(mat_struct["C"]).to(device=device, dtype=torch.double)

    p = B.shape[1]
    m = C.shape[0]
    stability_margin = 1

    return A, B, C, p, m, stability_margin


def create_eigenval_function(A, B, C, stability_margin):
    """Create the user function for eigenvalue optimization."""

    def user_fn(X_struct):
        # user defined variable, matrix form. torch tensor
        X = X_struct.X

        # objective function
        M = A + B @ X @ C
        [D, _] = LA.eig(M)
        f = torch.max(D.imag)

        # inequality constraint, matrix form
        ci = pygransoStruct()
        ci.c1 = torch.max(D.real) + stability_margin

        # equality constraint
        ce = None

        return [f, ci, ce]

    return user_fn


def run_eigenval_optimization(device, max_iterations=50):
    """Run eigenvalue optimization on specified device."""
    # Load data
    A, B, C, p, m, stability_margin = load_eigenval_data(device)

    # Set up variables and function
    var_in = {"X": [p, m]}
    user_fn = create_eigenval_function(A, B, C, stability_margin)

    def comb_fn(X_struct):
        return user_fn(X_struct)

    # Set up options
    opts = pygransoStruct()
    opts.torch_device = device
    opts.maxit = max_iterations
    opts.x0 = torch.zeros(p * m, 1).to(device=device, dtype=torch.double)
    opts.print_frequency = 10
    opts.quadprog_info_msg = False  # Suppress QP solver notice

    # Run optimization
    start_time = time.time()
    soln = pygranso(var_spec=var_in, combined_fn=comb_fn, user_opts=opts)
    end_time = time.time()

    return soln, end_time - start_time


class TestEigenvalOptimization:
    """Test class for eigenvalue optimization."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cpu_cuda_consistency(self, cuda_available, tolerance, max_iterations):
        """Test that CPU and CUDA produce consistent results."""
        if not cuda_available:
            pytest.skip("CUDA not available")

        # Run on CPU
        cpu_soln, cpu_time = run_eigenval_optimization(
            torch.device("cpu"), max_iterations
        )

        # Run on CUDA
        cuda_soln, cuda_time = run_eigenval_optimization(
            torch.device("cuda"), max_iterations
        )

        # Compare final x values
        cpu_x = cpu_soln.final.x.cpu()
        cuda_x = cuda_soln.final.x.cpu()

        x_diff = torch.abs(cpu_x - cuda_x)
        max_diff = torch.max(x_diff).item()

        # Compare objective values
        obj_diff = abs(cpu_soln.final.f - cuda_soln.final.f)

        # Assertions
        assert max_diff < tolerance, (
            f"Maximum difference in x ({max_diff:.2e}) exceeds tolerance "
            f"({tolerance:.2e})"
        )
        assert obj_diff < tolerance, (
            f"Objective value difference ({obj_diff:.2e}) exceeds tolerance "
            f"({tolerance:.2e})"
        )

        # Log performance info
        print(f"\nCPU time: {cpu_time:.2f}s")
        print(f"CUDA time: {cuda_time:.2f}s")
        if cuda_time > 0:
            speedup = cpu_time / cuda_time
            print(f"CUDA speedup: {speedup:.2f}x")

    def test_cpu_only(self, max_iterations):
        """Test that CPU optimization runs successfully."""
        soln, cpu_time = run_eigenval_optimization(torch.device("cpu"), max_iterations)

        # Basic assertions
        assert hasattr(soln, "final"), "Solution should have 'final' attribute"
        assert hasattr(soln.final, "x"), "Solution should have 'final.x' attribute"
        assert hasattr(soln.final, "f"), "Solution should have 'final.f' attribute"
        assert cpu_time > 0, "CPU optimization should take some time"

        print(f"CPU optimization completed in {cpu_time:.2f}s")
        print(f"Final objective value: {soln.final.f:.6f}")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_only(self, max_iterations):
        """Test that CUDA optimization runs successfully."""
        soln, cuda_time = run_eigenval_optimization(
            torch.device("cuda"), max_iterations
        )

        # Basic assertions
        assert hasattr(soln, "final"), "Solution should have 'final' attribute"
        assert hasattr(soln.final, "x"), "Solution should have 'final.x' attribute"
        assert hasattr(soln.final, "f"), "Solution should have 'final.f' attribute"
        assert cuda_time > 0, "CUDA optimization should take some time"

        print(f"CUDA optimization completed in {cuda_time:.2f}s")
        print(f"Final objective value: {soln.final.f:.6f}")

    def test_solution_structure(self, max_iterations):
        """Test that the solution has the expected structure."""
        soln, _ = run_eigenval_optimization(torch.device("cpu"), max_iterations)

        # Check solution structure
        assert hasattr(soln, "final"), "Solution should have 'final' attribute"
        assert hasattr(soln.final, "x"), "Solution should have 'final.x' attribute"
        assert hasattr(soln.final, "f"), "Solution should have 'final.f' attribute"

        # Check data types
        assert torch.is_tensor(soln.final.x), "final.x should be a torch tensor"
        assert isinstance(soln.final.f, (int, float)), "final.f should be a scalar"

        # Check shapes
        assert soln.final.x.dim() == 2, "final.x should be 2-dimensional"
        assert soln.final.x.shape[1] == 1, "final.x should have shape (n, 1)"
