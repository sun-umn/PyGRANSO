import pytest
import torch

from pygranso.private import torchOSQP as runtime
from pygranso.private.osqpWorkspace import TorchOSQPWorkspace
from pygranso.private.torchQDLDLSolve import QDLDLSolver
from pygranso.private.torchOSQP import (
    _residuals,
    admm_vector_update,
    build_kkt_matrix,
    build_kkt_rhs,
    recover_z_tilde,
    solve_torch_osqp_direct,
)


def test_residuals_match_direct_matrix_vector_calculations(
    floating_dtype, available_device
):
    dtype = floating_dtype
    device = available_device
    P = torch.tensor([[2.0, 0.5], [0.5, 1.0]], dtype=dtype, device=device)
    q = torch.tensor([-1.0, 2.0], dtype=dtype, device=device)
    A = torch.tensor([[1.0, -1.0], [0.0, 1.0]], dtype=dtype, device=device)
    x = torch.tensor([0.3, -0.2], dtype=dtype, device=device)
    z = torch.tensor([0.1, 0.4], dtype=dtype, device=device)
    y = torch.tensor([0.2, -0.5], dtype=dtype, device=device)

    assert torch.allclose(P @ x, torch.tensor([0.5, -0.05], dtype=dtype, device=device))
    assert torch.allclose(A @ x, torch.tensor([0.5, -0.2], dtype=dtype, device=device))
    assert torch.allclose(A.T @ y, torch.tensor([0.2, -0.7], dtype=dtype, device=device))

    residuals = _residuals(P, q, A, x, z, y, 0.01, 0.1)
    expected = torch.tensor([0.6, 1.25, 0.06, 0.21], dtype=dtype, device=device)
    assert torch.allclose(torch.stack(residuals), expected)

    empty = torch.empty((0, 2), dtype=dtype, device=device)
    empty_vector = torch.empty(0, dtype=dtype, device=device)
    unconstrained = _residuals(
        P, q, empty, x, empty_vector, empty_vector, 0.01, 0.1
    )
    assert unconstrained[0].item() == 0.0
    assert torch.allclose(unconstrained[1], torch.tensor(1.95, dtype=dtype, device=device))
    assert torch.allclose(unconstrained[2], torch.tensor(0.01, dtype=dtype, device=device))
    assert torch.allclose(unconstrained[3], torch.tensor(0.21, dtype=dtype, device=device))


def test_pinned_matrix_vector_and_empty_shape_identities(
    floating_dtype, available_device
):
    dtype = floating_dtype
    device = available_device
    A = torch.tensor(
        [
            [0.02388881397881171, 0.11670985678100221, 0.1637416544531679, 0.08204500601562204],
            [0.8074227810412945, 0.0320813486181537, 0.09208896568911618, 0.6223133637152601],
            [0.11317416004642233, 0.6317521922027585, 0.6703066467384771, 0.6163837910949103],
            [0.7867947609394985, 0.46593324116426427, 0.06556315241841992, 0.9153082686153085],
            [0.9268052023585862, 0.8744412014424732, 0.6928622323352233, 0.41875301731404],
        ],
        dtype=dtype,
        device=device,
    )
    P_upper = torch.tensor(
        [
            [0.11174393006374239, 1.6155131493997872, 1.0185760041285183, 0.9470061687356901],
            [0.0, 0.7985820269731905, 1.2230586790284623, 0.2500183399320113],
            [0.0, 0.0, 1.1514103206781972, 0.5849662484122946],
            [0.0, 0.0, 0.0, 1.2406411113098743],
        ],
        dtype=dtype,
        device=device,
    )
    P = P_upper + torch.triu(P_upper, diagonal=1).T
    x = A.new_tensor(
        [-0.6326942125201124, -0.8083277018861249, -0.36626701885025204, -0.11471739196713679]
    )
    y = A.new_tensor(
        [-1.4013182000990516, -0.03509478412884679, -1.6674865904394212, 1.3921384005454986, -0.08099695140769417]
    )
    expected_Ax = A.new_tensor(
        [-0.17883928141982014, -0.6419032804273073, -0.8984885919346091, -1.0034426355544523, -1.5950301727983345]
    )
    expected_Ax_cumulative = A.new_tensor(
        [-1.5801574815188717, -0.6769980645561541, -2.5659751823740304, 0.3886957649910463, -1.6760271242060287]
    )
    expected_ATy = A.new_tensor(
        [0.7697302518417538, -0.6402953578080189, -1.31526009421207, 0.07569525174898725]
    )
    expected_ATy_cumulative = A.new_tensor(
        [0.13703603932164143, -1.4486230596941438, -1.6815271130623222, -0.03902214021814954]
    )
    expected_Px = A.new_tensor(
        [-1.8582726436154313, -2.144289302653823, -2.121908782149807, -1.157839028917209]
    )
    expected_Px_cumulative = A.new_tensor(
        [-2.4909668561355436, -2.952617004539948, -2.488175801000059, -1.2725564208843456]
    )
    tolerance = 1e-4

    torch.testing.assert_close(A @ x, expected_Ax, atol=tolerance, rtol=0.0)
    torch.testing.assert_close(y + A @ x, expected_Ax_cumulative, atol=tolerance, rtol=0.0)
    torch.testing.assert_close(A.T @ y, expected_ATy, atol=tolerance, rtol=0.0)
    torch.testing.assert_close(x + A.T @ y, expected_ATy_cumulative, atol=tolerance, rtol=0.0)
    torch.testing.assert_close(P @ x, expected_Px, atol=tolerance, rtol=0.0)
    torch.testing.assert_close(x + P @ x, expected_Px_cumulative, atol=tolerance, rtol=0.0)

    empty_2_by_2 = A.new_zeros((2, 2))
    no_rows = A.new_empty((0, 2))
    no_columns = A.new_empty((2, 0))
    vector_2 = A.new_tensor([1.0, 2.0])
    empty_vector = A.new_empty(0)
    zeros_2 = A.new_zeros(2)

    assert torch.equal(empty_2_by_2 @ vector_2, zeros_2)
    assert torch.equal(empty_2_by_2.T @ vector_2, zeros_2)
    assert torch.equal(no_rows @ vector_2, empty_vector)
    assert torch.equal(no_rows.T @ empty_vector, zeros_2)
    assert torch.equal(no_columns @ empty_vector, zeros_2)
    assert torch.equal(no_columns.T @ vector_2, empty_vector)


def test_pinned_matrix_row_selection_and_diagonal_extraction(
    floating_dtype, available_device
):
    A = torch.arange(
        1,
        21,
        dtype=floating_dtype,
        device=available_device,
    ).reshape(5, 4)
    selections = (
        (torch.tensor([True, True, False, True, True], device=available_device), 4),
        (torch.tensor([True, True, True, True, True], device=available_device), 5),
        (torch.tensor([True, False, True, False, True], device=available_device), 3),
        (torch.tensor([False, False, False, False, False], device=available_device), 0),
    )
    expected_rows = ((0, 1, 3, 4), (0, 1, 2, 3, 4), (0, 2, 4), ())

    for (mask, expected_count), rows in zip(selections, expected_rows):
        selected = A[mask]
        expected = A.new_empty((0, 4)) if not rows else A[list(rows)]
        assert selected.shape[1] == A.shape[1]
        assert selected.shape[0] == expected_count
        assert torch.equal(selected, expected)

    nonsymmetric = A.new_tensor(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
    )
    symmetric_upper = A.new_tensor(
        [[2.0, 3.0, 4.0], [0.0, 5.0, 6.0], [0.0, 0.0, 7.0]]
    )
    symmetric = symmetric_upper + torch.triu(symmetric_upper, diagonal=1).T
    assert torch.equal(torch.diagonal(nonsymmetric), A.new_tensor([1.0, 5.0, 9.0]))
    assert torch.equal(torch.diagonal(symmetric), A.new_tensor([2.0, 5.0, 7.0]))


def test_pinned_vector_math_operations_match_torch_primitives(
    floating_dtype, available_device
):
    """Translate the first 60 assertions in test_vector_math_ops.cpp."""

    v1 = torch.tensor(
        [
            -0.31389947196684775,
            0.05410227877154389,
            0.27279133916445375,
            -0.9821881249409777,
            -1.107373047165193,
            0.19958453284708083,
            -0.46674961687980204,
            0.23550561173022522,
            0.7595195224783792,
            -1.6487873663509485,
        ],
        dtype=floating_dtype,
        device=available_device,
    )
    v2 = v1.new_tensor(
        [
            0.2543881165176173,
            1.2246469675357323,
            -0.2975268443704732,
            -0.8108145832375699,
            0.7522438271795928,
            0.25344651620814146,
            0.8958830707775604,
            -0.3452157100512797,
            -1.4818182737222112,
            -0.11001076471125099,
        ]
    )
    v3 = v1.new_tensor(
        [
            -0.4458281530112322,
            0.7753238220475741,
            0.1936328483771538,
            -1.6308492324351012,
            -1.1951630801031998,
            0.8837890365872553,
            0.6797650174178466,
            -0.6402433659084887,
            -0.001048796567280681,
            0.4455735537761861,
        ]
    )
    sc1 = 0.4684043358472779
    sc2 = 0.8762421961143501
    sc3 = 0.256485627221562
    tolerance = 1e-4

    def assert_reference(actual, expected):
        expected_tensor = (
            expected.to(device=actual.device, dtype=actual.dtype)
            if torch.is_tensor(expected)
            else actual.new_tensor(expected)
        )
        torch.testing.assert_close(
            actual,
            expected_tensor,
            atol=tolerance,
            rtol=0.0,
        )

    expected_add = [
        -0.059511355449230474,
        1.2787492463072763,
        -0.02473550520601947,
        -1.7930027081785478,
        -0.3551292199856002,
        0.4530310490552223,
        0.4291334538977583,
        -0.1097100983210545,
        -0.722298751243832,
        -1.7587981310621994,
    ]
    expected_sub = [
        -0.568287588484465,
        -1.1705446887641884,
        0.5703181835349269,
        -0.1713735417034078,
        -1.8596168743447858,
        -0.05386198336106063,
        -1.3626326876573625,
        0.5807213217815049,
        2.2413377962005905,
        -1.5387766016396975,
    ]
    assert_reference(v1 + v2, expected_add)
    accumulated = v1.clone()
    accumulated.add_(v2)
    assert_reference(accumulated, expected_add)
    assert_reference(v1 - v2, expected_sub)
    accumulated = v1.clone()
    accumulated.sub_(v2)
    assert_reference(accumulated, expected_sub)

    expected_scaled = [
        0.07587372819334756,
        1.0984290902540985,
        -0.1329289294679404,
        -1.1705311273976917,
        0.14044944644872148,
        0.3155667925133792,
        0.5663830050982421,
        -0.192180722257705,
        -0.9426694609191986,
        -0.8686952253558041,
    ]
    expected_scaled_increment = [
        -0.09099387008405763,
        1.1271896270698332,
        0.012085763650297843,
        -1.6926580759986076,
        -0.44822526402388296,
        0.4216650648068339,
        0.31826093251999515,
        -0.06698696017828282,
        -0.5389121759303462,
        -1.745183440417754,
    ]
    expected_scaled3 = [
        -0.03847478526476883,
        1.2972885070517892,
        -0.0832648869012285,
        -1.5888205156826116,
        -0.16609270578360164,
        0.5422459778940014,
        0.7407329619539345,
        -0.3563939435371878,
        -0.9429384621645853,
        -0.7544120129421786,
    ]
    expected_scaled3_increment = [
        -0.20534238354217402,
        1.3260490438675239,
        0.06174980621700975,
        -2.1109474642835275,
        -0.7547674162562061,
        0.6483442501874561,
        0.49261088937568753,
        -0.23120018145776558,
        -0.539181177175733,
        -1.6309002280041285,
    ]
    assert_reference(sc1 * v1 + sc2 * v2, expected_scaled)
    assert_reference(v1 + sc2 * v2, expected_scaled_increment)
    accumulated = v1.clone()
    accumulated.copy_(sc1 * accumulated + sc2 * v2)
    assert_reference(accumulated, expected_scaled)
    assert_reference(sc1 * v1 + sc2 * v2 + sc3 * v3, expected_scaled3)
    assert_reference(v1 + sc2 * v2 + sc3 * v3, expected_scaled3_increment)
    accumulated = v1.clone()
    accumulated.copy_(sc1 * accumulated + sc2 * v2 + sc3 * v3)
    assert_reference(accumulated, expected_scaled3)

    expected_scalar_product = [
        -0.14703187368944257,
        0.025341741955809297,
        0.1277766460462155,
        -0.4600611763400619,
        -0.5186983366925886,
        0.09348626055362612,
        -0.2186275443015551,
        0.11031184965080305,
        0.35576223748952684,
        -0.7722991512889985,
    ]
    expected_product = [
        -0.07985229544952101,
        0.06625619163434404,
        -0.08116274631319541,
        0.7963724551849092,
        -0.8330145391150725,
        0.05058400453912201,
        -0.4181530800545269,
        -0.08130023697451069,
        -1.12546990765723,
        0.18138435901851738,
    ]
    multiplied = v1.clone()
    multiplied.mul_(1.0)
    assert_reference(multiplied, v1)
    multiplied = v1.clone()
    multiplied.mul_(sc1)
    assert_reference(multiplied, expected_scalar_product)
    assert_reference(v1 * v2, expected_product)
    multiplied = v1.clone()
    multiplied.mul_(v2)
    assert_reference(multiplied, expected_product)

    assert_reference(
        torch.sqrt(v1 + 4.0),
        [
            1.9199220109247022,
            2.013480141141587,
            2.067073133482329,
            1.737185043413344,
            1.7007724576893897,
            2.049288787078844,
            1.879694225963414,
            2.0580344048946864,
            2.181632306892795,
            1.5333664381513805,
        ],
    )
    ones = torch.ones_like(v1)
    assert_reference(torch.sqrt(ones), ones)
    assert_reference(
        torch.reciprocal(v1),
        [
            -3.185733297778896,
            18.483509802288935,
            3.665805531300774,
            -1.018134891480278,
            -0.9030380525875527,
            5.010408300357561,
            -2.142476316713338,
            4.246183318746192,
            1.3166218515844226,
            -0.6065063454562809,
        ],
    )
    assert_reference(torch.reciprocal(ones), ones)

    shifted_positive = v1 + 4.0
    shifted_negative = -v2 - 4.0
    expected_maximum = [
        0.2543881165176173,
        1.2246469675357323,
        0.27279133916445375,
        -0.8108145832375699,
        0.7522438271795928,
        0.25344651620814146,
        0.8958830707775604,
        0.23550561173022522,
        0.7595195224783792,
        -0.11001076471125099,
    ]
    expected_minimum = [
        -0.31389947196684775,
        0.05410227877154389,
        -0.2975268443704732,
        -0.9821881249409777,
        -1.107373047165193,
        0.19958453284708083,
        -0.46674961687980204,
        -0.3452157100512797,
        -1.4818182737222112,
        -1.6487873663509485,
    ]
    assert_reference(
        torch.maximum(shifted_positive, shifted_negative), shifted_positive
    )
    assert_reference(torch.maximum(v1, v2), expected_maximum)
    assert_reference(torch.maximum(ones, ones), ones)
    assert_reference(
        torch.minimum(shifted_positive, shifted_negative), shifted_negative
    )
    assert_reference(torch.minimum(v1, v2), expected_minimum)
    assert_reference(torch.minimum(ones, ones), ones)

    ten = torch.full_like(v1, 10.0)
    negative_ten = -ten
    zero = torch.zeros_like(v1)
    assert_reference(torch.minimum(torch.maximum(v1, negative_ten), ten), v1)
    assert_reference(torch.minimum(torch.maximum(ten, negative_ten), zero), zero)
    assert_reference(torch.minimum(torch.maximum(negative_ten, zero), ten), zero)
    bounded = torch.minimum(torch.maximum(v1, v3), v2)
    assert_reference(
        bounded,
        [
            -0.31389947196684775,
            0.7753238220475741,
            -0.2975268443704732,
            -0.9821881249409777,
            -1.107373047165193,
            0.25344651620814146,
            0.6797650174178466,
            -0.3452157100512797,
            -1.4818182737222112,
            -0.11001076471125099,
        ],
    )
    fixed = torch.minimum(torch.maximum(v1, zero), zero)
    assert_reference(fixed, zero)
    assert_reference(fixed, zero)

    assert_reference(torch.linalg.vector_norm(ten, ord=1), 100.0)
    assert_reference(torch.linalg.vector_norm(-v1, ord=1), 6.040500912295451)
    assert_reference(torch.linalg.vector_norm(v1.abs(), ord=1), 6.040500912295451)
    assert_reference(torch.linalg.vector_norm(v1[:1].abs(), ord=1), 0.31389947196684775)
    assert_reference(torch.linalg.vector_norm(v1.new_empty(0), ord=1), 0.0)
    norm2 = torch.linalg.vector_norm(v1, ord=2)
    assert_reference(norm2, 2.444456053273491)
    assert_reference(torch.linalg.vector_norm(v1, ord=float("inf")), 1.6487873663509485)
    assert_reference(torch.linalg.vector_norm(-v1, ord=float("inf")), 1.6487873663509485)
    assert_reference(
        torch.linalg.vector_norm(v1 * v2, ord=float("inf")), 1.12546990765723
    )
    assert_reference(
        torch.linalg.vector_norm(v1 * -v2, ord=float("inf")), 1.12546990765723
    )
    assert_reference(
        torch.linalg.vector_norm(v1 - v2, ord=float("inf")), 2.2413377962005905
    )
    assert_reference(
        torch.linalg.vector_norm(-v1 - -v2, ord=float("inf")), 2.2413377962005905
    )

    dot_v1 = torch.dot(v1, v1)
    assert_reference(dot_v1, 5.975365396385413)
    assert_reference(dot_v1, norm2 * norm2)
    assert_reference(dot_v1, 5.975365396385413)
    assert_reference(torch.dot(v1, v2), -1.524355795187164)

    def signed_dot(a, b, sign):
        if sign == 1:
            return torch.dot(a, torch.clamp_min(b, 0.0))
        if sign == -1:
            return torch.dot(a, torch.clamp_max(b, 0.0))
        return torch.dot(a, b)

    assert_reference(signed_dot(v1, v1, 1), 0.7495089552254102)
    assert_reference(signed_dot(v1, v1, 1), 0.7495089552254102)
    assert_reference(signed_dot(v1, v2, 1), -1.2141797184456542)
    assert_reference(signed_dot(v2, v1, 1), -1.1710926947714702)
    assert_reference(signed_dot(v1, shifted_negative, 1), 0.0)
    assert_reference(signed_dot(v1, v1, -1), 5.225856441160003)
    assert_reference(signed_dot(v1, v1, -1), 5.225856441160003)
    assert_reference(signed_dot(v1, v2, -1), -0.3101760767415097)
    assert_reference(signed_dot(v2, v1, -1), -0.35326310041569386)
    assert_reference(signed_dot(v1, shifted_positive, -1), 0.0)
    assert_reference(signed_dot(v1, v1, -4), 5.975365396385413)
    fallback_dot = signed_dot(v1, v1, 6)
    assert_reference(fallback_dot, norm2 * norm2)
    assert_reference(fallback_dot, 5.975365396385413)
    assert_reference(signed_dot(v1, v2, 0), -1.524355795187164)


def test_kkt_blocks_rhs_and_recovery(floating_dtype, available_device):
    P = torch.tensor(
        [[2.0, 0.5], [0.5, 1.0]],
        dtype=floating_dtype,
        device=available_device,
    )
    A = torch.tensor(
        [[1.0, -1.0], [0.0, 1.0]],
        dtype=floating_dtype,
        device=available_device,
    )
    rho = torch.tensor(
        [0.1, 100.0], dtype=floating_dtype, device=available_device
    )
    sigma = 1e-6
    K = build_kkt_matrix(P, A, sigma, rho)
    assert K.shape == (4, 4)
    assert torch.allclose(K, K.T)
    assert torch.allclose(
        K[:2, :2],
        P + sigma * torch.eye(2, dtype=P.dtype, device=P.device),
    )
    assert torch.allclose(K[:2, 2:], A.T)
    assert torch.allclose(K[2:, :2], A)
    assert torch.allclose(K[2:, 2:], -torch.diag(rho.reciprocal()))

    x = P.new_tensor([0.3, -0.2])
    z = P.new_tensor([0.1, 0.4])
    y = P.new_tensor([0.2, -0.5])
    q = P.new_tensor([-1.0, 2.0])
    rhs = build_kkt_rhs(x, z, y, q, sigma, rho.reciprocal())
    solution = torch.linalg.solve(K, rhs)
    z_tilde = recover_z_tilde(rhs[2:], solution[2:], rho.reciprocal())
    tolerance = 3e-5 if floating_dtype == torch.float32 else 1e-12
    assert torch.allclose(
        z_tilde, A @ solution[:2], atol=tolerance, rtol=tolerance
    )

    scalar_rho = 0.25
    constant_rho = P.new_full((A.shape[0],), scalar_rho)
    scalar_kkt = build_kkt_matrix(P, A, sigma, constant_rho)
    scalar_rhs = build_kkt_rhs(x, z, y, q, sigma, constant_rho.reciprocal())
    assert torch.allclose(
        scalar_kkt[2:, 2:], -torch.eye(2, device=P.device, dtype=P.dtype) / scalar_rho
    )
    assert torch.allclose(
        scalar_rhs,
        torch.cat((sigma * x - q, z - y / scalar_rho)),
    )

    empty = P.new_empty(0)
    empty_A = P.new_empty((0, P.shape[0]))
    assert torch.equal(
        build_kkt_matrix(P, empty_A, sigma, empty),
        P + sigma * torch.eye(2, device=P.device, dtype=P.dtype),
    )
    assert torch.equal(build_kkt_rhs(x, empty, empty, q, sigma, empty), sigma * x - q)
    assert recover_z_tilde(empty, empty, empty).numel() == 0


def test_shared_admm_vector_update_matches_equations(
    floating_dtype, available_device
):
    x_tilde = torch.tensor(
        [1.0, -2.0], dtype=floating_dtype, device=available_device
    )
    x = x_tilde.new_tensor([0.25, 0.5])
    z_tilde = x_tilde.new_tensor([2.0, -2.0, 0.3, 4.0, -5.0])
    z = x_tilde.new_tensor([0.5, -0.5, 0.1, -1.0, 2.0])
    y = x_tilde.new_tensor([0.2, -0.6, 0.7, -0.8, 0.9])
    rho = x_tilde.new_tensor([0.1, 2.0, 3.0, 4.0, 5.0])
    lower = x_tilde.new_tensor([1.0, -1.0, 0.0, -torch.inf, -torch.inf])
    upper = x_tilde.new_tensor([1.0, 1.0, torch.inf, 0.0, torch.inf])
    originals = tuple(
        value.clone() for value in (x_tilde, x, z_tilde, z, y, rho, lower, upper)
    )
    x_next, z_next, y_next = admm_vector_update(
        x_tilde, x, z_tilde, z, y, rho, lower, upper, 1.6,
        rho_inv_vec=rho.reciprocal(),
    )
    tolerance = 3e-6 if floating_dtype == torch.float32 else 1e-12
    assert torch.allclose(
        x_next, x_tilde.new_tensor([1.45, -3.5]), atol=tolerance, rtol=tolerance
    )
    assert torch.allclose(
        z_next,
        x_tilde.new_tensor([1.0, -1.0, 0.6533333333333333, 0.0, -9.02]),
        atol=tolerance,
        rtol=tolerance,
    )
    delta_x = x_next - x
    delta_y = y_next - y
    assert torch.allclose(
        delta_x, x_tilde.new_tensor([1.2, -4.0]), atol=tolerance, rtol=tolerance
    )
    assert torch.allclose(
        delta_y,
        x_tilde.new_tensor([0.19, -3.8, -0.7, 28.0, -0.9]),
        atol=tolerance,
        rtol=tolerance,
    )
    assert torch.allclose(
        y_next,
        x_tilde.new_tensor([0.39, -4.4, 0.0, 27.2, 0.0]),
        atol=tolerance,
        rtol=tolerance,
    )
    for value, original in zip(
        (x_tilde, x, z_tilde, z, y, rho, lower, upper), originals
    ):
        assert torch.equal(value, original)

    empty = x_tilde.new_empty(0)
    empty_x, empty_z, empty_y = admm_vector_update(
        x_tilde, x, empty, empty, empty, empty, empty, empty, 1.6,
        rho_inv_vec=empty,
    )
    assert torch.allclose(empty_x, x_next, atol=tolerance, rtol=tolerance)
    assert empty_z.numel() == 0
    assert empty_y.numel() == 0


@pytest.mark.parametrize(
    "x0,y0,expected_lower",
    [
        (0.0, 0.0, "0x0.0p+0"),
        (0.25, float.fromhex("0x1.3333333333332p-2"), "-0x1.5fffffffffffep+1"),
        (0.25, float.fromhex("0x1.3333333333333p-2"), "-0x1.6000000000000p+1"),
        (0.25, float.fromhex("0x1.3333333333334p-2"), "-0x1.6000000000001p+1"),
        (3.0, float.fromhex("0x1.3333333333333p-2"), "0x0.0p+0"),
    ],
    ids=["cold", "adjacent-down", "equal", "adjacent-up", "cancellation"],
)
def test_c03_first_dense_rhs_uses_stored_vector_reciprocal(
    x0, y0, expected_lower, monkeypatch
):
    def tensor(values):
        return torch.tensor(values, dtype=torch.float64)

    P, q, A, l, u = map(tensor, (
        [[1.0]], [-0.5], [[1.0], [0.5], [-0.25]],
        [0.0, 0.125, -1e30], [1.0, 0.125, 1e30],
    ))
    settings = dict(
        rho=0.1, sigma=0.0625, alpha=1.6, scaling=0, adaptive_rho=False,
        max_iter=1, check_termination=0, eps_abs=1e-9, eps_rel=1e-9,
        polishing=False, return_info=True,
    )
    workspace = TorchOSQPWorkspace()
    y1 = 0.0 if x0 == 0 else -0.5
    if x0 != 0:
        solve_torch_osqp_direct(P, q, A, l, u, settings, workspace)
        # Known warm iterate with the same z = A*x as C's osqp_warm_start.
        workspace.state = {"x": tensor([x0]), "z": A[:, 0] * x0, "y": tensor([y0, y1, 0.0])}

    seen = []
    solve = QDLDLSolver.solve

    def capture(self, rhs, **kwargs):
        assert type(self) is QDLDLSolver
        seen.append((rhs.tolist(), (-self.matrix.diag()[1:]).tolist()))
        return solve(self, rhs, **kwargs)

    monkeypatch.setattr(QDLDLSolver, "solve", capture)
    _, info = solve_torch_osqp_direct(P, q, A, l, u, settings, workspace)
    assert info["factorizations_this_solve"] == int(x0 == 0)
    assert len(seen) == 1
    rhs, inverse = seen[0]
    # Retained C setup stores different reciprocals for the inequality,
    # equality, and loose rows of this fixture before assembling the RHS.
    assert inverse == [10.0, 0.01, 1e6]
    expected = [0.0625 * x0 + 0.5, float.fromhex(expected_lower),
                (0.5 * x0) + -0.01 * y1, -0.25 * x0 + 0.0]
    assert tuple(v.hex() for v in rhs) == tuple(v.hex() for v in expected)


@pytest.mark.parametrize(
    "z0,nu0,expected",
    [
        (0.25, 1.0, "0x1.d000000000000p+2"),
        (0.0, float.fromhex("0x1.3333333333332p-2"), "-0x1.0000000000000p-50"),
        (0.0, float.fromhex("0x1.3333333333333p-2"), "0x0.0p+0"),
        (0.0, float.fromhex("0x1.3333333333334p-2"), "0x1.0000000000000p-51"),
    ],
    ids=["one-ulp", "adjacent-down", "cancellation", "adjacent-up"],
)
def test_c03_dense_ztilde_uses_saved_rhs_and_inverse(z0, nu0, expected, monkeypatch):
    def tensor(values):
        return torch.tensor(values, dtype=torch.float64)

    P, q, A, l, u = map(tensor, (
        [[1.0]], [-0.5], [[1.0], [0.5], [-0.25]],
        [0.0, 0.125, -1e30], [1.0, 0.125, 1e30],
    ))
    settings = dict(
        rho=0.1, sigma=0.0625, alpha=1.6, scaling=0, adaptive_rho=False,
        max_iter=1, check_termination=0, eps_abs=1e-9, eps_rel=1e-9,
        polishing=False, return_info=True,
    )
    workspace = TorchOSQPWorkspace()
    solve_torch_osqp_direct(P, q, A, l, u, settings, workspace)
    workspace.state = {
        "x": tensor([4 * z0]),
        "z": tensor([z0, 0.125, -0.0625]),
        "y": tensor([float.fromhex("0x1.3333333333333p-2"), -0.5, 0.0]),
    }
    # This fixture specifies retained internal iterates, including independent z.
    workspace.work_iterates = {**workspace.state, "source": workspace.state}
    solution_nu = tensor([nu0, -0.25, 0.0])
    solve = QDLDLSolver.solve
    update = runtime.admm_vector_update
    recorded = []

    def fixed_nu(self, rhs, **kwargs):
        assert rhs[1].item().hex() == (z0 - 3.0).hex()
        assert (-self.matrix.diag()[1:]).tolist() == [10.0, 0.01, 1e6]
        solution, info = solve(self, rhs, **kwargs)
        return torch.cat((solution[:1], solution_nu)), info

    def capture(x_tilde, x, z_tilde, *args, **kwargs):
        recorded.append([v.hex() for v in z_tilde.tolist()])
        return update(x_tilde, x, z_tilde, *args, **kwargs)

    monkeypatch.setattr(QDLDLSolver, "solve", fixed_nu)
    monkeypatch.setattr(runtime, "admm_vector_update", capture)
    _, info = solve_torch_osqp_direct(P, q, A, l, u, settings, workspace)
    assert info["factorizations_this_solve"] == 0
    assert recorded == [[expected, "0x1.051eb851eb852p-3", "-0x1.0000000000000p-4"]]


@pytest.mark.parametrize(
    "lower,upper,projected",
    [
        (0.0, 8.0, "0x1.7333333333333p+1"),
        (4.0, 8.0, "0x1.0000000000000p+2"),
        (0.0, 2.0, "0x1.0000000000000p+1"),
        (float.fromhex("0x1.7333333333333p+1"), 8.0, "0x1.7333333333333p+1"),
        (0.0, float.fromhex("0x1.7333333333333p+1"), "0x1.7333333333333p+1"),
        (0.0, float.fromhex("0x1.7333333333332p+1"), "0x1.7333333333332p+1"),
        (float.fromhex("0x1.7333333333334p+1"), 8.0, "0x1.7333333333334p+1"),
    ],
    ids=["interior", "lower-clip", "upper-clip", "equal-lower", "equal-upper",
         "adjacent-below-upper", "adjacent-above-lower"],
)
def test_c03_dense_projection_uses_stored_reciprocal(
    lower, upper, projected, monkeypatch
):
    def tensor(values):
        return torch.tensor(values, dtype=torch.float64)

    P, q, A, l, u = map(tensor, (
        [[1.0]], [-0.5], [[1.0], [0.5], [-0.25]],
        [lower, 0.125, -1e30], [upper, 0.125, 1e30],
    ))
    settings = dict(
        rho=0.1, sigma=0.0625, alpha=1.6, scaling=0, adaptive_rho=False,
        max_iter=1, check_termination=0, eps_abs=1e-9, eps_rel=1e-9,
        polishing=False, return_info=True,
    )
    workspace = TorchOSQPWorkspace()
    solve_torch_osqp_direct(P, q, A, l, u, settings, workspace)
    workspace.state = {
        "x": tensor([0.5]),
        "z": tensor([0.5, 0.125, -0.125]),
        "y": tensor([float.fromhex("0x1.3333333333333p-2"), -0.5, 0.0]),
    }
    update, minimum = runtime.admm_vector_update, torch.minimum
    preclamp = []
    clamped = []

    def capture_minimum(argument, bound):
        preclamp.append(argument[0].item().hex())
        return minimum(argument, bound)

    def capture_update(*args, **kwargs):
        with monkeypatch.context() as projection_only:
            projection_only.setattr(torch, "minimum", capture_minimum)
            result = update(*args, **kwargs)
        clamped.append(result[1][0].item().hex())
        return result

    monkeypatch.setattr(runtime, "recover_z_tilde", lambda *args, **kwargs: tensor(
        [0.125, 0.25, -0.125]
    ))
    monkeypatch.setattr(runtime, "admm_vector_update", capture_update)
    _, info = solve_torch_osqp_direct(P, q, A, l, u, settings, workspace)

    assert type(workspace.linear_solver) is QDLDLSolver
    assert info["factorizations_this_solve"] == 0
    assert (-workspace.linear_solver.matrix.diag()[1:]).tolist() == [10.0, 0.01, 1e6]
    assert (preclamp, clamped) == (["0x1.7333333333333p+1"], [projected])
