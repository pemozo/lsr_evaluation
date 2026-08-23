import math
import numbers

import torch


def _validate_generation_parameters(batch_size, node_cnt, params):
    required = {
        "int_min",
        "int_max",
        "scaler",
        "lambda_asym",
        "affected_fraction",
        "q_mode",
        "placement_mode",
        "direction_mode",
        "seed",
    }
    missing = sorted(required.difference(params))
    if missing:
        raise ValueError(f"Missing problem generation parameters: {missing}")

    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")
    if not isinstance(node_cnt, int) or isinstance(node_cnt, bool) or node_cnt < 2:
        raise ValueError("node_cnt must be an integer greater than or equal to 2.")

    int_min = params["int_min"]
    int_max = params["int_max"]
    if not isinstance(int_min, int) or isinstance(int_min, bool) or int_min < 1:
        raise ValueError("int_min must be a positive integer.")
    if not isinstance(int_max, int) or isinstance(int_max, bool) or int_max <= int_min:
        raise ValueError("int_max must be an integer greater than int_min.")

    scaler = params["scaler"]
    if not isinstance(scaler, numbers.Real) or isinstance(scaler, bool) or scaler <= 0:
        raise ValueError("scaler must be positive.")

    lambda_asym = params["lambda_asym"]
    if (
        not isinstance(lambda_asym, numbers.Real)
        or isinstance(lambda_asym, bool)
        or not 0.0 <= lambda_asym < 1.0
    ):
        raise ValueError("lambda_asym must satisfy 0 <= lambda_asym < 1.")

    affected_fraction = params["affected_fraction"]
    if (
        not isinstance(affected_fraction, numbers.Real)
        or isinstance(affected_fraction, bool)
        or not 0.0 <= affected_fraction <= 1.0
    ):
        raise ValueError("affected_fraction must be in [0, 1].")

    q_mode = params["q_mode"]
    if q_mode == "constant":
        if "q_value" not in params:
            raise ValueError("q_value is required when q_mode='constant'.")
        q_value = params["q_value"]
        if (
            not isinstance(q_value, numbers.Real)
            or isinstance(q_value, bool)
            or not 0.0 <= q_value <= 1.0
        ):
            raise ValueError("q_value must be in [0, 1].")
    elif q_mode == "uniform":
        if "q_min" not in params or "q_max" not in params:
            raise ValueError("q_min and q_max are required when q_mode='uniform'.")
        q_min = params["q_min"]
        q_max = params["q_max"]
        if (
            not isinstance(q_min, numbers.Real)
            or isinstance(q_min, bool)
            or not isinstance(q_max, numbers.Real)
            or isinstance(q_max, bool)
            or not 0.0 <= q_min <= q_max <= 1.0
        ):
            raise ValueError("q_min and q_max must satisfy 0 <= q_min <= q_max <= 1.")
    else:
        raise ValueError("q_mode must currently be either 'constant' or 'uniform'.")

    if params["placement_mode"] != "random":
        raise ValueError("placement_mode must currently be 'random'.")
    if params["direction_mode"] != "random":
        raise ValueError("direction_mode must currently be 'random'.")

    seed = params["seed"]
    if (
        not isinstance(seed, int)
        or isinstance(seed, bool)
        or not 0 <= seed <= torch.iinfo(torch.int64).max
    ):
        raise ValueError("seed must be an integer in [0, 2**63 - 1].")

    target_rate = params.get("target_triangle_violation_rate")
    if target_rate is not None and (
        not isinstance(target_rate, numbers.Real)
        or isinstance(target_rate, bool)
        or not 0.0 <= target_rate <= 1.0
    ):
        raise ValueError("target_triangle_violation_rate must be None or in [0, 1].")

    triangle_tolerance = params.get("triangle_violation_tolerance", 0.01)
    if (
        not isinstance(triangle_tolerance, numbers.Real)
        or isinstance(triangle_tolerance, bool)
        or not 0.0 <= triangle_tolerance <= 1.0
    ):
        raise ValueError("triangle_violation_tolerance must be in [0, 1].")

    triangle_eps = params.get("triangle_violation_eps", 1e-7)
    if (
        not isinstance(triangle_eps, numbers.Real)
        or isinstance(triangle_eps, bool)
        or triangle_eps < 0.0
    ):
        raise ValueError("triangle_violation_eps must be non-negative.")

    enforce_triangle_inequality = params.get("enforce_triangle_inequality", False)
    if not isinstance(enforce_triangle_inequality, bool):
        raise ValueError("enforce_triangle_inequality must be a bool.")
    if (
        enforce_triangle_inequality
        and target_rate is not None
        and target_rate > triangle_tolerance
    ):
        raise ValueError(
            "A positive target_triangle_violation_rate is incompatible with "
            "enforce_triangle_inequality=True unless it lies within "
            "triangle_violation_tolerance of zero."
        )


def _make_symmetric_metric_base(batch_size, node_cnt, params, generator):
    """Create the original symmetric metric matrix S without changing its logic."""
    problems = torch.randint(
        low=params["int_min"],
        high=params["int_max"],
        size=(batch_size, node_cnt, node_cnt),
        dtype=torch.int64,
        generator=generator,
    )

    upper = torch.triu(problems, diagonal=1)
    problems = upper + upper.transpose(1, 2)

    idx = torch.arange(node_cnt)
    problems[:, idx, idx] = 0

    # Metric closure of the symmetric base matrix (Floyd-Warshall style).
    for k in range(node_cnt):
        via_k = problems[:, :, k].unsqueeze(2) + problems[:, k, :].unsqueeze(1)
        problems = torch.minimum(problems, via_k)

    return problems.float() / params["scaler"]


def _make_asymmetry_matrices(batch_size, node_cnt, params, generator):
    """Return symmetric local strengths Q and antisymmetric directions Z."""
    q_matrix = torch.zeros((batch_size, node_cnt, node_cnt), dtype=torch.float32)
    direction_matrix = torch.zeros_like(q_matrix)

    pair_indices = torch.triu_indices(node_cnt, node_cnt, offset=1)
    pair_count = pair_indices.size(1)
    affected_count = int(
        math.floor(float(params["affected_fraction"]) * pair_count + 0.5)
    )

    for batch_idx in range(batch_size):
        if affected_count == 0:
            continue

        if affected_count == pair_count:
            selected = torch.arange(pair_count)
        else:
            selected = torch.randperm(pair_count, generator=generator)[:affected_count]

        selected_i = pair_indices[0, selected]
        selected_j = pair_indices[1, selected]

        if params["q_mode"] == "constant":
            q_values = torch.full((affected_count,), float(params["q_value"]))
        else:
            q_min = float(params["q_min"])
            q_max = float(params["q_max"])
            q_values = q_min + (q_max - q_min) * torch.rand(
                affected_count, generator=generator
            )

        # A sign of +1 makes i -> j more expensive and j -> i cheaper.
        signs = 2 * torch.randint(
            0, 2, (affected_count,), generator=generator, dtype=torch.int64
        ).float() - 1.0

        q_matrix[batch_idx, selected_i, selected_j] = q_values
        q_matrix[batch_idx, selected_j, selected_i] = q_values
        direction_matrix[batch_idx, selected_i, selected_j] = signs
        direction_matrix[batch_idx, selected_j, selected_i] = -signs

    return q_matrix, direction_matrix, pair_indices, affected_count


def _directed_metric_closure(problems):
    """Return a numerically stable directed all-pairs shortest-path closure."""
    closed = problems.clone()
    for _ in range(8):
        previous = closed
        for k in range(closed.size(1)):
            via_k = closed[:, :, k].unsqueeze(2) + closed[:, k, :].unsqueeze(1)
            closed = torch.minimum(closed, via_k)
        if torch.equal(closed, previous):
            return closed

    raise RuntimeError(
        "Directed metric closure did not reach a floating-point fixed point "
        "within 8 relaxation passes."
    )


def _triangle_violation_diagnostics(problems, eps):
    """Compute v_ijk diagnostics over all ordered triples of distinct nodes."""
    batch_size, node_cnt, _ = problems.shape
    total_triples = node_cnt * (node_cnt - 1) * (node_cnt - 2)
    violation_count = torch.zeros(batch_size, dtype=torch.int64)
    violation_sum = torch.zeros(batch_size, dtype=problems.dtype)
    violation_max = torch.zeros(batch_size, dtype=problems.dtype)

    if total_triples == 0:
        return {
            "triangle_violation_rate": violation_sum.clone(),
            "triangle_violation_mean": violation_sum,
            "triangle_violation_max": violation_max,
        }

    distinct_ij = ~torch.eye(node_cnt, dtype=torch.bool)
    for k in range(node_cnt):
        relevant = distinct_ij.clone()
        relevant[k, :] = False
        relevant[:, k] = False

        direct = problems[:, relevant]
        indirect = (
            problems[:, :, k].unsqueeze(2) + problems[:, k, :].unsqueeze(1)
        )[:, relevant]
        strengths = torch.clamp_min(direct - indirect, 0.0) / direct
        violated = strengths > eps

        violation_count += violated.sum(dim=1)
        violation_sum += torch.where(violated, strengths, 0.0).sum(dim=1)
        violation_max = torch.maximum(
            violation_max,
            torch.where(violated, strengths, 0.0).max(dim=1).values,
        )

    violation_mean = torch.where(
        violation_count > 0,
        violation_sum / violation_count.to(problems.dtype),
        torch.zeros_like(violation_sum),
    )
    return {
        "triangle_violation_rate": violation_count.to(problems.dtype)
        / total_triples,
        "triangle_violation_mean": violation_mean,
        "triangle_violation_max": violation_max,
    }


def _calibrate_asymmetry_strengths(
    symmetric_base, perturbation, max_lambda, target_rate, eps
):
    """Choose the closest attainable violation rate by scaling perturbations."""
    batch_size, node_cnt, _ = symmetric_base.shape
    total_triples = node_cnt * (node_cnt - 1) * (node_cnt - 2)
    effective_lambdas = torch.zeros(batch_size, dtype=symmetric_base.dtype)
    if total_triples == 0 or max_lambda == 0.0:
        return effective_lambdas

    distinct_ij = ~torch.eye(node_cnt, dtype=torch.bool)
    target_count = float(target_rate) * total_triples

    for batch_idx in range(batch_size):
        base = symmetric_base[batch_idx]
        delta = perturbation[batch_idx]
        activation_thresholds = []
        initially_violated = 0

        for k in range(node_cnt):
            relevant = distinct_ij.clone()
            relevant[k, :] = False
            relevant[:, k] = False

            intercept = (
                (1.0 - eps) * base
                - base[:, k].unsqueeze(1)
                - base[k, :].unsqueeze(0)
            )[relevant]
            slope = (
                (1.0 - eps) * delta
                - delta[:, k].unsqueeze(1)
                - delta[k, :].unsqueeze(0)
            )[relevant]
            at_max = intercept + max_lambda * slope

            initially_violated += int((intercept > 0.0).sum().item())
            activates = (intercept <= 0.0) & (slope > 0.0) & (at_max > 0.0)
            if activates.any():
                activation_thresholds.append(
                    -intercept[activates] / slope[activates]
                )

        if not activation_thresholds:
            continue

        thresholds = torch.sort(torch.cat(activation_thresholds)).values
        activation_count = thresholds.numel()
        desired_activations = target_count - initially_violated

        candidate_counts = {0, activation_count}
        for rank in (math.floor(desired_activations), math.ceil(desired_activations)):
            rank = min(max(rank, 1), activation_count)
            pivot = thresholds[rank - 1]
            candidate_counts.add(
                int(torch.searchsorted(thresholds, pivot, right=False).item())
            )
            candidate_counts.add(
                int(torch.searchsorted(thresholds, pivot, right=True).item())
            )

        selected_count = min(
            candidate_counts,
            key=lambda count: (
                abs(initially_violated + count - target_count),
                count,
            ),
        )
        if selected_count == 0:
            effective_lambda = 0.0
        elif selected_count == activation_count:
            effective_lambda = max_lambda
        else:
            lower = thresholds[selected_count - 1].item()
            upper = thresholds[selected_count].item()
            effective_lambda = (lower + upper) / 2.0

        effective_lambdas[batch_idx] = effective_lambda

    return effective_lambdas


def get_random_problems(
    batch_size,
    node_cnt,
    problem_gen_params,
    return_diagnostics=False,
    return_internal_matrices=False,
):
    """Generate controlled ATSP matrices.

    By default only D is returned, preserving the original tensor-only API. If
    ``return_diagnostics`` is true, the return value is ``(D, diagnostics)``.
    Full S, Q and Z tensors are included only when
    ``return_internal_matrices`` is also true.
    """
    _validate_generation_parameters(batch_size, node_cnt, problem_gen_params)
    if return_internal_matrices and not return_diagnostics:
        raise ValueError("return_internal_matrices requires return_diagnostics=True.")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(problem_gen_params["seed"])

    symmetric_base = _make_symmetric_metric_base(
        batch_size, node_cnt, problem_gen_params, generator
    )
    q_matrix, direction_matrix, pair_indices, selected_pair_count = (
        _make_asymmetry_matrices(
            batch_size, node_cnt, problem_gen_params, generator
        )
    )

    lambda_asym = float(problem_gen_params["lambda_asym"])
    perturbation = symmetric_base * direction_matrix * q_matrix
    target_rate = problem_gen_params.get("target_triangle_violation_rate")
    triangle_eps = float(problem_gen_params.get("triangle_violation_eps", 1e-7))
    triangle_tolerance = float(
        problem_gen_params.get("triangle_violation_tolerance", 0.01)
    )
    enforce_triangle_inequality = problem_gen_params.get(
        "enforce_triangle_inequality", False
    )

    if target_rate is None:
        effective_lambdas = torch.full(
            (batch_size,), lambda_asym, dtype=symmetric_base.dtype
        )
    else:
        effective_lambdas = _calibrate_asymmetry_strengths(
            symmetric_base,
            perturbation,
            lambda_asym,
            float(target_rate),
            triangle_eps,
        )

    if torch.count_nonzero(effective_lambdas) == 0:
        # This explicit branch guarantees bitwise equality D == S for lambda=0.
        perturbed_problems = symmetric_base.clone()
    else:
        perturbed_problems = symmetric_base * (
            1.0
            + effective_lambdas[:, None, None] * direction_matrix * q_matrix
        )

    diagonal = torch.arange(node_cnt)
    perturbed_problems[:, diagonal, diagonal] = 0.0

    # Validate the exact perturbation identity before an optional final closure.
    pair_i, pair_j = pair_indices
    perturbed_forward = perturbed_problems[:, pair_i, pair_j]
    perturbed_backward = perturbed_problems[:, pair_j, pair_i]
    perturbed_pairwise_asymmetry = torch.abs(
        perturbed_forward - perturbed_backward
    ) / (perturbed_forward + perturbed_backward)
    expected_asymmetry = (
        effective_lambdas[:, None] * q_matrix[:, pair_i, pair_j]
    )

    if not torch.allclose(
        perturbed_pairwise_asymmetry, expected_asymmetry, rtol=1e-5, atol=1e-7
    ):
        max_error = torch.max(
            torch.abs(perturbed_pairwise_asymmetry - expected_asymmetry)
        )
        raise RuntimeError(
            "Generated pairwise asymmetry does not match lambda_asym * q_ij; "
            f"maximum error: {max_error.item():.3e}."
        )

    pair_means = (perturbed_forward + perturbed_backward) / 2.0
    base_upper = symmetric_base[:, pair_i, pair_j]
    if not torch.allclose(pair_means, base_upper, rtol=1e-6, atol=1e-7):
        max_error = torch.max(torch.abs(pair_means - base_upper))
        raise RuntimeError(
            "Generated directed pair means do not match the symmetric base; "
            f"maximum error: {max_error.item():.3e}."
        )

    if enforce_triangle_inequality:
        asymmetric_problems = _directed_metric_closure(perturbed_problems)
    else:
        asymmetric_problems = perturbed_problems

    if not return_diagnostics:
        return asymmetric_problems

    forward = asymmetric_problems[:, pair_i, pair_j]
    backward = asymmetric_problems[:, pair_j, pair_i]
    pairwise_asymmetry = torch.abs(forward - backward) / (forward + backward)
    actually_asymmetric = forward != backward
    asym_pair_count = actually_asymmetric.sum(dim=1)
    total_pair_count = pair_indices.size(1)
    diagnostics = {
        "pairwise_asym_mean": pairwise_asymmetry.mean(dim=1),
        "pairwise_asym_max": pairwise_asymmetry.max(dim=1).values,
        "asym_pair_count": asym_pair_count,
        "asym_pair_fraction": asym_pair_count.float() / total_pair_count,
        "selected_pair_count": torch.full(
            (batch_size,), selected_pair_count, dtype=torch.int64
        ),
        "effective_lambda_asym": effective_lambdas,
        "enforce_triangle_inequality": enforce_triangle_inequality,
        "triangle_violation_eps": triangle_eps,
    }
    diagnostics.update(
        _triangle_violation_diagnostics(asymmetric_problems, triangle_eps)
    )
    if target_rate is not None:
        target_errors = torch.abs(
            diagnostics["triangle_violation_rate"] - float(target_rate)
        )
        diagnostics.update(
            {
                "target_triangle_violation_rate": float(target_rate),
                "triangle_violation_target_error": target_errors,
                "triangle_violation_target_met": target_errors
                <= triangle_tolerance,
            }
        )
    if return_internal_matrices:
        diagnostics.update(
            {
                "symmetric_base": symmetric_base,
                "q_matrix": q_matrix,
                "direction_matrix": direction_matrix,
            }
        )

    return asymmetric_problems, diagnostics
