import unittest

import torch

try:
    from .Ctrl_ATSProblemDef import get_random_problems
except ImportError:
    from Ctrl_ATSProblemDef import get_random_problems


def _params(**overrides):
    params = {
        "int_min": 1,
        "int_max": 10_000,
        "scaler": 10_000,
        "lambda_asym": 0.6,
        "affected_fraction": 1.0,
        "q_mode": "constant",
        "q_value": 1.0,
        "placement_mode": "random",
        "direction_mode": "random",
        "seed": 42,
    }
    params.update(overrides)
    return params


class ControlledAtspGeneratorTests(unittest.TestCase):
    def test_default_perturbation_identity_is_preserved(self):
        problems, diagnostics = get_random_problems(
            2,
            8,
            _params(),
            return_diagnostics=True,
            return_internal_matrices=True,
        )
        expected = diagnostics["symmetric_base"] * (
            1.0
            + 0.6
            * diagnostics["direction_matrix"]
            * diagnostics["q_matrix"]
        )

        self.assertTrue(torch.equal(problems, expected))

    def test_default_mode_reports_triangle_diagnostics(self):
        problems, diagnostics = get_random_problems(
            3, 8, _params(), return_diagnostics=True
        )

        self.assertEqual(problems.shape, (3, 8, 8))
        self.assertFalse(diagnostics["enforce_triangle_inequality"])
        self.assertTrue(torch.all(diagnostics["triangle_violation_rate"] >= 0.0))
        self.assertTrue(torch.all(diagnostics["triangle_violation_rate"] <= 1.0))
        self.assertTrue(torch.any(diagnostics["triangle_violation_rate"] > 0.0))

    def test_target_rate_uses_closest_scaled_perturbation(self):
        target = 0.05
        _, diagnostics = get_random_problems(
            2,
            10,
            _params(
                target_triangle_violation_rate=target,
                triangle_violation_tolerance=0.02,
            ),
            return_diagnostics=True,
        )

        self.assertTrue(torch.all(diagnostics["effective_lambda_asym"] <= 0.6))
        self.assertTrue(torch.all(diagnostics["effective_lambda_asym"] >= 0.0))
        self.assertTrue(torch.all(diagnostics["triangle_violation_target_met"]))

    def test_enforcement_closes_final_directed_matrix(self):
        problems, diagnostics = get_random_problems(
            2,
            9,
            _params(
                enforce_triangle_inequality=True,
                triangle_violation_eps=0.0,
            ),
            return_diagnostics=True,
        )

        self.assertTrue(diagnostics["enforce_triangle_inequality"])
        self.assertTrue(torch.all(diagnostics["triangle_violation_rate"] == 0.0))
        for k in range(problems.size(1)):
            via_k = problems[:, :, k].unsqueeze(2) + problems[:, k, :].unsqueeze(1)
            self.assertTrue(torch.all(problems <= via_k))


if __name__ == "__main__":
    unittest.main()
