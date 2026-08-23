import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import locality_experiment as locality


def re_value(text, key):
    for line in text.splitlines():
        if line.startswith(f"{key} ="):
            return line.split("=", 1)[1].strip()
    raise AssertionError(f"Missing {key}")


class LocalityExperimentTests(unittest.TestCase):
    def make_instance(self, matrix, symmetry_type="asymmetric"):
        indices = np.arange(locality.TARGET_N)
        return locality.PreparedInstance(
            dataset="rrnco",
            instance_id="0",
            instance_index=0,
            source_file=Path("synthetic"),
            source_dimension=locality.TARGET_N,
            matrix=matrix,
            selected_source_indices=indices,
            selected_original_node_ids=indices,
            node_selection_seed=None,
            symmetry_type=symmetry_type,
            np_allclose_d_dt=symmetry_type == "symmetric",
        )

    def make_args(self):
        return SimpleNamespace(
            seed=1234,
            max_trials=None,
            scale=1_000_000,
            runs=1,
            keep_work=False,
            incoming="auto",
            k_values=[1, 2, 4, 8, 16],
        )

    def solved(self, tour):
        return {
            "tour": np.asarray(tour),
            "lkh_seed": 1234,
            "lkh_objective": 64.0,
            "lkh_scaled_objective": 64_000_000,
            "lkh_runtime_seconds": 0.1,
            "lkh_problem_type": "ATSP",
        }

    def test_knn_never_contains_diagonal(self):
        rng = np.random.default_rng(7)
        matrix = rng.random((locality.TARGET_N, locality.TARGET_N))
        np.fill_diagonal(matrix, -1000.0)
        order = locality.knn_order(matrix)
        self.assertFalse(
            np.any(order[:, : max(locality.DEFAULT_K_VALUES)] == np.arange(locality.TARGET_N)[:, None])
        )

    def test_offset_average_is_invariant_to_tour_start(self):
        n = locality.TARGET_N
        rng = np.random.default_rng(19)
        matrix = rng.random((n, n))
        np.fill_diagonal(matrix, 0.0)
        instance = self.make_instance(matrix)
        args = self.make_args()

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.object(locality, "solve_instance", return_value=self.solved(np.arange(n))):
                _, original, _, _ = locality.analyze_instance(
                    instance, args, 16, "unused", Path(temp_dir)
                )
            with patch.object(
                locality, "solve_instance", return_value=self.solved(np.roll(np.arange(n), 11))
            ):
                _, rotated, _, _ = locality.analyze_instance(
                    instance, args, 16, "unused", Path(temp_dir)
                )

        np.testing.assert_allclose(
            [row["locality_outgoing"] for row in original],
            [row["locality_outgoing"] for row in rotated],
        )
        np.testing.assert_allclose(
            [row["locality_incoming"] for row in original],
            [row["locality_incoming"] for row in rotated],
        )

    def test_problem_writer_preserves_atsp_direction(self):
        matrix = np.array([[0, 1, 2], [3, 0, 4], [5, 6, 0]], dtype=np.int64)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "directed.tsp"
            problem_type = locality.write_problem(path, "directed", matrix, symmetric=False)
            text = path.read_text(encoding="ascii")

        self.assertEqual(problem_type, "ATSP")
        self.assertIn("TYPE: ATSP", text)
        self.assertIn("0 1 2\n3 0 4\n5 6 0", text)

    def test_tour_parser_validates_and_converts_indexing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "tour.txt"
            nodes = "\n".join(str(node) for node in range(1, locality.TARGET_N + 1))
            path.write_text(
                "TYPE : TOUR\nCOMMENT : Length = 123\nTOUR_SECTION\n"
                + nodes
                + "\n-1\nEOF\n",
                encoding="ascii",
            )
            tour, objective = locality.read_tour(path, locality.TARGET_N)

        np.testing.assert_array_equal(tour, np.arange(locality.TARGET_N))
        self.assertEqual(objective, 123)

    def test_lkh_wrapper_recomputes_objective_and_cleans_work_files(self):
        n = locality.TARGET_N
        matrix = np.ones((n, n), dtype=np.float64)
        np.fill_diagonal(matrix, 0.0)
        matrix[0, 1] = 2.0
        instance = self.make_instance(matrix)
        args = self.make_args()
        expected_scaled = 65_000_000

        def fake_lkh(command, cwd, **unused):
            work_dir = Path(cwd)
            parameter_text = (work_dir / command[1]).read_text(encoding="ascii")
            problem_name = re_value(parameter_text, "PROBLEM_FILE")
            tour_name = re_value(parameter_text, "OUTPUT_TOUR_FILE")
            problem_text = (work_dir / problem_name).read_text(encoding="ascii")
            self.assertIn("TYPE: ATSP", problem_text)
            self.assertIn("RUNS = 1", parameter_text)
            self.assertIn("MAX_TRIALS = 128", parameter_text)
            self.assertIn("SEED = 1234", parameter_text)
            nodes = "\n".join(str(node) for node in range(1, n + 1))
            (work_dir / tour_name).write_text(
                f"TYPE : TOUR\nCOMMENT : Length = {expected_scaled}\n"
                f"TOUR_SECTION\n{nodes}\n-1\nEOF\n",
                encoding="ascii",
            )
            return SimpleNamespace(returncode=0, stdout="fake LKH output\n")

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            with patch.object(locality.subprocess, "run", side_effect=fake_lkh):
                solved = locality.solve_instance(instance, args, "fake-LKH", output_dir)
            self.assertAlmostEqual(solved["lkh_objective"], 65.0)
            self.assertEqual(solved["lkh_scaled_objective"], expected_scaled)
            self.assertFalse(any((output_dir / "work").iterdir()))
            self.assertTrue(any((output_dir / "lkh_logs" / "rrnco").iterdir()))

    def test_repository_segment_size_is_discovered_as_50(self):
        project_root = Path(__file__).resolve().parents[2]
        segment_size, sources = locality.discover_segment_size(project_root)
        self.assertEqual(segment_size, 50)
        self.assertTrue(sources)

    def test_aggregation_comparisons_and_final_report(self):
        args = SimpleNamespace(
            datasets=["ctrl_atsp_data", "matnet", "rrnco"],
            k_values=[1, 2],
            bootstrap_samples=100,
            seed=1234,
            paired_comparisons=False,
            node_selection="random",
            runs=1,
            max_trials=None,
            scale=1_000_000,
            workers=2,
        )
        instance_rows = []
        node_rows = []
        manifests = []
        for dataset_index, dataset in enumerate(args.datasets):
            symmetry = "symmetric" if dataset == "ctrl_atsp_data" else "asymmetric"
            for instance_id in range(4):
                manifests.append({"dataset": dataset, "symmetry_type": symmetry})
                for k in args.k_values:
                    value = 0.25 + 0.01 * dataset_index + 0.001 * instance_id + 0.002 * k
                    instance_rows.append(
                        {
                            "dataset": dataset,
                            "instance_id": str(instance_id),
                            "K": k,
                            "locality_outgoing": value,
                            "locality_incoming": value + (0.005 if symmetry == "asymmetric" else 0.0),
                            "node_std_outgoing": 0.1,
                            "node_std_incoming": 0.1,
                            "random_baseline": 15 / 63,
                        }
                    )
                    for node_id in range(locality.TARGET_N):
                        node_rows.append(
                            {
                                "dataset": dataset,
                                "instance_id": str(instance_id),
                                "K": k,
                                "locality_outgoing": value,
                                "locality_incoming": value
                                + (0.005 if symmetry == "asymmetric" else 0.0),
                            }
                        )

        summaries = locality.aggregate(instance_rows, node_rows, args)
        comparisons = locality.compare_datasets(instance_rows, args)
        report = locality.build_final_report(
            args, 16, 50, "LKH", manifests, summaries, comparisons, elapsed=1.5
        )

        self.assertEqual(len(summaries), 12)
        self.assertEqual(len(comparisons), 8)
        self.assertIn("LOCALITY EXPERIMENT - FINAL RESULTS", report)
        self.assertIn("MATNET vs CTRL_ATSP_DATA", report)
        self.assertIn("RRNCO vs CTRL_ATSP_DATA", report)
        self.assertIn("comparison mode: independent", report)


if __name__ == "__main__":
    unittest.main()
