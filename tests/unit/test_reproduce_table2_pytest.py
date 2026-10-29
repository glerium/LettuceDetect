"""CPU checks for the Table 2 evaluation protocol; no model weights loaded."""

import importlib.util
from pathlib import Path
import unittest


ENTRY = Path(__file__).resolve().parents[2] / "scripts/reproduce_table2.py"
SPEC = importlib.util.spec_from_file_location("reproduce_table2", ENTRY)
evaluation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluation)


class Table2ProtocolTests(unittest.TestCase):
    def test_continuous_token_score_and_blend(self):
        self.assertAlmostEqual(evaluation.token_score([
            {"prob": 0.12, "pred": 0}, {"prob": 0.42, "pred": 0},
        ]), 0.42)
        self.assertAlmostEqual(evaluation.blend_score(0.2, 0.8), 0.53)
        for tokens in ([], [{"prob": float("nan")}], [{"prob": 1.1}]):
            with self.assertRaises(ValueError):
                evaluation.token_score(tokens)

    def test_paper_score_retains_zero_for_no_hallucinated_spans(self):
        supported = [{"prob": 0.12, "pred": 0}, {"prob": 0.42, "pred": 0}]
        self.assertAlmostEqual(evaluation.token_score(supported), 0.42)
        self.assertEqual(evaluation.paper_token_score(supported), 0.0)
        self.assertAlmostEqual(evaluation.paper_token_score(
            supported + [{"prob": 0.8, "pred": 1}]), 0.8)
        with self.assertRaises(ValueError):
            evaluation.paper_token_score([])

    def test_labels_match_raw_spans(self):
        for value in (0, 1):
            item = {"id": "x", "hallucination_labels": "[{}]" if value else "[]",
                    "hallucination_labels_processed": {"baseless_info": value}}
            self.assertEqual(evaluation.hallucination_label(item), value)
        with self.assertRaises(ValueError):
            evaluation.hallucination_label({"id": "x", "hallucination_labels": "[{}]",
                                           "hallucination_labels_processed": {}})

    def test_selection_preserves_all_tasks_and_smoke_cells(self):
        data = []
        for generator in evaluation.SUPPORTED_GENERATORS:
            for task in evaluation.TASKS:
                for _ in range(150):
                    data.append({"id": str(len(data)), "model": generator,
                                 "task_type": task, "context": "context", "output": "answer",
                                 "query": "", "hallucination_labels": "[]",
                                 "hallucination_labels_processed": {}})
        self.assertEqual(len(evaluation.select_samples(data, smoke=False)), 900)
        smoke = evaluation.select_samples(data, smoke=True)
        self.assertEqual(len(smoke), 6)
        self.assertEqual(len({(s["model"], s["task_type"]) for s in smoke}), 6)
        incomplete = data.copy()
        incomplete.pop(next(i for i, item in enumerate(incomplete)
                            if item["model"] == "mistral-7B-instruct"))
        with self.assertRaises(ValueError):
            evaluation.select_samples(incomplete, smoke=False,
                                      generators=["mistral-7B-instruct"])
        mistral = evaluation.select_samples(data, smoke=False,
                                             generators=["mistral-7B-instruct"])
        self.assertEqual(len(mistral), 450)
        self.assertEqual({s["model"] for s in mistral}, {"mistral-7B-instruct"})
        self.assertEqual(len(evaluation.select_samples(
            data, smoke=True, generators=["mistral-7B-instruct"])), 3)
        self.assertEqual(len(evaluation.select_samples(
            data, smoke=False, generators=evaluation.SUPPORTED_GENERATORS)), 2700)
        self.assertEqual(len(evaluation.select_samples(
            data, smoke=True, generators=evaluation.SUPPORTED_GENERATORS)), 18)
        with self.assertRaises(ValueError):
            evaluation.select_samples(data, False, generators=["mistral-7B-instruct"] * 2)

    def test_grouped_metrics_and_fixed_threshold(self):
        rows = []
        for generator in evaluation.GENERATORS:
            for label, score in [(0, 0.1), (1, 0.5)]:
                rows.append(dict(generator=generator, task_type="QA", label=label,
                                 lettucedetect=score, probe=score, blend=score))
        metrics = evaluation.summarize(rows)
        for group in metrics.values():
            self.assertEqual(group["n"], 2)
            for method in group["methods"].values():
                self.assertEqual(method["auroc"], 1.0)
                self.assertEqual(method["f1_at_0_5"], 1.0)
        single_class = [dict(generator=g, task_type="QA", label=0,
                             lettucedetect=0.1, probe=0.1, blend=0.1)
                        for g in evaluation.GENERATORS]
        for group in evaluation.summarize(single_class).values():
            self.assertIsNone(group["methods"]["blend"]["auroc"])
        mistral = [dict(r, generator="mistral-7B-instruct") for r in rows[:2]]
        metrics = evaluation.summarize(mistral)
        self.assertEqual(set(metrics), {"mistral-7B-instruct"})
        self.assertEqual(metrics["mistral-7B-instruct"]["methods"]["blend"]["auroc"], 1.0)

    def test_full_test_pooled_and_macro_metrics(self):
        rows = []
        for generator in evaluation.SUPPORTED_GENERATORS:
            for label, score in [(0, 0.1), (1, 0.9)]:
                rows.append(dict(generator=generator, task_type="QA", label=label,
                                 lettucedetect=score, probe=score, blend=score))
        result = evaluation.summarize_all(rows)
        self.assertEqual(result["n"], 12)
        self.assertEqual(len(evaluation.summarize(rows)), 6)
        for block in ("pooled", "generator_macro"):
            for method in result[block].values():
                self.assertEqual(method, {"auroc": 1.0, "f1_at_0_5": 1.0})


if __name__ == "__main__":
    unittest.main()
