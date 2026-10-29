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
        for generator in evaluation.GENERATORS:
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
        with self.assertRaises(ValueError):
            evaluation.select_samples(data[:-1], smoke=False)

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


if __name__ == "__main__":
    unittest.main()
