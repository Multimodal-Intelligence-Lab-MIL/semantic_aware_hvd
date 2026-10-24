import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import result_hatemm as evaluator


def record(raw, stored="nonhate"):
    return {"raw_model_output": raw, "verdict_label": stored}


def segment(raw, votes=None):
    return {
        "context_steps": [record(raw, "uncertain")],
        "final_label_before_fallback": "uncertain",
        "final_label_after_fallback": "nonhate",
        "fallback_votes": [record(vote) for vote in votes or []],
    }


def video(segments, **extra):
    return {"final_video_label": "nonhate", "segment_results": segments, **extra}


class VerdictRecoveryTests(unittest.TestCase):
    def test_markdown_escape_preserves_positive_verdict(self):
        raw = r'{"verdict\_label": "hate", "rationale": "Example."}'
        self.assertEqual(evaluator.parse_raw_verdict(raw), ("hate", "repaired_json"))
        self.assertEqual(evaluator.extract_label_from_data(video([segment(raw)])), 1)

    def test_complete_label_can_survive_truncated_rationale(self):
        raw = r'{"verdict\_label": "hate", "rationale": "Unfinished'
        self.assertEqual(evaluator.parse_raw_verdict(raw), ("hate", "partial_json"))

    def test_incomplete_or_ambiguous_label_is_not_guessed(self):
        for raw in (
            '{"verdict_label": "ha', '{"verdict_label": "hate',
            '{"verdict_label": "hate or nonhate"}', '{"verdict_label": 1garbage',
            '{"verdict_label": "hate", "verdict_label": "nonhate"}',
            '{"verdict_label": "hate"}\n{"verdict_label": "nonhate"}',
            '```json\n{"verdict_label": "hate"}\n```\n```json\n{"verdict_label": "nonhate"}\n```',
            '"hate',
            'This may contain hate but is not an answer.',
        ):
            with self.subTest(raw=raw):
                self.assertIsNone(evaluator.parse_raw_verdict(raw)[0])

    def test_nested_evidence_and_rationale_are_not_decisions(self):
        for raw in (
            '{"hate_hypothesis": {"verdict_label": "hate"}}',
            '{"rationale": "hate", "hypothesis_label": "hate"}',
            '{"rationale": "The example says \\"verdict_label\\": \\"hate\\"',
        ):
            with self.subTest(raw=raw):
                self.assertIsNone(evaluator.parse_raw_verdict(raw)[0])

    def test_thinking_and_fences(self):
        raw = '<think>{"verdict_label": "hate"}</think>\n```json\n{"verdict_label": "nonhate"}\n```'
        self.assertEqual(evaluator.parse_raw_verdict(raw)[0], "nonhate")
        self.assertIsNone(evaluator.parse_raw_verdict('<think>{"verdict_label": "hate"}')[0])

    def test_json_escapes_still_decode(self):
        raw = json.dumps({"verdict_label": "nonhate", "rationale": 'Quote "x"; path C:\\tmp; newline\n'})
        self.assertEqual(evaluator.parse_raw_verdict(raw), ("nonhate", "json"))

    def test_unknown_votes_do_not_count_as_negative(self):
        unknown = '{"verdict_label": "uncertain"}'
        self.assertIsNone(evaluator.recover_segment_label(segment(unknown, ["hate", "", ""])))
        self.assertEqual(evaluator.recover_segment_label(segment(unknown, ["hate", "hate", ""])), 1)
        self.assertEqual(evaluator.recover_segment_label(segment(unknown, ["nonhate", "nonhate", ""])), 0)

    def test_direct_verdict_precedes_fallback(self):
        self.assertEqual(evaluator.recover_segment_label(segment('"nonhate"', ["hate"] * 3)), 0)

    def test_invalid_raw_does_not_reuse_saved_default(self):
        self.assertIsNone(evaluator.extract_label_from_data(video([segment("")])))

    def test_video_or_rule_and_incomplete_coverage(self):
        self.assertEqual(evaluator.extract_label_from_data(video([segment("hate"), segment("")])), 1)
        self.assertIsNone(evaluator.extract_label_from_data(video([segment("nonhate"), segment("")])))
        self.assertEqual(evaluator.extract_label_from_data(video([segment("nonhate")])), 0)
        self.assertIsNone(evaluator.extract_label_from_data(video([segment("nonhate")], unprocessed_segment_ids=[2])))
        self.assertEqual(evaluator.extract_label_from_data(video([segment("hate")], unprocessed_segment_ids=[2])), 1)

    def test_title_attribution(self):
        data = video([segment("nonhate")], title_reasoning=record("hate"))
        self.assertEqual(evaluator.extract_label_from_data(data), 1)

    def test_stored_and_legacy_modes(self):
        data = video([segment("hate")])
        self.assertEqual(evaluator.extract_label_from_data(data, "stored"), 0)
        self.assertEqual(evaluator.extract_label_from_data({"final_video_label": "hate"}), 1)
        legacy = {"multi_turn_results": {"turn_4": {"raw_response": '{"final_decision": {"label": 1}}'}}}
        self.assertEqual(evaluator.extract_label_from_data(legacy), 1)

    def test_invalid_final_result_does_not_use_stale_legacy_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "video_result.json").write_text(json.dumps(video([segment("")])))
            (path / "mars_result.json").write_text('{"label": 0}')
            self.assertIsNone(evaluator.extract_label_for_video_dir(path))

    def test_metrics_report_missing_labels_and_strict_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = root / "results"
            for name, raw in (("positive", "hate"), ("negative", "nonhate"), ("unknown", "")):
                path = results / name
                path.mkdir(parents=True)
                (path / "video_result.json").write_text(json.dumps(video([segment(raw)], video_id=name)))
            splits = root / "splits.json"
            splits.write_text(json.dumps({"folds": [{"fold": 0, "test": [
                {"video_id": "positive", "label": 1},
                {"video_id": "negative", "label": 0},
                {"video_id": "unknown", "label": 0},
            ]}]}))
            with contextlib.redirect_stdout(io.StringIO()):
                report = evaluator.evaluate(results, splits, False)
                with self.assertRaises(RuntimeError):
                    evaluator.evaluate(results, splits, True)
            self.assertEqual(report["total_matched"], 2)
            self.assertEqual(report["total_test"], 3)
            self.assertEqual(report["confusion_matrix"], [[1, 0], [0, 1]])
            self.assertEqual(report["missing_test_video_ids"], ["unknown"])
            self.assertEqual(len(report["recovery_diagnostics"]["changed_predictions"]), 1)


if __name__ == "__main__":
    unittest.main()
