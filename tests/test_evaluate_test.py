import io
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from evaluate_test import DEFAULT_MODEL_PATHS, evaluate_variants


class FakeModel:
    def __init__(self):
        self.val_calls = []

    def val(self, **kwargs):
        self.val_calls.append(kwargs)
        print("fake validation output")
        return {"metrics/mAP50(B)": 0.123}


class TestEvaluateTest(unittest.TestCase):
    def test_evaluates_three_checkpoints_on_test_split_and_logs_output(self):
        loaded_paths = []
        models = []

        def fake_loader(path):
            loaded_paths.append(path)
            model = FakeModel()
            models.append(model)
            return model

        with tempfile.TemporaryDirectory() as temporary_directory:
            log_directory = Path(temporary_directory)
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                results = evaluate_variants(
                    model_paths=DEFAULT_MODEL_PATHS,
                    data="VisDrone.yaml",
                    batch=4,
                    imgsz=640,
                    device="0",
                    project="runs/s-yolov11-test",
                    log_directory=log_directory,
                    model_loader=fake_loader,
                )

            self.assertEqual(list(DEFAULT_MODEL_PATHS), [
                "ghostv3",
                "dwconv",
                "ghostv3_dwconv",
            ])
            self.assertEqual(len(results), 3)
            self.assertEqual(len(loaded_paths), 3)
            for model in models:
                self.assertEqual(len(model.val_calls), 1)
                self.assertEqual(model.val_calls[0]["split"], "test")
                self.assertEqual(model.val_calls[0]["data"], "VisDrone.yaml")

            for name in DEFAULT_MODEL_PATHS:
                model_log = log_directory / f"{name}.log"
                self.assertTrue(model_log.exists())
                log_text = model_log.read_text(encoding="utf-8")
                self.assertIn("split=test", log_text)
                self.assertIn("fake validation output", log_text)

            self.assertTrue((log_directory / "queue.log").exists())


if __name__ == "__main__":
    unittest.main()
