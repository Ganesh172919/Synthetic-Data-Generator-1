import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class MockGeneratorTest(unittest.TestCase):
    def test_mock_cli_without_torch(self):
        generator = Path(__file__).resolve().parents[1] / "Pre-Work" / "universal_dataset_generator.py"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "dataset.jsonl"
            config = root / "job.json"
            config.write_text(json.dumps({
                "prompt": "Generate simple software engineering Q&A pairs.",
                "provider": "mock", "parseMode": "qa", "targetCount": 100,
                "batchSize": 10, "outputFormat": "jsonl", "outputPath": str(output),
                "checkpointPath": str(root / "checkpoint.json"),
                "autoSaveSeconds": 2, "saveInterval": 10,
            }), encoding="utf-8")
            environment = dict(os.environ, SYNTHGEN_AUTO_INSTALL="0", PYTHONIOENCODING="utf-8")
            result = subprocess.run([
                sys.executable, "-c",
                "import runpy, sys; sys.modules['torch'] = None; "
                "sys.argv = sys.argv[1:]; runpy.run_path(sys.argv[0], run_name='__main__')",
                str(generator), "--config", str(config),
            ], cwd=root, env=environment, capture_output=True, text=True, encoding="utf-8", timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(records), 100)
            self.assertEqual(len({record["id"] for record in records}), 100)
            self.assertTrue(all(record["question"] and record["answer"] for record in records))


if __name__ == "__main__":
    unittest.main()
