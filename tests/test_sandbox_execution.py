"""Smoke-test sandbox execution with generated analytics code."""

import unittest

from tools.script_executor.sandbox import run_script_safely


class SandboxExecutionTests(unittest.TestCase):
    def test_executes_matplotlib_and_pandas_code(self):
        code = """
import io
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

df = pd.DataFrame({'x': [1, 2, 3], 'y': [2, 4, 6]})
print(df['y'].mean())
buf = io.BytesIO()
plt.plot(df['x'], df['y'])
plt.savefig(buf, format='png')
print('ok')
"""
        result = run_script_safely(code, timeout=60)
        self.assertEqual(result["returncode"], 0, msg=result["stderr"])
        self.assertIn("4.0", result["stdout"])
        self.assertIn("ok", result["stdout"])


if __name__ == "__main__":
    unittest.main()
