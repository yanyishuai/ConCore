"""Verify CoTAS script executor can import common data-science libraries."""

import importlib
import unittest

REQUIRED_PACKAGES = (
    "matplotlib",
    "seaborn",
    "numpy",
    "scipy",
    "sklearn",
    "statsmodels",
    "plotly",
    "pandas",
)


class DataScienceImportTests(unittest.TestCase):
    def test_required_packages_import(self):
        for name in REQUIRED_PACKAGES:
            with self.subTest(package=name):
                module = importlib.import_module(name)
                self.assertIsNotNone(module)

    def test_matplotlib_can_render_noninteractive(self):
        matplotlib = importlib.import_module("matplotlib")
        matplotlib.use("Agg")
        pyplot = importlib.import_module("matplotlib.pyplot")
        pyplot.plot([1, 2, 3], [1, 4, 9])
        pyplot.close()

    def test_sklearn_simple_fit(self):
        sklearn_datasets = importlib.import_module("sklearn.datasets")
        sklearn_linear = importlib.import_module("sklearn.linear_model")
        x, y = sklearn_datasets.make_regression(n_samples=20, n_features=1, noise=0.1, random_state=0)
        model = sklearn_linear.LinearRegression().fit(x, y)
        self.assertAlmostEqual(model.score(x, y), 1.0, places=1)


if __name__ == "__main__":
    unittest.main()
