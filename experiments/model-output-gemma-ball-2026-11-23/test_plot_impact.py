"""Synthetic controls for the impact chart CLI."""

from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import plot_impact as plot
from test_analyze import a, metrics_only_row, synthetic_row


class PlotImpactTests(unittest.TestCase):
    def setUp(self):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.addCleanup(plot.plt.close, "all")
        self.raw, self.chart = Path(tmp.name) / "synthetic.jsonl", Path(tmp.name) / "chart.png"

    def plot(self, data):
        self.raw.write_bytes(data)
        with patch("sys.argv", ["plot_impact.py", str(self.raw), "--model", "gemma-3-270m-it",
                                "--out", str(self.chart)]):
            plot.main()

    def test_refuses_sweep_without_plottable_points(self):
        for data in (metrics_only_row(), synthetic_row(variant=a.HADAMARD)):
            err = StringIO()
            with self.subTest(data=data), redirect_stderr(err), self.assertRaises(SystemExit) as exit_info:
                self.plot(data)
            self.assertEqual(exit_info.exception.code, 2)
            self.assertIn("no plottable KL measurements for model", err.getvalue())
            self.assertFalse(self.chart.exists())


if __name__ == "__main__":
    unittest.main()
