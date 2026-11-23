"""Exercise the documented receipt extraction against an unfetched synthetic commit."""

from os import environ
from pathlib import Path
from subprocess import run
from tempfile import TemporaryDirectory
import unittest


class ReceiptTests(unittest.TestCase):
    def test_fetch_recipe_extracts_exact_blob_bytes(self):
        experiment = Path(__file__).resolve().parent
        receipts = (experiment / "reports/impact/RESULTS_RECEIPTS.md").read_text()
        recipe = receipts.split("```sh\n", 1)[1].split("```", 1)[0]
        with TemporaryDirectory() as tmp:
            source, replay = Path(tmp) / "source", Path(tmp) / "replay"
            for repo in (source, replay):
                run(["git", "init", "-q", repo], check=True)

            def git(repo, *args, **kwargs):
                return run(["git", *args], cwd=repo, check=True, capture_output=True, **kwargs).stdout

            for model, raw in (("270m", b'{"synthetic":270}\r\n'), ("1b", b'{"synthetic":1000}')):
                path = source / experiment.name / "out" / f"gemma-3-{model}-it" / "results.jsonl"
                path.parent.mkdir(parents=True)
                path.write_bytes(raw)
            git(source, "add", "--", experiment.name)
            tree = git(source, "write-tree").decode().strip()
            identity = {f"GIT_{role}_{field}": value for role in ("AUTHOR", "COMMITTER")
                        for field, value in (("NAME", "Fixture"), ("EMAIL", "fixture@example.invalid"))}
            pin = git(source, "commit-tree", tree, "-m", "Synthetic results",
                      env=environ | identity).decode().strip()
            git(source, "update-ref", "refs/heads/results", pin)
            git(replay, "remote", "add", "origin", str(source))
            pinned = recipe.splitlines()[0].removeprefix("pin=")
            run(["sh", "-c", recipe.replace(pinned, pin)], cwd=replay, check=True, capture_output=True)
            self.assertEqual((replay / "270m-results.jsonl").read_bytes(), b'{"synthetic":270}\r\n')
            self.assertEqual((replay / "1b-results.jsonl").read_bytes(), b'{"synthetic":1000}')


if __name__ == "__main__":
    unittest.main()
