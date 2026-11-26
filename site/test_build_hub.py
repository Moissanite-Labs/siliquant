#!/usr/bin/env python3
"""Tests for site/build_hub.py. Stdlib unittest only, no network.

Each test copies the repo parts the builder reads into a temp dir and
runs the builder there with --root, except the committed-docs check,
which runs against the real repo. Run with:

    python3 -I -B site/test_build_hub.py
    python3 -I -B -W error site/test_build_hub.py
"""

import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from html.parser import HTMLParser

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILDER = os.path.join(REPO_ROOT, "site", "build_hub.py")


def run_builder(*args):
    return subprocess.run([sys.executable, "-I", "-B", BUILDER] + list(args),
                          capture_output=True, text=True, cwd=REPO_ROOT)


def copy_repo_parts(dst):
    os.makedirs(os.path.join(dst, "site"))
    for name in ("build_hub.py", "hub_template.html"):
        shutil.copy2(os.path.join(REPO_ROOT, "site", name),
                     os.path.join(dst, "site", name))
    os.makedirs(os.path.join(dst, "docs"))
    for name in ("index.html", "data.json", "hub.html"):
        src = os.path.join(REPO_ROOT, "docs", name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(dst, "docs", name))
    shutil.copytree(os.path.join(REPO_ROOT, "experiments"),
                    os.path.join(dst, "experiments"))
    shutil.copytree(os.path.join(REPO_ROOT, "studies"),
                    os.path.join(dst, "studies"))


def read_text(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def read_bytes(path):
    with open(path, "rb") as fh:
        return fh.read()


class IdCollector(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.ids = []

    def handle_starttag(self, tag, attrs):
        for key, value in attrs:
            if key == "id" and value:
                self.ids.append(value)


class BuildHubTests(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="hub-test-")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        copy_repo_parts(self.tmp)

    def build(self, *extra):
        return run_builder("--root", self.tmp, *extra)

    def docs_path(self, name):
        return os.path.join(self.tmp, "docs", name)

    def test_01_build_is_deterministic(self):
        first = self.build()
        self.assertEqual(first.returncode, 0, first.stderr)
        data1 = read_bytes(self.docs_path("data.json"))
        html1 = read_bytes(self.docs_path("hub.html"))
        second = self.build()
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertEqual(data1, read_bytes(self.docs_path("data.json")))
        self.assertEqual(html1, read_bytes(self.docs_path("hub.html")))

    def test_02_check_passes_on_committed_docs(self):
        result = run_builder("--check")
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_03_check_fails_when_hub_html_is_hand_edited(self):
        built = self.build()
        self.assertEqual(built.returncode, 0, built.stderr)
        with open(self.docs_path("hub.html"), "a", encoding="utf-8") as fh:
            fh.write("<!-- hand edit -->\n")
        result = self.build("--check")
        self.assertEqual(result.returncode, 1)
        self.assertIn("hub.html", result.stderr)

    def test_04_check_fails_when_data_json_is_hand_edited(self):
        built = self.build()
        self.assertEqual(built.returncode, 0, built.stderr)
        with open(self.docs_path("data.json"), "a", encoding="utf-8") as fh:
            fh.write("\n")
        result = self.build("--check")
        self.assertEqual(result.returncode, 1)
        self.assertIn("data.json", result.stderr)

    def test_05_build_fails_when_an_index_gemma_number_is_changed(self):
        path = self.docs_path("index.html")
        text = read_text(path)
        self.assertEqual(text.count("4.4137"), 1)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text.replace("4.4137", "4.4138"))
        result = self.build()
        self.assertEqual(result.returncode, 1)
        self.assertIn("gemma-impact", result.stderr)

    def test_06_build_fails_when_an_impact_table_row_is_deleted(self):
        path = os.path.join(
            self.tmp, "experiments/model-output-gemma-ball-2026-11-23/reports/impact",
            "gemma-3-270m-it/IMPACT_TABLES.md")
        text = read_text(path)
        needle = "| Real matrix / file:Q4_K_M-imat | 4.4137 |"
        self.assertEqual(text.count(needle), 1)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text.replace(needle + " 1.0201 | 18.77% | interpolated |\n", ""))
        result = self.build()
        self.assertEqual(result.returncode, 1)

    def test_07_generated_hub_html_loads_nothing_external(self):
        built = self.build()
        self.assertEqual(built.returncode, 0, built.stderr)
        page = read_text(self.docs_path("hub.html"))
        self.assertNotIn("<script", page)
        self.assertIsNone(re.search(r"""src\s*=\s*["']https?://""", page))
        self.assertNotIn("{{", page)
        self.assertIn('href="data.json"', page)

    def test_08_generated_hub_html_parses_with_unique_ids(self):
        built = self.build()
        self.assertEqual(built.returncode, 0, built.stderr)
        parser = IdCollector()
        parser.feed(read_text(self.docs_path("hub.html")))
        parser.close()
        self.assertEqual(len(parser.ids), len(set(parser.ids)),
                         "duplicate ids: %r" % parser.ids)

    def test_09_generated_docs_contain_no_throughput_or_banned_text(self):
        built = self.build()
        self.assertEqual(built.returncode, 0, built.stderr)
        for name in ("hub.html", "data.json"):
            text = read_text(self.docs_path(name))
            for token in ("tok/s", "GB/s", "Gw/s", "1000+", "drop-in"):
                self.assertNotIn(token, text, "%s contains %r" % (name, token))


if __name__ == "__main__":
    unittest.main()
