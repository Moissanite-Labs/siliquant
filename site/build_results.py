"""Build the archived GitHub Pages report and its print edition."""

import argparse
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chrome", default="google-chrome", help="Chrome or Chromium executable")
    args = parser.parse_args()
    source = Path(__file__).resolve().with_name("results.html")
    docs = source.parents[1] / "docs"
    archive = docs / "archive" / "results-2026-11-23.html"
    build = source.parent / ".build"
    build.mkdir(exist_ok=True)
    with TemporaryDirectory(prefix="siliquant-results-", dir=build) as work:
        work = Path(work)
        pdf = work / "siliquant-results.pdf"
        subprocess.run(
            [
                args.chrome,
                "--headless",
                f"--user-data-dir={work / 'chrome'}",
                "--no-pdf-header-footer",
                f"--print-to-pdf={pdf}",
                source.as_uri(),
            ],
            check=True,
        )
        archive.parent.mkdir(exist_ok=True)
        shutil.copyfile(pdf, docs / pdf.name)
        shutil.copyfile(source, archive)
    print("Built docs/archive/results-2026-11-23.html and docs/siliquant-results.pdf")


if __name__ == "__main__":
    main()
